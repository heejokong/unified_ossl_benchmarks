import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
from copy import deepcopy

from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from skimage.filters import threshold_otsu

from semilearn.core import AlgorithmBase
from semilearn.core.utils import get_data_loader
from semilearn.algorithms.utils import ce_loss, consistency_loss, SSL_Argument, str2bool, interleave, mixup_one_target

from semilearn.core.utils import ALGORITHMS


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class WrappedDataset(Dataset):
    def __init__(self,
                 lb_dset,
                 ulb_dset):
        super(WrappedDataset, self).__init__()
        self.transform = lb_dset.transform

        self.data_x = deepcopy(lb_dset.data)
        self.targets_x = deepcopy(lb_dset.targets)

        self.data_u = deepcopy(ulb_dset.data)
        self.targets_u = deepcopy(ulb_dset.targets)

        self.id_scores = np.zeros(len(self.data_x) + len(self.data_u), dtype=np.float32)
        self.id_scores[:len(self.data_x)] = 1.0
        # self.soft_labels[len(self.data_x):] = 0.0

        self.prediction = np.zeros((len(self.data_x) + len(self.data_u), 5), dtype=np.float32)
        self.prediction[:len(self.data_x), :] = 1.0
        self.count = 0

    def score_update(self, results):
        self.count += 1

        # While updating the noisy label y_i by the probability s,
        # we used the average output probability of the network of the past 10 epochs as s.
        idx = (self.count - 1) % 5
        self.prediction[len(self.data_x):, idx] = results[len(self.data_x):]

        if self.count >= 5:
            self.id_scores = self.prediction.mean(axis=1)

    def __len__(self):
        return len(self.data_x) + len(self.data_u)

    def __getitem__(self, idx):
        if idx < len(self.data_x):
            img, target = self.data_x[idx], self.targets_x[idx]
        else:
            img, target = self.data_u[idx - len(self.data_x)], self.targets_u[idx - len(self.data_x)]

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        if isinstance(img, str):
            img = pil_loader(img)

        if self.transform is not None:
            img = self.transform(img)

        return {'idx': idx, 'x': img, 'y': target, 'y_d': self.id_scores[idx]}


class MTCNet(nn.Module):
    def __init__(self, base):
        super(MTCNet, self).__init__()
        self.backbone = base
        self.feat_planes = base.num_features

        self.domain_classifier = nn.Linear(self.feat_planes, 1)

        nn.init.xavier_normal_(self.domain_classifier.weight.data)
        self.domain_classifier.bias.data.zero_()

    def forward(self, x, **kwargs):
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)
        domain_logits = self.domain_classifier(feat)

        return {'feat': feat, 'logits': logits, 'domain_logits': domain_logits}


@ALGORITHMS.register('mtc')
class MTC(AlgorithmBase):
    """
        MTC algorithm (https://arxiv.org/abs/2007.11330).
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # mtc specified arguments
        self.pre_epochs = args.pre_epoch
        self.T = args.T
        self.unsup_warm_up = args.unsup_warm_up
        self.mixup_alpha = args.mixup_alpha
        self.mixup_manifold = args.mixup_manifold

        self.results = np.zeros(len(self.loader_dict['train_wrapped']), dtype=np.float32)

    def set_model(self):
        model = super().set_model()
        model = MTCNet(model)
        return model

    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model = MTCNet(ema_model)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def set_data_loader(self):
        loader_dict = super().set_data_loader()

        wrapped_dset = WrappedDataset(self.dataset_dict['train_lb'], self.dataset_dict['train_ulb'])
        loader_dict['train_wrapped'] = get_data_loader(self.args,
                                                       wrapped_dset,
                                                       self.args.batch_size,
                                                       data_sampler=self.args.train_sampler,
                                                       num_iters=self.num_train_iter,
                                                       num_epochs=self.epochs,
                                                       num_workers=self.args.num_workers,
                                                       distributed=self.distributed)
        return loader_dict

    def select_id_samples(self):
        id_scores = self.loader_dict['train_wrapped'].dataset.id_scores[self.args.lb_dest_len:].copy()
        th = threshold_otsu(id_scores.reshape(-1, 1))
        self.print_fn(f"Selecting ID samples with threshold {th:.3f}")

        selected_idx = np.arange(self.args.ulb_dest_len)[id_scores >= th]
        selected_dset = deepcopy(self.dataset_dict['train_ulb'])
        selected_dset.data = selected_dset.data[selected_idx]
        selected_dset.targets = selected_dset.targets[selected_idx]

        num_selected = len(selected_dset.targets)
        num_true_id = (selected_dset.targets < self.num_classes).sum()
        num_all_id = (self.dataset_dict['train_ulb'].targets < self.num_classes).sum()

        prec = num_true_id / num_selected
        recall = num_true_id / num_all_id
        ratio = num_selected / len(self.dataset_dict['train_ulb'])

        if self.rank == 0:
            self.print_fn(f"Selected ratio = {ratio:.3f}, "
                          f"precision = {prec:.3f}, recall: {recall:.3f}")

        self.loader_dict['train_ulb_selected'] = get_data_loader(self.args,
                                                                 selected_dset,
                                                                 self.args.batch_size * self.args.uratio,
                                                                 data_sampler=self.args.train_sampler,
                                                                 num_iters=self.num_train_iter // self.epochs,
                                                                 num_epochs=1,
                                                                 num_workers=2 * self.args.num_workers,
                                                                 distributed=self.distributed)

    def evaluate_open(self):
        self.model.eval()
        self.ema.apply_shadow()

        id_scores = self.loader_dict['train_wrapped'].dataset.id_scores[self.args.lb_dest_len:].copy()
        thres = threshold_otsu(id_scores.reshape(-1, 1))

        full_loader = self.loader_dict['test']['full']
        extended_loader = self.loader_dict['test']['extended']

        total_num = 0.0
        y_true_list = []
        pred_p_list = []
        pred_hat_q_list = []

        results = {}
        with torch.no_grad():
            id_scores = []
            for data in full_loader:
                x = data['x_lb']
                y = data['y_lb']

                if isinstance(x, dict):
                    x = {k: v.cuda() for k, v in x.items()}
                else:
                    x = x.cuda()
                y = y.cuda()
                y_true_list.extend(y.cpu().tolist())

                num_batch = y.shape[0]
                total_num += num_batch

                outputs = self.model(x)
                logits = outputs['logits']
                domain_logits = outputs['domain_logits']

                p = F.softmax(logits, 1)
                pred_p = p.data.max(1)[1]
                pred_p_list.extend(pred_p.cpu().tolist())

                id_score = torch.sigmoid(domain_logits).view(-1)
                id_scores.append(id_score.cpu().numpy())
                pred_hat_q = pred_p.clone()
                pred_hat_q[id_score < thres] = self.num_classes
                pred_hat_q_list.extend(pred_hat_q.cpu().tolist())

            y_true = np.array(y_true_list)
            closed_mask = y_true < self.num_classes
            open_mask = y_true >= self.num_classes
            y_true[open_mask] = self.num_classes

            results['scores_full'] = np.concatenate(id_scores)
            results['scores_id'] = results['scores_full'][closed_mask]
            results['scores_ood'] = results['scores_full'][open_mask]

            pred_p = np.array(pred_p_list)
            pred_hat_q = np.array(pred_hat_q_list)

            c_acc_c_p = accuracy_score(y_true[closed_mask], pred_p[closed_mask])
            o_acc_c_hq = accuracy_score(y_true[closed_mask], pred_hat_q[closed_mask])
            o_acc_f_hq = balanced_accuracy_score(y_true, pred_hat_q)

            id_scores = []
            for data in extended_loader:
                x = data['x_lb']
                y = data['y_lb']

                if isinstance(x, dict):
                    x = {k: v.cuda() for k, v in x.items()}
                else:
                    x = x.cuda()
                y = y.cuda()
                y_true_list.extend(y.cpu().tolist())

                num_batch = y.shape[0]
                total_num += num_batch

                outputs = self.model(x)
                logits = outputs['logits']
                domain_logits = outputs['domain_logits']

                p = F.softmax(logits, 1)
                pred_p = p.data.max(1)[1]
                pred_p_list.extend(pred_p.cpu().tolist())

                id_score = torch.sigmoid(domain_logits).view(-1)
                id_scores.append(id_score.cpu().numpy())
                pred_hat_q = pred_p.clone()
                pred_hat_q[id_score < thres] = self.num_classes
                pred_hat_q_list.extend(pred_hat_q.cpu().tolist())

            y_true = np.array(y_true_list)
            open_mask = y_true >= self.num_classes
            y_true[open_mask] = self.num_classes

            pred_hat_q = np.array(pred_hat_q_list)

            results['scores_extended'] = np.concatenate(id_scores)

            o_acc_e_hq = balanced_accuracy_score(y_true, pred_hat_q)

            self.ema.restore()
            self.model.train()

            self.print_fn(f"\n#############################################################\n"
                          f" Test Results of Model in Epoch {self.epoch}\n"
                          f" Closed Accuracy on Closed Test Data (p) : {c_acc_c_p * 100:.2f}\n"
                          f" Open Accuracy on Closed Test Data (hq)  : {o_acc_c_hq * 100:.2f}\n"
                          f" Open Accuracy on Full Test Data (hq)    : {o_acc_f_hq * 100:.2f}\n"
                          f" Open Accuracy on Extended Test Data (hq): {o_acc_e_hq * 100:.2f}\n"
                          f"#############################################################\n"
                          )

            return results

    def train(self):
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch

            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break

            self.call_hook("before_train_epoch")

            if self.epoch < self.pre_epochs:
                # warmup stage
                self.results = np.zeros(len(self.loader_dict['train_wrapped'].dataset), dtype=np.float32)
                self.select_id_samples()
                for data_wrapped in self.loader_dict['train_wrapped']:
                    self.call_hook("before_train_step")
                    self.tb_dict = self.warmup_step(**self.process_batch(**data_wrapped))
                    self.call_hook("after_train_step")
                    self.it += 1
                self.loader_dict['train_wrapped'].dataset.score_update(self.results)
            else:
                # train stage
                self.results = np.zeros(len(self.loader_dict['train_wrapped'].dataset), dtype=np.float32)
                self.select_id_samples()
                for data_lb, data_ulb_selected, data_wrapped in zip(self.loader_dict['train_lb'],
                                                                    self.loader_dict['train_ulb_selected'],
                                                                    self.loader_dict['train_wrapped']):
                    # prevent the training iterations exceed args.num_train_iter
                    if self.it >= self.num_train_iter:
                        break

                    self.call_hook("before_train_step")
                    self.tb_dict = self.train_step(**self.process_batch(**data_lb,
                                                                        **data_ulb_selected,
                                                                        **data_wrapped))
                    self.call_hook("after_train_step")
                    self.it += 1
                self.loader_dict['train_wrapped'].dataset.score_update(self.results)
            self.call_hook("after_train_epoch")

        self.call_hook("after_run")

    def warmup_step(self, idx, x, y_d):
        domain_logits = self.model(x)['domain_logits']
        probs = torch.sigmoid(domain_logits).view(-1)
        domain_loss = F.binary_cross_entropy_with_logits(domain_logits, y_d.view(-1, 1))

        self.results[idx.cpu().detach().numpy().tolist()] = probs.cpu().detach().numpy().tolist()

        total_loss = domain_loss

        # parameter updates
        self.call_hook("param_update", "ParamUpdateHook", loss=total_loss)

        tb_dict = {}
        tb_dict['train/domain_loss'] = domain_loss.item()
        tb_dict['train/total_loss'] = total_loss.item()
        return tb_dict

    def train_step(self, x_lb, y_lb, x_ulb_w_0, x_ulb_w_1, idx, x, y_d):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            with torch.no_grad():
                self.bn_controller.freeze_bn(self.model)
                outs_x_ulb_w0 = self.model(x_ulb_w_0)
                logits_x_ulb_w0 = outs_x_ulb_w0['logits']
                outs_x_ulb_w1 = self.model(x_ulb_w_1)
                logits_x_ulb_w1 = outs_x_ulb_w1['logits']
                self.bn_controller.unfreeze_bn(self.model)

                # avg
                avg_prob_x_ulb = (torch.softmax(logits_x_ulb_w0, dim=1) + torch.softmax(logits_x_ulb_w1, dim=1)) / 2
                # avg_prob_x_ulb = (avg_prob_x_ulb / avg_prob_x_ulb.sum(dim=-1, keepdim=True))
                # sharpening
                sharpen_prob_x_ulb = avg_prob_x_ulb ** (1 / self.T)
                sharpen_prob_x_ulb = (sharpen_prob_x_ulb / sharpen_prob_x_ulb.sum(dim=-1, keepdim=True)).detach()

            self.bn_controller.freeze_bn(self.model)
            domain_logits = self.model(x)['domain_logits']
            self.bn_controller.unfreeze_bn(self.model)
            probs = torch.sigmoid(domain_logits).view(-1)
            domain_loss = F.binary_cross_entropy_with_logits(domain_logits, y_d.view(-1, 1))
            self.results[idx.cpu().detach().numpy().tolist()] = probs.cpu().detach().numpy().tolist()

            # with torch.no_grad():
            # Pseudo Label
            input_labels = torch.cat([F.one_hot(y_lb, self.num_classes), sharpen_prob_x_ulb, sharpen_prob_x_ulb], dim=0)
            # Mix up
            if self.mixup_manifold:
                outs_x_lb = self.model(x_lb)
                inputs = torch.cat((outs_x_lb['feat'], outs_x_ulb_w0['feat'], outs_x_ulb_w1['feat']))
            else:
                inputs = torch.cat([x_lb, x_ulb_w_0, x_ulb_w_1])
            mixed_x, mixed_y, _ = mixup_one_target(inputs, input_labels,
                                                   self.mixup_alpha,
                                                   is_bias=True)
            mixed_x = list(torch.split(mixed_x, num_lb))
            mixed_x = interleave(mixed_x, num_lb)

            if self.mixup_manifold:
                logits = [self.model(mixed_x[0], only_fc=self.mixup_manifold)]
                # calculate BN for only the first batch
                self.bn_controller.freeze_bn(self.model)
                for ipt in mixed_x[1:]:
                    logits.append(self.model(ipt, only_fc=self.mixup_manifold))
                self.bn_controller.unfreeze_bn(self.model)
            else:
                logits = [self.model(mixed_x[0])['logits']]
                # calculate BN for only the first batch
                self.bn_controller.freeze_bn(self.model)
                for ipt in mixed_x[1:]:
                    logits.append(self.model(ipt)['logits'])
                self.bn_controller.unfreeze_bn(self.model)

            # put interleaved samples back
            logits = interleave(logits, num_lb)

            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)

            sup_loss = ce_loss(logits_x, mixed_y[:num_lb], reduction='mean')
            unsup_loss = consistency_loss(logits_u, mixed_y[num_lb:], name='mse')

            # set ramp_up for lambda_u
            unsup_warmup = float(np.clip(self.it / (self.unsup_warm_up * self.num_train_iter), 0.0, 1.0))
            lambda_u = self.lambda_u * unsup_warmup

            total_loss = sup_loss + domain_loss + lambda_u * unsup_loss

        # parameter updates
        self.call_hook("param_update", "ParamUpdateHook", loss=total_loss)

        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        tb_dict['train/unsup_loss'] = unsup_loss.item()
        tb_dict['train/domain_loss'] = domain_loss.item()
        tb_dict['train/total_loss'] = total_loss.item()
        return tb_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        wrapped_dset = self.loader_dict['train_wrapped'].dataset
        save_dict['score_update_count'] = wrapped_dset.count
        save_dict['id_scores'] = wrapped_dset.id_scores
        save_dict['prediction'] = wrapped_dset.prediction
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        wrapped_dset = self.loader_dict['train_wrapped'].dataset
        wrapped_dset.count = checkpoint['score_update_count']
        wrapped_dset.id_scores = checkpoint['id_scores']
        wrapped_dset.prediction = checkpoint['prediction']
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--pre_epoch', int, 25, 'number of pre-training epochs'),
            SSL_Argument('--T', float, 0.5, 'parameter for Temperature Sharpening'),
            SSL_Argument('--unsup_warm_up', float, 1 / 64, 'ramp up ratio for unsupervised loss'),
            SSL_Argument('--mixup_alpha', float, 0.5, 'parameter for Beta distribution of Mix Up'),
            SSL_Argument('--mixup_manifold', str2bool, False, 'use manifold mixup (for nlp)'),
        ]
