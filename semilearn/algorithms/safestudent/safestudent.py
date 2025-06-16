import copy
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from sklearn.metrics import accuracy_score, precision_score, recall_score

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from semilearn.core import AlgorithmBase
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset
from semilearn.algorithms.utils import ce_loss, consistency_loss, SSL_Argument
from semilearn.core.utils import EMA, get_data_loader
from semilearn.core.hooks import Hook, get_priority, CheckpointHook, TimerHook, LoggingHook, DistSamplerSeedHook, \
    ParamUpdateHook, EvaluationHook, EMAHook

from semilearn.core.utils import ALGORITHMS


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class SafeStudentDataset(BasicDataset):
    def __init__(self, dset, name):
        self.data = copy.deepcopy(dset.data)
        self.targets = copy.deepcopy(dset.targets)
        super(SafeStudentDataset, self).__init__(alg='t2t',
                                                 data=self.data,
                                                 targets=self.targets,
                                                 num_classes=dset.num_classes,
                                                 transform=dset.transform,
                                                 strong_transform=dset.strong_transform)
        self.name = name
        self.data_index = None
        self.targets_index = None
        self.max_uc_score = 0.
        self.scores = np.zeros(len(self.data))
        self.scores_index = None
        self.set_index()

    def update_scores(self, indices, scores, max_uc_score):
        self.scores[indices] = scores
        self.max_uc_score = max_uc_score

    def set_index(self, indices=None):
        if indices is not None:
            self.data_index = self.data[indices]
            self.targets_index = self.targets[indices]
            self.scores_index = self.scores[indices]
        else:
            self.data_index = self.data
            self.targets_index = self.targets
            self.scores_index = self.scores

    def __len__(self):
        return len(self.data_index)

    def __sample__(self, idx):
        if self.targets is None:
            target = None
        else:
            target = self.targets_index[idx]
        img = self.data_index[idx]
        score = self.scores_index[idx]

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)  # shape of img should be [H, W, C]
        if isinstance(img, str):
            img = pil_loader(img)

        return img, target, score

    def __getitem__(self, idx):
        img, target, score = self.__sample__(idx)

        img_w = self.transform(img)
        if self.name == 'train_sc':
            # For rotation recognition pretext task
            return {'x_sc_w': img_w, 'x_sc_s': self.strong_transform(img)}
        elif self.name == 'train_uc':
            weight = np.exp(1 - score / self.max_uc_score)
            return {'x_uc_s': self.strong_transform(img), 'w_uc': weight}


def calculate_ED(logits):
    return torch.logsumexp(logits, dim=1) * (1 - 1 / torch.exp(logits).max(dim=1)[0])


@ALGORITHMS.register('safestudent')
class SafeStudent(AlgorithmBase):
    """
        SafeStudent algorithm (https://arxiv.org/abs/1703.01780).
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        super().__init__(args, net_builder, tb_log, logger, **kwargs)
        # mean teacher specificed arguments
        self.pre_epochs = args.pre_epoch
        self.thres1 = args.thres1
        self.thres2 = args.thres2
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2

    def set_hooks(self):
        """
        register necessary training hooks
        """
        # parameter update hook is called inside each train_step
        self.register_hook(TimerHook(), None, "HIGHEST")
        # self.register_hook(EMAHook(), None, "HIGHEST")
        self.register_hook(EvaluationHook(), None, "HIGHEST")
        self.register_hook(CheckpointHook(), None, "VERY_HIGH")
        self.register_hook(DistSamplerSeedHook(), None, "HIGH")
        self.register_hook(LoggingHook(), None, "LOW")

        # for hooks to be called in train_step, name it for simpler calling
        self.register_hook(ParamUpdateHook(), "ParamUpdateHook")

    def set_dataset(self):
        dataset_dict = super(SafeStudent, self).set_dataset()
        dataset_dict['train_sc'] = SafeStudentDataset(dset=dataset_dict['train_ulb'], name='train_sc')
        dataset_dict['train_uc'] = SafeStudentDataset(dset=dataset_dict['train_ulb'], name='train_uc')
        return dataset_dict

    def train(self):
        self.model.train()

        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if self.resume:
            self.ema.load(self.ema_model)

        self.call_hook("before_run")

        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch

            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break

            self.call_hook("before_train_epoch")
            if self.epoch < self.pre_epochs:
                for data_lb in self.loader_dict['train_lb']:
                    self.call_hook("before_train_step")
                    self.tb_dict = self.warmup_step(**self.process_batch(**data_lb))
                    self.ema.load(self.model)
                    self.ema_model.load_state_dict(self.model.state_dict())
                    self.call_hook("after_train_step")
                    self.it += 1
            else:
                if self.epoch == self.pre_epochs:
                    self.ema.register()
                self.select_samples()
                self.loader_dict['train_sc'] = get_data_loader(self.args,
                                                               self.dataset_dict['train_sc'],
                                                               self.args.batch_size * self.args.uratio,
                                                               data_sampler=self.args.train_sampler,
                                                               num_iters=self.num_train_iter,
                                                               num_epochs=self.epochs,
                                                               num_workers=2 * self.args.num_workers,
                                                               distributed=self.distributed)
                self.loader_dict['train_uc'] = get_data_loader(self.args,
                                                               self.dataset_dict['train_uc'],
                                                               self.args.batch_size * self.args.uratio,
                                                               data_sampler=self.args.train_sampler,
                                                               num_iters=self.num_train_iter,
                                                               num_epochs=self.epochs,
                                                               num_workers=2 * self.args.num_workers,
                                                               distributed=self.distributed)

                for data_lb, data_sc, data_uc in zip(self.loader_dict['train_lb'],
                                                     self.loader_dict['train_sc'],
                                                     self.loader_dict['train_uc'], ):
                    # prevent the training iterations exceed args.num_train_iter
                    if self.it >= self.num_train_iter:
                        break

                    self.call_hook("before_train_step")
                    self.tb_dict = self.train_step(**self.process_batch(**data_lb,
                                                                        **data_sc,
                                                                        **data_uc))
                    self.ema.update()
                    self.ema_model.load_state_dict(self.model.state_dict())
                    self.ema_model.load_state_dict(self.ema.shadow, strict=False)
                    self.call_hook("after_train_step")
                    self.it += 1

    def select_samples(self):
        loader = DataLoader(dataset=self.dataset_dict['train_ulb'],
                            batch_size=self.args.eval_batch_size,
                            drop_last=False,
                            shuffle=False,
                            num_workers=4)
        self.model.eval()
        self.ema.apply_shadow()
        self.print_fn(f"Selecting samples...")
        indices = []
        ED_scores = []
        targets = []

        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                x = data['x_ulb_w'].to(self.gpu)
                y = data['y_ulb']
                idx = data['idx_ulb']

                output = self.model(x)
                logits = output['logits']
                ED_score = calculate_ED(logits).cpu()
                ED_scores.append(ED_score)
                indices.append(idx)
                targets.append(y)
            indices = torch.cat(indices).numpy()
            targets = torch.cat(targets).numpy()
            ED_scores = torch.cat(ED_scores).cpu()

        total_num = len(indices)

        sorted_indices = torch.argsort(ED_scores)
        sc_thres = ED_scores[sorted_indices[int(total_num * self.thres1)]].item()
        uc_thres = ED_scores[sorted_indices[int(total_num * self.thres2)]].item()

        sorted_indices = sorted_indices.numpy()
        sc_indices = indices[sorted_indices[int(total_num * self.thres1):]]
        uc_indices = indices[sorted_indices[:int(total_num * self.thres2)]]

        ED_scores = ED_scores.numpy()
        num_true_sc = (targets[sc_indices] < self.num_classes).sum()
        num_true_uc = (targets[uc_indices] >= self.num_classes).sum()

        if self.rank == 0:
            self.print_fn(f"Thres1 = {sc_thres}, Thres2 = {uc_thres} ")
            self.print_fn(f"Number of true SC in D_SC = {num_true_sc}, "
                          f"number of true UC in D_UC = {num_true_uc} ")

        self.ema.restore()
        self.model.train()

        if self.epoch >= self.pre_epochs:
            self.dataset_dict['train_sc'].update_scores(indices, ED_scores, uc_thres)
            self.dataset_dict['train_sc'].set_index(sc_indices)
            self.dataset_dict['train_uc'].update_scores(indices, ED_scores, uc_thres)
            self.dataset_dict['train_uc'].set_index(uc_indices)

    def warmup_step(self, x_lb, y_lb):
        outs_x_lb = self.model(x_lb)
        logits_x_lb = outs_x_lb['logits']

        sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
        total_loss = sup_loss

        self.call_hook("param_update", "ParamUpdateHook", loss=total_loss)

        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        tb_dict['train/total_loss'] = total_loss.item()
        return tb_dict

    def train_step(self, x_lb, y_lb, x_sc_w, x_sc_s, x_uc_s, w_uc):
        # inference and calculate sup/unsup losses
        with self.amp_cm():
            outs_x_lb = self.model(x_lb)
            logits_x_lb = outs_x_lb['logits']

            self.ema.apply_shadow()
            with torch.no_grad():
                self.bn_controller.freeze_bn(self.model)
                outs_sc_w = self.model(x_sc_w)
                logits_sc_w = outs_sc_w['logits']
                self.bn_controller.unfreeze_bn(self.model)
            self.ema.restore()
            probs_sc_w = torch.softmax(logits_sc_w, dim=1)
            preds_sc_w = probs_sc_w.max(1)[1]

            self.bn_controller.freeze_bn(self.model)
            outs_sc_s = self.model(x_sc_s)
            logits_sc_s = outs_sc_s['logits']
            outs_uc_s = self.model(x_uc_s)
            logits_uc_s = outs_uc_s['logits']
            self.bn_controller.unfreeze_bn(self.model)

            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

            log_preds_sc_s = F.log_softmax(logits_sc_s, -1)
            cbe_loss1 = F.nll_loss(log_preds_sc_s, preds_sc_w)
            cbe_loss2 = F.kl_div(log_preds_sc_s, probs_sc_w)
            cbe_loss = cbe_loss1 + cbe_loss2

            num_uc = logits_uc_s.shape[0]
            uni_dist = torch.full((num_uc, self.num_classes), 1 / self.num_classes).to(self.gpu)
            ucd_loss = (F.kl_div(F.log_softmax(logits_uc_s, -1),
                                 uni_dist, reduction='none').sum(1) * w_uc).mean()

            total_loss = sup_loss + self.lambda1 * cbe_loss + self.lambda2 * ucd_loss

        # parameter updates
        self.call_hook("param_update", "ParamUpdateHook", loss=total_loss)

        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        tb_dict['train/cbe_loss_1'] = cbe_loss1.item()
        tb_dict['train/cbe_loss_2'] = cbe_loss2.item()
        tb_dict['train/cbe_loss'] = cbe_loss.item()
        tb_dict['train/ucd_loss'] = ucd_loss.item()
        tb_dict['train/total_loss'] = total_loss.item()
        return tb_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--pre_epoch', int, 25, 'number of pre-training epochs'),
            SSL_Argument('--thres1', float, 0.4),
            SSL_Argument('--thres2', float, 0.3),
            SSL_Argument('--lambda1', float, 1.0),
            SSL_Argument('--lambda2', float, 0.05),
        ]
