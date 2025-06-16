
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import copy
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import DistAlignQueueHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool, concat_all_gather
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset

from semilearn.algorithms.utils import ce_loss
from torch.utils.data import DataLoader
from skimage.filters import threshold_otsu
from sklearn.mixture import GaussianMixture


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class UAGreg_Dataset(BasicDataset):
    def __init__(self, dset, name):
        self.data = copy.deepcopy(dset.data)
        self.targets = copy.deepcopy(dset.targets)
        super(UAGreg_Dataset, self).__init__(alg='committee', data=self.data, targets=self.targets, num_classes=dset.num_classes,
                                         transform=dset.transform, strong_transform=dset.strong_transform)
        self.name = name
        self.data_index = None
        self.targets_index = None
        self.set_index()

    def set_index(self, indices=None):
        if indices is not None:
            self.data_index = self.data[indices]
            self.targets_index = self.targets[indices]
        else:
            self.data_index = self.data
            self.targets_index = self.targets

    def __len__(self):
        return len(self.data_index)

    def __sample__(self, idx):
        target = self.targets_index[idx] if self.targets is not None else None
        img = self.data_index[idx]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)  # shape of img should be [H, W, C]
        if isinstance(img, str):
            img = pil_loader(img)
        return img, target

    def __getitem__(self, idx):
        img, target = self.__sample__(idx)
        img_w = self.transform(img)
        if self.name == 'train_lb':
            return {'idx_lb': idx, 'x_lb': img_w, 'x_lb_w_0': img_w, 'x_lb_w_1': self.transform(img), 'y_lb': target}
        elif self.name == 'train_ulb':
            return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s_0': self.strong_transform(img), 'x_ulb_s_1': self.strong_transform(img), 'y_ulb': target}


class UAGreg_Net(nn.Module):
    def __init__(self, base, num_classes):
        super(UAGreg_Net, self).__init__()
        self.backbone = base
        self.feat_planes = base.num_features

        self.proj_dim = self.feat_planes // 2
        if self.feat_planes > 256:
            self.proj_dim = 128
        # print(f"\n\nPROJECTION_DIM: {self.proj_dim}\n\n")
        self.mlp_proj = nn.Sequential(
            nn.Linear(self.feat_planes, self.feat_planes),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Linear(self.feat_planes, self.proj_dim))

        self.fc_open = nn.Linear(self.feat_planes, num_classes)
        nn.init.xavier_normal_(self.fc_open.weight.data)
        self.fc_open.bias.data.zero_()

    def l2norm(self, x, power=2):
        norm = x.pow(power).sum(1, keepdim=True).pow(1. / power)
        out = x.div(norm)
        return out

    def forward(self, x, **kwargs):
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)
        logits_open = self.fc_open(feat)
        feat_proj = self.l2norm(self.mlp_proj(feat))
        return {'logits':logits, 'logits_open': logits_open, 'feat':feat_proj}

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher


@ALGORITHMS.register('uagreg')
class UAGreg(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)

    def set_dataset(self):
        dataset_dict = super(UAGreg, self).set_dataset()
        dataset_dict['train_lb'] = UAGreg_Dataset(dset=dataset_dict['train_lb'], name='train_lb')
        dataset_dict['train_ulb'] = UAGreg_Dataset(dset=dataset_dict['train_ulb'], name='train_ulb')
        dataset_dict['train_ulb_selected'] = UAGreg_Dataset(dset=dataset_dict['train_ulb'], name='train_ulb_selected')
        return dataset_dict

    def set_hooks(self):
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def set_model(self):
        model = super().set_model()
        model = UAGreg_Net(model, num_classes=self.num_classes)
        return model
    
    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model = UAGreg_Net(ema_model, num_classes=self.num_classes)
        ema_model.load_state_dict(self.check_prefix_state_dict(self.model.state_dict()))
        return ema_model

    def train(self):
        """ train function """
        self.model.train()
        self.call_hook("before_run")

        self.lower_th = torch.ones(1)[0].float()
        self.upper_th = torch.ones(1)[0].float()
        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch
            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break
            self.call_hook("before_train_epoch")

            if self.epoch > 0:
                curr_lower_th, curr_upper_th = self.get_threshold()
                self.lower_th = self.args.ema_momentum * self.lower_th + (1. - self.args.ema_momentum) * curr_lower_th
                self.upper_th = self.args.ema_momentum * self.upper_th + (1. - self.args.ema_momentum) * curr_upper_th
            print(f"LOWER_TH: {self.lower_th}, UPPER_TH: {self.upper_th}")

            for data_lb, data_ulb in zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb']):
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break
                self.call_hook("before_train_step")
                self.tb_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")
        self.call_hook("after_run")


    def train_step(self, x_lb_w_0, x_lb_w_1, y_lb, x_ulb_w, x_ulb_s_0, x_ulb_s_1):
        num_lb = y_lb.shape[0] 

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            inputs = torch.cat((x_lb_w_0, x_lb_w_1, x_ulb_w, x_ulb_s_0, x_ulb_s_1))
            outputs = self.model(inputs)
            logits, logits_open, feats = outputs['logits'], outputs['logits_open'], outputs['feat']
            logits_x_lb = logits[:num_lb * 2]
            logits_x_ulb_w, logits_x_ulb_s_0, logits_x_ulb_s_1 = logits[num_lb * 2:].chunk(3)
            logits_open_x_lb = logits_open[:num_lb * 2]
            logits_open_x_ulb_w, logits_open_x_ulb_s_0, logits_open_x_ulb_s_1 = logits_open[num_lb * 2:].chunk(3)
            feat_x_ulb_w, feat_x_ulb_s_0, feat_x_ulb_s_1 = feats[num_lb * 2:].chunk(3)

            # supervised loss
            loss_s = ce_loss(logits_x_lb, y_lb.repeat(2), reduction='mean')
            loss_s += ce_loss(logits_open_x_lb, y_lb.repeat(2), reduction='mean')

            ##  ##
            ood_score, _ = torch.max(torch.softmax(logits_open_x_ulb_w / self.args.temp_o, dim=1), dim=-1)
            id_mask = ood_score.ge(self.upper_th)
            ood_mask = ood_score.lt(self.lower_th)
            graph_mask = ood_mask.clone()
            # print(f"OOD_SCORES: {ood_score}")
            # print(f"SUM_OF_ID_MASK: {id_mask.sum()}")
            # print(f"SUM_OF_OOD_MASK: {ood_mask.sum()}")

            ## PENALIZING LOSS FOR OUTLIER SAMPLES ##
            probs = torch.softmax(logits_open_x_ulb_w, dim=1)
            avg_probs = (probs[ood_mask]).mean(dim=0)
            loss_o = -torch.sum(-avg_probs * torch.log(avg_probs + 1e-8))

            with torch.no_grad():
                logits_x_ulb_w = logits_x_ulb_w.detach()
                probs = torch.softmax(logits_x_ulb_w, dim=1)

            # EMBEDDING SIMILARITY
            sim = torch.exp(torch.mm(feat_x_ulb_s_0, feat_x_ulb_s_1.t()) / self.args.graph_t) 
            sim_probs = sim / sim.sum(1, keepdim=True)

            # PSEUDO-LABEL GRAPH
            Q = torch.mm(probs, probs.t())
            Q.fill_diagonal_(1)
            pos_mask = (Q >= self.args.g_cutoff).float()
            graph_mask = (graph_mask.unsqueeze(0) == graph_mask.unsqueeze(-1)).float()

            Q = Q * pos_mask
            Q = Q * graph_mask
            Q = Q / Q.sum(1, keepdim=True)

            # CONTRASTIVE LOSS
            loss_g = -(torch.log(sim_probs + 1e-7) * Q).sum(1)
            loss_g = loss_g.mean()

            if self.epoch >= self.args.start_fix:
                pseudo_label = torch.softmax(logits_x_ulb_w.detach(), dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(self.args.p_cutoff).float()
                mask = mask * id_mask.float()
                loss_u = (F.cross_entropy(logits_x_ulb_s_0, targets_u, reduction='none') * mask).mean()

            total_loss = loss_s + loss_o + self.args.lambda_u * loss_u + self.args.lambda_g * loss_g

        self.call_hook("param_update", "ParamUpdateHook", loss=total_loss)

        tb_dict = {'train/sup_loss': loss_s.item(),
                    'train/unsup_loss': loss_u.item(),
                    'train/total_loss': total_loss.item(), 
                    }
        return tb_dict


    def get_threshold(self):
        loader = DataLoader(dataset=self.dataset_dict['train_ulb'], batch_size=self.args.eval_batch_size, drop_last=False, shuffle=False, num_workers=1)

        self.model.eval()
        self.ema.apply_shadow()
        self.print_fn(f"Selecting...")
        pred_scores = torch.Tensor([]).to(self.gpu)
        pred_labels = torch.Tensor([]).to(self.gpu)
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                x = data['x_ulb_w']
                y = data['y_ulb']

                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                outputs = self.model(x)
                logits, logits_open = outputs['logits'], outputs['logits_open']
                max_probs, preds = torch.max(F.softmax(logits_open / self.args.temp_o, dim=1), dim=1)

                # 
                pred_scores = torch.cat((pred_scores, max_probs), dim=0)
                pred_labels = torch.cat((pred_labels, preds), dim=0)

        self.model.train()
        # FOR GAUSSIAN MIXTURE MODELS (GMMs)
        otsu_th = threshold_otsu(pred_scores.cpu().numpy())
        init_centers = np.array([[otsu_th], [otsu_th]])
        gmm = GaussianMixture(n_components=2, means_init=init_centers)
        # 
        gmm.fit(pred_scores.unsqueeze(-1).cpu().numpy())
        threshold = np.squeeze(gmm.means_, axis=1)
        threshold.sort()

        return threshold


    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--smoothing_alpha', float, 0.9),
            SSL_Argument('--da_len', int, 256),
            # 
            SSL_Argument('--start_fix', int, 0),
            SSL_Argument('--ema_momentum', float, 0.9),
            SSL_Argument('--temp_o', float, 1.5),
            SSL_Argument('--graph_t', float, 0.2),
            SSL_Argument('--g_cutoff', float, 0.8),
            SSL_Argument('--lambda_u', float, 1.0),
            SSL_Argument('--lambda_g', float, 1.0),
        ]
