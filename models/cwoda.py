from __future__ import print_function
import argparse
import time
import os
import datetime
from utils import utils
from utils.utils import OptimWithSheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from models.function import HLoss
from models.function import BetaMixture1D, VariationalBetaMixture1D
from models.function import CrossEntropyLoss
from models.basenet import *
import copy
from tqdm import tqdm
from utils.utils import inverseDecayScheduler, CosineScheduler, StepScheduler, ConstantScheduler
from sklearn.metrics import average_precision_score, roc_auc_score

import torch
import numpy as np
import scipy.io

import torch
from utils.compute_openmax import *
from utils.contrastive_loss import *
from utils.visualize_features import *






class CWODA():
    def __init__(self, args, num_class, src_dset, target_dset):
        self.model = 'CWODA'
        self.args = args
        self.ablation_mode = getattr(args, 'ablation_mode', 'full_osda')
        self.all_num_class = num_class
        self.known_num_class = num_class - 1
        self.dataset = args.dataset
        self.src_dset = src_dset
        self.target_dset = target_dset
        self.device = self.args.device

        self.build_model_init()
        self.ent_criterion = HLoss()
        self.bmm_model = self.cont = self.k = 0
        self.bmm_model_maxLoss = torch.log(torch.FloatTensor([self.known_num_class])).to(self.device)
        self.bmm_model_minLoss = torch.FloatTensor([0.0]).to(self.device)
        self.bmm_update_cnt = 0

        loader_workers = getattr(self.args, 'num_workers', 4)
        prefetch_factor = getattr(self.args, 'prefetch_factor', 4)
        persistent_workers = not getattr(self.args, 'no_persistent_workers', False)
        self.src_train_loader, self.src_val_loader, self.src_test_loader, self.src_train_idx = src_dset.get_loaders(
            class_balance_train=True,
            num_workers=loader_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers)
        self.target_train_loader, self.target_val_loader, self.target_test_loader, self.tgt_train_idx = target_dset.get_loaders(
            num_workers=loader_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers)
        self.num_batches = min(len(self.src_train_loader), len(self.target_train_loader))
        self.openmax_means = None
        self.openmax_weibull_params = None
        self.openmax_cache_epoch = None

        self.cutoff = False
        if self.args.dataset.lower() == 'visda':
            self.cutoff = True


    def build_model_init(self):
        self.G, self.E, self.C = utils.get_model_init(self.args, known_num_class=self.known_num_class, all_num_class=self.all_num_class)

        # if self.args.cuda:
        #     self.G.to(self.args.device)
        #     self.E.to(self.args.device)
        #     self.C.to(self.args.device)

        scheduler = lambda step, initial_lr: inverseDecayScheduler(step, initial_lr, gamma=0, power=0.75,
                                                                  max_iter=self.args.warmup_iter)

        g_model = utils.unwrap_model(self.G)
        if 'vgg' == self.args.net:
            for name, param in g_model.named_parameters():
                if 'lower' in name:
                    param.requires_grad = False
                elif 'upper' in name:
                    param.requires_grad = False
            params = list(list(g_model.linear1.parameters()) + list(g_model.linear2.parameters()) + list(
                g_model.bn1.parameters()) + list(g_model.bn2.parameters()))
        else:
            params = list(self.G.parameters())

        self.opt_w_g = OptimWithSheduler(optim.SGD(params, lr=self.args.g_lr * self.args.e_lr, weight_decay=5e-4, momentum=0.9,
                               nesterov=True), scheduler)
        self.opt_w_e = OptimWithSheduler(optim.SGD(self.E.parameters(), lr=self.args.e_lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler)
        self.opt_w_c = OptimWithSheduler(optim.SGD(self.C.parameters(), lr=self.args.e_lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler)


    def build_model(self):
        def weights_init_bias_zero(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)

        # Preserve the pretrained classifier; initialize only the domain head.
        self.DC = utils.maybe_data_parallel(
            Net_CLS_DC(utils.unwrap_model(self.G).output_num(),
                       out_dim=self.domain_discriminator_classes(),
                       bottle_neck_dim=self.args.bottle_neck_dim2), self.args)

        self.DC.apply(weights_init_bias_zero)

        # if self.args.cuda:
        #     self.E.to(self.args.device)
        #     self.DC.to(self.args.device)

        SCHEDULER = {'cos': CosineScheduler, 'step': StepScheduler, 'id': inverseDecayScheduler, 'constant':ConstantScheduler}
        scheduler = lambda step, initial_lr: SCHEDULER[self.args.scheduler](step, initial_lr, gamma=10, power=0.75,
                                                                            max_iter=self.num_batches*self.args.training_iter)
        scheduler_dc = lambda step, initial_lr: SCHEDULER[self.args.scheduler](step, initial_lr, gamma=10, power=0.75,
                                                                            max_iter=self.num_batches*self.args.training_iter*self.args.update_freq_D)

        g_model = utils.unwrap_model(self.G)
        if 'vgg' == self.args.net:
            for name,param in g_model.named_parameters():
                if 'lower' in name:
                    param.requires_grad = False
                elif 'upper' in name:
                    param.requires_grad = False
            params = list(list(g_model.linear1.parameters()) + list(g_model.linear2.parameters()) + list(
                g_model.bn1.parameters()) + list(g_model.bn2.parameters()))
        else:
            params = list(self.G.parameters())

        self.opt_g = OptimWithSheduler(
            optim.SGD(params, lr=self.args.g_lr * self.args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler)
        self.opt_c = OptimWithSheduler(
            optim.SGD(self.C.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler)
        self.opt_dc = OptimWithSheduler(
            optim.SGD(self.DC.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler_dc)

        scheduler_e = lambda step, initial_lr: inverseDecayScheduler(step, initial_lr, gamma=0, power=0.75,
                                                                     max_iter=self.num_batches*self.args.training_iter)
        self.opt_e = OptimWithSheduler(
            optim.SGD(self.E.parameters(), lr=self.args.e_lr, weight_decay=5e-4, momentum=0.9, nesterov=True),
            scheduler_e)
        self.freeze_GE()

    def network_initialization(self):
        e_model = utils.unwrap_model(self.E)
        if 'resnet' in self.args.net:
            try:
                e_model.fc.reset_parameters()
                e_model.bottleneck.reset_parameters()
            except:
                e_model.fc.reset_parameters()
        elif 'vgg' in self.args.net:
            try:
                e_model.fc.reset_parameters()
                e_model.bottleneck.reset_parameters()
            except:
                e_model.fc.reset_parameters()

    def train_init(self):
        print('train_init starts')
        reconstruction_loss = nn.MSELoss()

        t1 = time.time()
        epoch_cnt =0
        step=0
        while step < self.args.warmup_iter + 1:
            self.G.train()
            self.E.train()
            self.C.train()
            epoch_cnt +=1
            for batch_idx, ((img_s, _, _), label_s, _) in enumerate(tqdm(self.src_train_loader)):
                if self.args.cuda:
                    img_s = Variable(img_s.to(self.device, non_blocking=True))
                    label_s = Variable(label_s.to(self.device, non_blocking=True))
                    #seq_s =  Variable(seq.cuda())

                step += 1
                if step >= self.args.warmup_iter + 1:
                    break

                self.opt_w_g.zero_grad()
                self.opt_w_e.zero_grad()
                self.opt_w_c.zero_grad()
                feat_s, recon_s = self.G(img_s)
                # loss_s_recon = reconstruction_loss(recon_s, img_s)

                out_s = self.E(feat_s)

                label_s_onehot = nn.functional.one_hot(label_s, num_classes=self.known_num_class)
                label_s_onehot = label_s_onehot * (1 - self.args.ls_eps)
                label_s_onehot = label_s_onehot + self.args.ls_eps / (self.known_num_class)
                loss_s = CrossEntropyLoss(label=label_s_onehot, predict_prob=F.softmax(out_s, dim=1))

                out_Cs = self.C(feat_s)
                label_s_onehot = nn.functional.one_hot(label_s, num_classes=self.all_num_class)
                label_s_onehot = label_s_onehot * (1 - self.args.ls_eps)
                label_s_onehot = label_s_onehot + self.args.ls_eps / (self.all_num_class)
                loss_Cs = CrossEntropyLoss(label=label_s_onehot, predict_prob=F.softmax(out_Cs, dim=1))

                # loss = loss_s + loss_Cs + 0.5*loss_s_recon
                loss = loss_s + loss_Cs

                loss.backward()
                self.opt_w_g.step()
                self.opt_w_e.step()
                self.opt_w_c.step()
                self.opt_w_g.zero_grad()
                self.opt_w_e.zero_grad()
                self.opt_w_c.zero_grad()

        duration = str(datetime.timedelta(seconds=time.time() - t1))[:7]
        print('train_init end with duration: %s'%duration)


    def compute_entropy(self, out_t):
        # 计算每个样本的熵
        prob = F.softmax(out_t, dim=1)  # 获取每个类别的预测概率
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)  # 计算熵
        return entropy

    def use_alignment(self):
        return self.ablation_mode in {'full_osda', 'da_wo_os'}

    def domain_discriminator_classes(self):
        """Use ordinary binary domain alignment in the DA-only control."""
        return 2 if self.ablation_mode == 'da_wo_os' else 3

    def use_unknown_aware_open_set(self):
        return self.ablation_mode in {'full_osda', 'os_wo_da'}

    def use_unknown_loss(self):
        return self.ablation_mode in {'full_osda', 'os_wo_da'}

    def use_target_pseudo_loss(self):
        return self.ablation_mode in {'full_osda', 'os_wo_da'}

    def pseudo_label_known_only(self):
        return self.ablation_mode == 'da_wo_os'

    def unknown_loss_weights(self, weights):
        if getattr(self.args, 'unknown_loss_weight_mode', 'weibull') == 'uniform':
            return torch.ones_like(weights)
        return weights

    def resolve_target_posteriors(self, out_t_free, batch_size, device):
        if self.use_unknown_aware_open_set():
            if getattr(self.args, 'confidence_estimator', 'weibull') == 'msp':
                known = F.softmax(out_t_free.detach(), dim=1).max(dim=1).values
                return known, 1.0 - known
            if self.openmax_means is None or self.openmax_weibull_params is None:
                if getattr(self.args, 'calibration_mode', 'source_class') == 'source_class':
                    raise RuntimeError('Source calibration must be refreshed before target weighting')
                means, weibull_params = compute_means_and_weibull_params_clustered(
                    out_t_free,
                    num_classes=self.known_num_class + 1,
                )
            else:
                means = self.openmax_means.to(device, non_blocking=True)
                weibull_params = self.openmax_weibull_params.to(device, non_blocking=True)
            _, unknown_scores = compute_openmax(out_t_free, means, weibull_params)
            w_unk_posterior = unknown_scores.detach().view(-1).to(device)
            w_k_posterior = (1.0 - w_unk_posterior).to(device)
            return w_k_posterior, w_unk_posterior

        w_k_posterior = torch.ones(batch_size, device=device)
        w_unk_posterior = torch.zeros(batch_size, device=device)
        return w_k_posterior, w_unk_posterior

    def build_target_domain_labels(self, label_dt_known, label_dt_unknown, w_k_posterior, w_unk_posterior):
        if self.use_unknown_aware_open_set():
            if getattr(self.args, 'alignment_confidence_mode', 'adaptive') == 'constant':
                w_k_posterior = torch.full_like(w_k_posterior, 0.5)
                w_unk_posterior = torch.full_like(w_unk_posterior, 0.5)
            label_dt_d = w_k_posterior[:, None] * label_dt_known + w_unk_posterior[:, None] * label_dt_unknown
            label_dt_g = w_k_posterior[:, None] * label_dt_known - w_unk_posterior[:, None] * label_dt_unknown
        else:
            label_dt_d = label_dt_known
            label_dt_g = label_dt_known
        return label_dt_d, label_dt_g

    def should_refresh_openmax(self, epoch):
        if getattr(self.args, 'confidence_estimator', 'weibull') == 'msp':
            return False
        if not self.use_unknown_aware_open_set():
            return False
        if not hasattr(self, 'G_freezed') or not hasattr(self, 'E_freezed'):
            return False
        if self.openmax_means is None or self.openmax_weibull_params is None:
            return True
        refresh_interval = max(1, getattr(self.args, 'openmax_refresh_interval', 1))
        return (epoch - 1) % refresh_interval == 0

    def refresh_openmax_cache(self, epoch):
        if not self.should_refresh_openmax(epoch):
            return

        max_samples = getattr(self.args, 'openmax_max_samples', 4096)
        remaining = None if max_samples <= 0 else int(max_samples)
        logits_chunks = []
        label_chunks = []
        source_calibration = getattr(self.args, 'calibration_mode', 'source_class') == 'source_class'
        calibration_loader = self.src_train_loader if source_calibration else self.target_train_loader

        self.G_freezed.eval()
        self.E_freezed.eval()
        with torch.no_grad():
            for (img_t, img_t_og, _), labels, _ in calibration_loader:
                if self.args.cuda:
                    img_t_og = img_t_og.to(self.device, non_blocking=True)
                feat_t_free, _ = self.G_freezed(img_t_og)
                out_t_free = self.E_freezed(feat_t_free)
                if remaining is None:
                    logits_chunks.append(out_t_free.detach())
                    label_chunks.append(labels)
                else:
                    take_count = min(remaining, out_t_free.size(0))
                    logits_chunks.append(out_t_free[:take_count].detach())
                    label_chunks.append(labels[:take_count])
                    remaining -= take_count
                    if remaining <= 0:
                        break

        if not logits_chunks:
            return

        target_logits = torch.cat(logits_chunks, dim=0)
        if source_calibration:
            means, weibull_params = compute_source_class_params(
                target_logits, torch.cat(label_chunks), self.known_num_class)
        else:
            means, weibull_params = compute_means_and_weibull_params_clustered(
                target_logits, num_classes=self.known_num_class + 1)
        self.openmax_means = means.detach()
        self.openmax_weibull_params = weibull_params.detach()
        self.openmax_cache_epoch = epoch

    def compute_oscr(self, known_scores, unknown_scores, known_correct):
        known_scores = np.asarray(known_scores, dtype=np.float64)
        unknown_scores = np.asarray(unknown_scores, dtype=np.float64)
        known_correct = np.asarray(known_correct, dtype=np.float64)

        if known_scores.size == 0 or unknown_scores.size == 0:
            return None

        all_scores = np.concatenate([known_scores, unknown_scores], axis=0)
        all_known_correct = np.concatenate([known_correct, np.zeros_like(unknown_scores)], axis=0)
        all_unknown_flags = np.concatenate([
            np.zeros_like(known_scores, dtype=np.float64),
            np.ones_like(unknown_scores, dtype=np.float64),
        ], axis=0)

        order = np.argsort(-all_scores)
        sorted_scores = all_scores[order]
        sorted_known_correct = all_known_correct[order]
        sorted_unknown_flags = all_unknown_flags[order]

        ccr = np.cumsum(sorted_known_correct) / float(known_scores.size)
        fpr = np.cumsum(sorted_unknown_flags) / float(unknown_scores.size)
        distinct_last = np.r_[np.diff(sorted_scores) != 0, True]
        ccr_points = np.concatenate([[0.0], ccr[distinct_last]])
        fpr_points = np.concatenate([[0.0], fpr[distinct_last]])
        return float(np.trapz(ccr_points, fpr_points))

    def should_visualize_features(self, epoch):
        interval = getattr(self.args, 'feature_vis_interval', 0)
        return interval > 0 and epoch % interval == 0

    def append_visualization_batch(self, feature_buffer, label_buffer, features, labels, remaining):
        if remaining == 0:
            return 0

        features_np = F.normalize(features, p=2, dim=1).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        take_count = labels_np.shape[0] if remaining is None else min(remaining, labels_np.shape[0])
        feature_buffer.append(features_np[:take_count])
        label_buffer.append(labels_np[:take_count])

        if remaining is None:
            return None
        return remaining - take_count

    def train(self):
        print('Train Starts')
        lamda1 = self.args.lamda1
        lamda2 = self.args.lamda2
        lamda3 = self.args.lamda3
        eval_interval = max(1, getattr(self.args, 'eval_interval', 1))
        metric_history = []
        selecting = getattr(self.args, 'selection_protocol', 'legacy') in {'validation_hos', 'development_hos'}
        from utils.validation_selection import HOSSelector, unknown_loss_coefficient
        selector = HOSSelector(self.args.result_artifact_dir, self.args.selection_protocol) if selecting else None
        selection_modules = {name: getattr(self, name) for name in ('G', 'E', 'C', 'DC')}

        format_metric = lambda value: 'NA' if value is None else f'{value:.3f}'

        # reconstruction_loss = nn.MSELoss()
        t1 = time.time()
        for epoch in range(1, self.args.training_iter + 1):
            lamda3 = unknown_loss_coefficient(self.args.lamda3, epoch,
                        getattr(self.args, 'unknown_loss_ramp_epochs', 0))
            # Teacher updates are training operations, independent of evaluation.
            if epoch > 1 and (epoch - 1) % max(1, self.args.update_term) == 0:
                self.freeze_GE()
            self.refresh_openmax_cache(epoch)
            confidence_sum = confidence_square_sum = 0.0
            confidence_count = 0
            joint_loader = zip(self.src_train_loader, self.target_train_loader)
            alpha = float((float(2) / (1 + np.exp(-10 * float((float(epoch) / float(self.args.training_iter)))))) - 1)
            should_visualize = self.should_visualize_features(epoch)
            max_vis_samples = getattr(self.args, 'feature_vis_max_samples', 0)
            remaining_source = None if max_vis_samples <= 0 else max_vis_samples
            remaining_target = None if max_vis_samples <= 0 else max_vis_samples
            source_feature = [] if should_visualize else None
            source_label = [] if should_visualize else None
            target_feature = [] if should_visualize else None
            target_label = [] if should_visualize else None

            for batch_idx, (((img_s, _, _), label_s, _), ((img_t, img_t_og, img_t_aug), label_t, index_t)) in enumerate(tqdm(joint_loader)):
                if getattr(self.args, 'max_batches', 0) and batch_idx >= self.args.max_batches:
                    break

                self.G.train()
                self.C.train()
                self.DC.train()
                self.E.train()
                if self.args.cuda:
                    img_s = Variable(img_s.to(self.device, non_blocking=True))
                    #seq_s = Variable(seq_s.cuda())
                    label_s = Variable(label_s.to(self.device, non_blocking=True))
                    img_t = Variable(img_t.to(self.device, non_blocking=True))
                    img_t_og = Variable(img_t_og.to(self.device, non_blocking=True))
                    #seq_t = Variable(seq_t.cuda())
                # DataLoader collates the augmentation committee as a list on
                # both CPU and CUDA. Select the first view before model input.
                img_t_aug = Variable(img_t_aug[0].to(self.device, non_blocking=self.args.cuda))

                batch_device = img_s.device
                if self.use_unknown_aware_open_set():
                    with torch.no_grad():
                        img_t_og_G, _ = self.G_freezed(img_t_og)
                        out_t_free = self.E_freezed(img_t_og_G)
                else:
                    out_t_free = None

                w_k_posterior, w_unk_posterior = self.resolve_target_posteriors(
                    out_t_free,
                    batch_size=img_t.size(0),
                    device=batch_device,
                )
                confidence_sum += w_unk_posterior.sum().item()
                confidence_square_sum += w_unk_posterior.square().sum().item()
                confidence_count += w_unk_posterior.numel()

                domain_class_count = self.domain_discriminator_classes()
                label_ds = torch.zeros(img_s.size()[0], dtype=torch.long, device=batch_device)
                label_ds = nn.functional.one_hot(label_ds, num_classes=domain_class_count)
                label_dt_known = torch.ones(img_t.size()[0], dtype=torch.long, device=batch_device)
                label_dt_known = nn.functional.one_hot(label_dt_known, num_classes=domain_class_count)
                label_dt_unknown = None
                if self.use_unknown_aware_open_set():
                    label_dt_unknown = 2 * torch.ones(
                        img_t.size()[0], dtype=torch.long, device=batch_device)
                    label_dt_unknown = nn.functional.one_hot(
                        label_dt_unknown, num_classes=domain_class_count)
                label_dt_d, label_dt_g = self.build_target_domain_labels(
                    label_dt_known,
                    label_dt_unknown,
                    w_k_posterior,
                    w_unk_posterior,
                )

                #########################################################################################################
                if self.use_alignment():
                    for d_step in range(self.args.update_freq_D):
                        self.opt_dc.zero_grad()
                        feat_s, recon_s = self.G(img_s)
                        out_ds = self.DC(feat_s.detach())
                        loss_ds = CrossEntropyLoss(label=label_ds, predict_prob=F.softmax(out_ds, dim=1))

                        feat_t, recon_t = self.G(img_t)
                        out_dt = self.DC(feat_t.detach())
                        loss_dt = CrossEntropyLoss(label=label_dt_d, predict_prob=F.softmax(out_dt, dim=1))

                        loss_D = 0.5 * (loss_ds + loss_dt)
                        loss_D.backward()

                        if self.args.opt_clip > 0.0:
                            torch.nn.utils.clip_grad_norm_(self.DC.parameters(), self.args.opt_clip)
                        self.opt_dc.step()
                        self.opt_dc.zero_grad()
                #########################################################################################################
                for _ in range(self.args.update_freq_G):
                    self.opt_g.zero_grad()
                    self.opt_c.zero_grad()
                    self.opt_e.zero_grad()
                    feat_s, recon_s = self.G(img_s)

                    if self.use_alignment():
                        out_ds = self.DC(feat_s)
                        loss_ds = CrossEntropyLoss(label=label_ds, predict_prob=F.softmax(out_ds, dim=1))
                    else:
                        loss_ds = feat_s.new_tensor(0.0)

                    feat_t, recon_t = self.G(img_t)

                    if self.use_alignment():
                        out_dt = self.DC(feat_t)
                        loss_dt = CrossEntropyLoss(label=label_dt_g, predict_prob=F.softmax(out_dt, dim=1))
                        loss_G = alpha * (-loss_ds - loss_dt)
                    else:
                        loss_G = feat_s.new_tensor(0.0)

                    #########################################################################################################
                    out_Es = self.E(feat_s)
                    label_s_onehot = nn.functional.one_hot(label_s, num_classes=self.known_num_class)
                    label_s_onehot = label_s_onehot * (1 - self.args.ls_eps)
                    label_s_onehot = label_s_onehot + self.args.ls_eps / (self.known_num_class)
                    loss_cls_Es = CrossEntropyLoss(label=label_s_onehot, predict_prob=F.softmax(out_Es, dim=1))

                    out_Cs = self.C(feat_s)
                    label_Cs_onehot = nn.functional.one_hot(label_s, num_classes=self.all_num_class)
                    label_Cs_onehot = label_Cs_onehot * (1 - self.args.ls_eps)
                    label_Cs_onehot = label_Cs_onehot + self.args.ls_eps / (self.all_num_class)
                    loss_cls_Cs = CrossEntropyLoss(label=label_Cs_onehot, predict_prob=F.softmax(out_Cs, dim=1))

                    feat_t_aug, recon_t = self.G(img_t_aug)
                    out_Ct = self.C(feat_t)
                    out_Ct_aug = self.C(feat_t_aug)

                    if self.cutoff and self.use_unknown_aware_open_set():
                        w_unk_posterior = w_unk_posterior.clone()
                        w_k_posterior = w_k_posterior.clone()
                        w_unk_posterior[w_unk_posterior < self.args.threshold] = 0.0
                        w_k_posterior[w_k_posterior < self.args.threshold] = 0.0

                    if self.use_unknown_loss():
                        label_unknown = self.known_num_class * torch.ones(
                            img_t.size()[0],
                            dtype=torch.long,
                            device=batch_device,
                        )
                        label_unknown = nn.functional.one_hot(label_unknown, num_classes=self.all_num_class)
                        label_unknown_lsr = label_unknown * (1 - self.args.ls_eps)
                        label_unknown_lsr = label_unknown_lsr + self.args.ls_eps / self.all_num_class
                        loss_cls_Ctu = alpha * CrossEntropyLoss(
                            label=label_unknown_lsr,
                            predict_prob=F.softmax(out_Ct_aug, dim=1),
                            instance_level_weight=self.unknown_loss_weights(w_unk_posterior),
                        )
                    else:
                        loss_cls_Ctu = feat_t.new_tensor(0.0)

                    if self.use_target_pseudo_loss():
                        pseudo_label = torch.softmax(out_Ct.detach(), dim=-1)
                        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                        targets_u_onehot = nn.functional.one_hot(targets_u, num_classes=self.all_num_class)
                        mask = max_probs.ge(self.args.threshold).float()
                        if self.pseudo_label_known_only():
                            mask = mask * targets_u.lt(self.known_num_class).float()
                        loss_ent_Ctk = CrossEntropyLoss(
                            label=targets_u_onehot,
                            predict_prob=F.softmax(out_Ct_aug, dim=1),
                            instance_level_weight=mask,
                        )
                    else:
                        loss_ent_Ctk = feat_t.new_tensor(0.0)

                    loss = loss_cls_Es + loss_cls_Cs + lamda1 * loss_G + lamda2 * loss_ent_Ctk + lamda3 * loss_cls_Ctu

                    if should_visualize:
                        remaining_source = self.append_visualization_batch(
                            source_feature, source_label, feat_s, label_s, remaining_source)
                        remaining_target = self.append_visualization_batch(
                            target_feature, target_label, feat_t, label_t, remaining_target)

                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite training loss')
                    loss.backward()
                    self.opt_g.step()
                    self.opt_c.step()
                    self.opt_e.step()
                    self.opt_g.zero_grad()
                    self.opt_c.zero_grad()
                    self.opt_e.zero_grad()

            import json
            mean_unknown = confidence_sum / max(1, confidence_count)
            diagnostics = dict(
                epoch=epoch,
                ablation_mode=self.ablation_mode,
                domain_discriminator_classes=self.domain_discriminator_classes(),
                alignment_enabled=self.use_alignment(),
                unknown_aware_open_set_enabled=self.use_unknown_aware_open_set(),
                target_pseudo_loss_enabled=self.use_target_pseudo_loss(),
                unknown_loss_enabled=self.use_unknown_loss(),
                effective_alignment_coefficient=lamda1 if self.use_alignment() else 0.0,
                effective_target_known_coefficient=lamda2 if self.use_target_pseudo_loss() else 0.0,
                effective_unknown_loss_coefficient=lamda3 if self.use_unknown_loss() else 0.0,
                unknown_weight_mean=mean_unknown,
                unknown_weight_std=max(
                    0.0,
                    confidence_square_sum / max(1, confidence_count) - mean_unknown**2,
                )**.5,
                alpha=alpha,
                calibration_mode=getattr(self.args, 'calibration_mode', 'source_class'),
                teacher_training=getattr(self, 'G_freezed', self.G).training,
            )
            with open(os.path.join(self.args.result_artifact_dir, 'training_diagnostics.jsonl'), 'a') as diagnostic_file:
                diagnostic_file.write(json.dumps(diagnostics) + '\n')

            if should_visualize and source_feature and target_feature:
                src_features = np.concatenate(source_feature, axis=0)
                src_labels = np.concatenate(source_label)
                tar_features = np.concatenate(target_feature, axis=0)
                tar_labels = np.concatenate(target_label)
                feature_fig_dir = f'./figure2/{self.args.source_domain}{self.args.target_domain}/epoch{epoch}_featureVisual'
                visualize_features_multi_class_light_dark(src_features, src_labels, tar_features, tar_labels, feature_fig_dir)

            if selecting and (epoch % eval_interval == 0 or epoch == self.args.training_iter):
                record = selector.observe(epoch, self.test(epoch, split='validation'), selection_modules)
                self.args.logger.info('%s metrics: %s', selector.role, record)
            if not selecting and epoch % eval_interval == 0 and epoch != self.args.training_iter:
                C_acc_os, C_acc_os_star, C_acc_unknown, C_acc_hos, auroc, aupr, oscr = self.test(epoch)
                metric_history.append(utils.build_metric_record(
                    epoch,
                    C_acc_os,
                    C_acc_os_star,
                    C_acc_unknown,
                    C_acc_hos,
                    auroc,
                    aupr,
                    oscr,
                ))
                utils.save_metric_history_summary(self.args, metric_history)
                self.args.logger.info(
                    'Epoch_{:>3}/{:>3}_OS_{}_OS*_{}_UNK_{}_HOS_{}_AUROC_{}_AUPR_{}_OSCR_{}_Time_{}_BMM{}'.format(
                        epoch,
                        self.args.training_iter,
                        format_metric(C_acc_os),
                        format_metric(C_acc_os_star),
                        format_metric(C_acc_unknown),
                        format_metric(C_acc_hos),
                        format_metric(auroc),
                        format_metric(aupr),
                        format_metric(oscr),
                        str(datetime.timedelta(seconds=time.time() - t1))[:7],
                        self.bmm_update_cnt,
                    ))
                t1 = time.time()

            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            print(f"Epoch [{epoch}/{self.args.training_iter}], total_loss: {loss:.4f}, loss_cls_Es: {loss_cls_Es:.4f}, loss_cls_Cs: {loss_cls_Cs:.4f}, loss_G:{loss_G:.4f}, loss_ent_Ctk:{loss_ent_Ctk:.4f}, loss_cls_Ctu:{loss_cls_Ctu:.4f}")

        if selecting:
            torch.save({name: module.state_dict() for name, module in selection_modules.items()},
                       os.path.join(self.args.result_artifact_dir, 'last_model.pt'))
            selector.load_best(selection_modules, self.device)
            import json
            with open(os.path.join(self.args.result_artifact_dir, selector.role + '_complete.json'), 'w') as f:
                json.dump(dict(final_epoch=self.args.training_iter, best=selector.best, final=selector.history[-1], selection_metric=selector.selection_metric, independent_validation=selector.role == 'validation', test_evaluated=False), f, indent=2)
            self.args.logger.info('Selected by %s: %s; independent_validation=%s', selector.selection_metric, selector.best, selector.role == 'validation')
            return

        C_acc_os, C_acc_os_star, C_acc_unknown, C_acc_hos, auroc, aupr, oscr = self.test(self.args.training_iter)
        metric_history.append(utils.build_metric_record(
            self.args.training_iter,
            C_acc_os,
            C_acc_os_star,
            C_acc_unknown,
            C_acc_hos,
            auroc,
            aupr,
            oscr,
        ))

        # feature_dir = f'./figure/{self.args.source_domain}{self.args.target_domain}/epoch{epoch}_featureVisual'
        # visualize_features(features, am_labels, feature_dir)

        self.args.logger.info(
            'Epoch_{:>3}/{:>3}_OS_{}_OS*_{}_UNK_{}_HOS_{}_AUROC_{}_AUPR_{}_OSCR_{}_Time_{}_BMM{}'.format(
                self.args.training_iter,
                self.args.training_iter,
                format_metric(C_acc_os),
                format_metric(C_acc_os_star),
                format_metric(C_acc_unknown),
                format_metric(C_acc_hos),
                format_metric(auroc),
                format_metric(aupr),
                format_metric(oscr),
                str(datetime.timedelta(seconds=time.time() - t1))[:7],
                self.bmm_update_cnt,
            ))

        metric_summary_artifacts = utils.save_metric_history_summary(self.args, metric_history)
        torch.save({name: getattr(self, name).state_dict() for name in ('G', 'E', 'C', 'DC')},
                   os.path.join(self.args.result_artifact_dir, 'final_model.pt'))
        if metric_summary_artifacts:
            self.args.logger.info('Metric history saved to {}'.format(metric_summary_artifacts['history_csv_path']))
            self.args.logger.info('Metric summary saved to {}'.format(metric_summary_artifacts['summary_path']))
            for summary_line in utils.render_metric_summary_lines(metric_summary_artifacts['summary']):
                self.args.logger.info(summary_line)

        # # scipy.io.savemat('./feature/feature_classes8.mat', mdict={'am_embeds_norm': features, 'am_labels': am_labels})
        # scipy.io.savemat(feature_dir+'_classes8.mat', mdict={'am_embeds_norm': features, 'am_labels': am_labels})




    def compute_probabilities_batch(self, out_t, unk=1):
        ent_t = self.ent_criterion(out_t)
        batch_ent_t = (ent_t - self.bmm_model_minLoss) / (self.bmm_model_maxLoss - self.bmm_model_minLoss + 1e-6)
        batch_ent_t[batch_ent_t >= 1 - 1e-4] = 1 - 1e-4
        batch_ent_t[batch_ent_t <=  1e-4] = 1e-4
        B = self.bmm_model.posterior(batch_ent_t.clone().cpu().numpy(), unk)
        B = torch.FloatTensor(B)
        return B

    def freeze_GE(self):
        self.G_freezed = copy.deepcopy(self.G)
        self.E_freezed = copy.deepcopy(self.E)
        self.G_freezed.eval().requires_grad_(False)
        self.E_freezed.eval().requires_grad_(False)
        self.openmax_means = None
        self.openmax_weibull_params = None
        self.openmax_cache_epoch = None

    def test(self, epoch, split='test'):
        if split not in {'validation', 'test'}:
            raise ValueError('Unknown evaluation split')
        selecting = getattr(self.args, 'selection_protocol', 'legacy') in {'validation_hos', 'development_hos'}
        if selecting and split == 'test' and not getattr(self.args, 'heldout_test_authorized', False):
            raise RuntimeError('Test evaluation is reserved for the locked selected configuration')
        eval_args = copy.copy(self.args)
        if selecting:
            role = 'development' if self.args.selection_protocol == 'development_hos' else split
            eval_args.result_artifact_dir = os.path.join(self.args.result_artifact_dir, role)
            os.makedirs(eval_args.result_artifact_dir, exist_ok=True)
        loader = self.target_val_loader if split == 'validation' else self.target_test_loader
        prior_modes = [self.G.training, self.C.training, self.E.training]
        self.G.eval()
        self.C.eval()
        self.E.eval()
        total_pred_chunks = []
        total_label_chunks = []
        total_prob_chunks = []
        total_unknown_chunks = []
        total_index_chunks = []
        all_ent_chunks = []

        with torch.no_grad():
            for batch_idx, ((img_t, _, _), label_t, index_t) in enumerate(tqdm(loader)):
                if getattr(self.args, 'max_batches', 0) and batch_idx >= self.args.max_batches:
                    break
                if self.args.cuda:
                    img_t = Variable(img_t.to(self.device, non_blocking=True))
                    label_t = Variable(label_t.to(self.device, non_blocking=True))
                feat_t, _ = self.G(img_t)
                out_t = F.softmax(self.C(feat_t), dim=1)
                if not self.use_unknown_aware_open_set():
                    # Closed-set controls have no learned unknown-class output.
                    known_prob = out_t[:, :self.known_num_class]
                    known_prob = known_prob / known_prob.sum(dim=1, keepdim=True)
                    out_t = torch.cat([known_prob, torch.zeros_like(known_prob[:, :1])], dim=1)

                pred = out_t.data.max(1)[1]
                total_pred_chunks.append(pred.cpu().numpy())
                total_label_chunks.append(label_t.cpu().numpy())
                total_index_chunks.append(index_t.cpu().numpy())
                total_prob_chunks.append(out_t.detach().cpu().numpy())

                # 记录未知类别在 softmax 输出中的概率
                p_unknown = out_t[:, self.known_num_class].detach().cpu().numpy()
                total_unknown_chunks.append(p_unknown)

                out_Et = self.E(feat_t)
                ent_Et = self.ent_criterion(out_Et)
                all_ent_chunks.append(ent_Et.cpu())

        total_pred_t = np.concatenate(total_pred_chunks).astype(np.int64)
        total_label_t = np.concatenate(total_label_chunks).astype(np.int64)
        total_prob_t = np.concatenate(total_prob_chunks, axis=0)
        total_p_unknown = np.concatenate(total_unknown_chunks)
        all_ent_t = torch.cat(all_ent_chunks, dim=0)
        binary_labels = (total_label_t >= self.known_num_class).astype(np.int64)
        has_unknown = np.any(binary_labels == 1)
        has_known = np.any(binary_labels == 0)
        if not self.use_unknown_aware_open_set():
            # Auxiliary threshold-free detection score, not an operational rejector.
            total_p_unknown = 1.0 - total_prob_t[:, :self.known_num_class].max(axis=1)
        if getattr(self.args, 'export_predictions', False):
            np.savez_compressed(
                os.path.join(eval_args.result_artifact_dir, f'predictions_epoch_{epoch:03d}.npz'),
                sample_index=np.concatenate(total_index_chunks), labels=total_label_t,
                probabilities=total_prob_t, predictions=total_pred_t,
                unknown_score=total_p_unknown,
            )

        if has_known and has_unknown:
            try:
                auroc = roc_auc_score(binary_labels, total_p_unknown)
            except Exception as e:
                print("Error when computing AUROC:", e)
                auroc = None
            try:
                aupr = average_precision_score(binary_labels, total_p_unknown)
            except Exception as e:
                print("Error when computing AUPR:", e)
                aupr = None

            known_mask = binary_labels == 0
            unknown_mask = binary_labels == 1
            max_known_scores = np.max(total_prob_t[:, :self.known_num_class], axis=1)
            known_scores = max_known_scores[known_mask]
            unknown_scores = max_known_scores[unknown_mask]
            known_correct = (total_prob_t[known_mask, :self.known_num_class].argmax(axis=1) == total_label_t[known_mask]).astype(np.float64)
            oscr = self.compute_oscr(known_scores, unknown_scores, known_correct)
        else:
            auroc = None
            aupr = None
            oscr = None

        print("AUROC:", auroc)
        print("AUPR:", aupr)
        print("OSCR:", oscr)

        max_target_label = int(np.max(total_label_t) + 1)
        m = utils.extended_confusion_matrix(
            total_label_t,
            total_pred_t,
            true_labels=list(range(max_target_label)),
            pred_labels=list(range(self.all_num_class)),
        )
        cm = utils.normalize_confusion_matrix(m)
        print(cm)
        known_eval_classes = min(self.known_num_class, cm.shape[0])
        acc_os_star = sum(cm[i][i] for i in range(known_eval_classes)) / self.known_num_class

        unknown_class_count = max(0, max_target_label - self.known_num_class)
        if unknown_class_count > 0:
            acc_unknown = sum(
                cm[i][self.known_num_class]
                for i in range(self.known_num_class, max_target_label)
            ) / unknown_class_count
            acc_os = (acc_os_star * self.known_num_class + acc_unknown) / (self.known_num_class + 1)
            acc_hos = 0.0 if (acc_os_star + acc_unknown) <= 0 else (
                2 * acc_os_star * acc_unknown / (acc_os_star + acc_unknown)
            )
        else:
            acc_unknown = None
            acc_os = acc_os_star
            acc_hos = None

        should_export_confusion = (
            getattr(self.args, 'result_artifact_dir', None) is not None and
            (epoch == 0 or epoch == self.args.training_iter or epoch % self.args.update_term == 0)
        )
        if should_export_confusion:
            confusion_artifacts = utils.export_target_confusion_matrices(
                self.args,
                epoch,
                total_label_t,
                total_pred_t,
                self.known_num_class,
                self.all_num_class,
            )
            if confusion_artifacts and hasattr(self.args, 'logger'):
                self.args.logger.info(
                    'Target {} confusion matrix saved to {}'.format(
                        self.args.target_domain, confusion_artifacts['target']['figure_path']))
                self.args.logger.info(
                    utils.render_confusion_matrix_text(
                        confusion_artifacts['target']['normalized'],
                        confusion_artifacts['target']['row_labels'],
                        confusion_artifacts['target']['col_labels']))

        for module, mode in zip((self.G, self.C, self.E), prior_modes):
            module.train(mode)

        return acc_os, acc_os_star, acc_unknown, acc_hos, auroc, aupr, oscr
