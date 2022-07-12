# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
import tqdm


# ----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):  # to be overridden by subclass
        raise NotImplementedError()


# ----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10,
                 pl_batch_shrink=2, pl_decay=0.01, pl_weight=2, with_dataaug=False):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.with_dataaug = with_dataaug
        self.pseudo_data = None

    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff,
                                         torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img, _ = self.G_synthesis(ws)
        return img, ws

    def run_D(self, img, c, sync):
        # Enable standard data augmentations when --with-dataaug=True
        if self.with_dataaug and self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def adaptive_pseudo_augmentation(self, real_img):
        # Apply Adaptive Pseudo Augmentation (APA)
        batch_size = real_img.shape[0]
        pseudo_flag = torch.ones([batch_size, 1, 1, 1], device=self.device)
        pseudo_flag = torch.where(torch.rand([batch_size, 1, 1, 1], device=self.device) < self.augment_pipe.p,
                                  pseudo_flag, torch.zeros_like(pseudo_flag))
        if torch.allclose(pseudo_flag, torch.zeros_like(pseudo_flag)):
            return real_img
        else:
            assert self.pseudo_data is not None
            return self.pseudo_data * pseudo_flag + real_img * (1 - pseudo_flag)

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1 = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl))  # May get synced by Gpl.
                # Update pseudo data
                self.pseudo_data = gen_img.detach()
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)  # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = \
                        torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True,
                                            only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                gen_logits = self.run_D(gen_img, gen_c, sync=False)  # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)  # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                # Apply Adaptive Pseudo Augmentation (APA) when --aug!='noaug'
                if self.augment_pipe is not None:
                    real_img_tmp = self.adaptive_pseudo_augmentation(real_img)
                else:
                    real_img_tmp = real_img
                real_img_tmp = real_img_tmp.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)  # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = \
                            torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True,
                                                only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()


# ----------------------------------------------------------------------------


class FewShotAdaptationLoss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, Extra, G_mapping_src, G_synthesis_src, augment_pipe=None,
                 style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2, feat_const_batch=4,
                 kl_weight=1000, high_p=1, cl_G_weight=2, cl_D_weight=0.5, with_dataaug=False):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.G_mapping_src = G_mapping_src
        self.G_synthesis_src = G_synthesis_src
        self.D = D
        self.Extra = Extra
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.feat_const_batch = feat_const_batch
        self.kl_weight = kl_weight
        self.high_p = max(min(high_p, 4), 1)
        self.cl_D_weight = cl_G_weight
        self.cl_D_weight = cl_D_weight
        self.with_dataaug = with_dataaug
        self.pseudo_data = None

        self.sfm = torch.nn.Softmax(dim=1)
        self.sim = torch.nn.CosineSimilarity()
        self.kl_loss = torch.nn.KLDivLoss(reduction="none")

    def run_G(self, z, c, sync, is_subspace, use_source=False, return_feats=False):

        mapping = self.G_mapping if not use_source else self.G_mapping_src
        synthesis = self.G_synthesis if not use_source else self.G_synthesis_src
        with misc.ddp_sync(mapping, sync):
            ws = mapping(z, c)
            if self.style_mixing_prob > 0 and is_subspace > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff,
                                         torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(synthesis, sync):
            img, feats = synthesis(ws, return_feats=return_feats)
        return img, ws, feats

    def run_D(self, img, c, sync, is_subspace, return_feats=False):
        p_ind = np.random.randint(0, self.high_p)
        # Enable standard data augmentations when --with-dataaug=True
        if self.with_dataaug and self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits, feats = self.D(img, c, extra=self.Extra, flag=is_subspace, p_ind=p_ind, return_feats=return_feats)

        return logits, feats

    def adaptive_pseudo_augmentation(self, real_img):
        # Apply Adaptive Pseudo Augmentation (APA)
        batch_size = real_img.shape[0]
        pseudo_flag = torch.ones([batch_size, 1, 1, 1], device=self.device)
        pseudo_flag = torch.where(torch.rand([batch_size, 1, 1, 1], device=self.device) < self.augment_pipe.p,
                                  pseudo_flag, torch.zeros_like(pseudo_flag))
        if torch.allclose(pseudo_flag, torch.zeros_like(pseudo_flag)):
            return real_img
        else:
            assert self.pseudo_data is not None
            return self.pseudo_data * pseudo_flag + real_img * (1 - pseudo_flag)

    def exp_sim(self, feat_ind, anchor_ind, other_ind, feat_list_target, feat_list_src, temperature=0.07):
        anchor_feat = torch.unsqueeze(
            feat_list_target[feat_ind[anchor_ind]][anchor_ind].reshape(-1), 0).to(
            torch.float32)
        compare_feat = torch.unsqueeze(
            feat_list_src[feat_ind[anchor_ind]][other_ind].reshape(-1), 0).to(torch.float32)

        return torch.exp(self.sim(anchor_feat, compare_feat) / temperature)

    def generator_contrastive_loss(self, feat_img, gen_z, gen_c):
        with torch.autograd.profiler.record_function("Gmain_dcl_forward"):
            batch_size = gen_z.size(0)

            idx_anchor = np.random.randint(0, batch_size)
            feat_ind_G = np.random.randint(0, self.G_synthesis_src.n_latent, size=batch_size)

            with torch.set_grad_enabled(False):
                _, _, feat_img_src = self.run_G(gen_z, gen_c, sync=False, is_subspace=0, use_source=True, return_feats=True)

            sims = [self.exp_sim(feat_ind_G, idx_anchor, idx, feat_img, feat_img_src) for idx in range(batch_size)]

            gen_cl_loss = self.cl_D_weight * -torch.log(sims[idx_anchor] / torch.cat(sims, dim=0).sum(dim=0))
            training_stats.report('Loss/G/contrastive', gen_cl_loss)

            return gen_cl_loss

    def discriminator_contrastive_loss(self, feat_img, feat_img_real, gen_z, gen_c):
        with torch.autograd.profiler.record_function("Dmain_dcl_forward"):
            batch_size = gen_z.size(0)

            idx_anchor = np.random.randint(0, batch_size)
            feat_ind_D = np.random.randint(0, self.D.num_feats, size=batch_size)

            with torch.set_grad_enabled(False):
                gen_img_src, _, _ = self.run_G(gen_z, gen_c, sync=False, is_subspace=0, use_source=True)
                _, feat_img_src = self.run_D(gen_img_src, gen_c, is_subspace=0, sync=False, return_feats=True)

            sims_real = [self.exp_sim(feat_ind_D, idx_anchor, idx, feat_img, feat_img_real) for idx in
                         range(batch_size)]
            sim_anchor = self.exp_sim(feat_ind_D, idx_anchor, idx_anchor, feat_img, feat_img_src)

            D_cl_loss = self.cl_D_weight * -torch.log(
                sim_anchor / (sim_anchor + torch.cat(sims_real, dim=0).sum(dim=0)))
            training_stats.report('Loss/D/contrastive', D_cl_loss)

            return D_cl_loss

    def dist_consistency_loss(self, gen_z):
        with torch.autograd.profiler.record_function("Gmain_dist_forward"):
            with torch.set_grad_enabled(False):
                z = torch.randn([self.feat_const_batch, self.G_mapping_src.z_dim], device=gen_z.device)
                c = torch.zeros_like(z)
                feat_ind = np.random.randint(0, self.G_synthesis_src.n_latent, size=self.feat_const_batch)

                # computing source distances
                src_sample, _, feat_src = self.run_G(z, c, sync=False, is_subspace=0, use_source=True,
                                                     return_feats=True)
                dist_src = torch.zeros([self.feat_const_batch, self.feat_const_batch - 1], device=gen_z.device)

                # iterating over different elements in the batch
                for pair1 in range(self.feat_const_batch):
                    tmpc = 0
                    # comparing the possible pairs
                    for pair2 in range(self.feat_const_batch):
                        if pair1 != pair2:
                            anchor_feat = torch.unsqueeze(
                                feat_src[feat_ind[pair1]][pair1].reshape(-1), 0).to(
                                torch.float32)  # avoid overflow issues with fp16 and cosine similarity
                            compare_feat = torch.unsqueeze(
                                feat_src[feat_ind[pair1]][pair2].reshape(-1), 0).to(torch.float32)
                            dist_src[pair1, tmpc] = self.sim(anchor_feat, compare_feat)
                            tmpc += 1
                dist_src = self.sfm(dist_src)

            # computing distances among target generations
            _, _, feat_target = self.run_G(z, c, sync=False, is_subspace=0, return_feats=True)
            dist_target = torch.zeros([self.feat_const_batch, self.feat_const_batch - 1], device=gen_z.device)

            # iterating over different elements in the batch
            for pair1 in range(self.feat_const_batch):
                tmpc = 0
                for pair2 in range(self.feat_const_batch):  # comparing the possible pairs
                    if pair1 != pair2:
                        anchor_feat = torch.unsqueeze(
                            feat_target[feat_ind[pair1]][pair1].reshape(-1), 0).to(torch.float32)
                        compare_feat = torch.unsqueeze(
                            feat_target[feat_ind[pair1]][pair2].reshape(-1), 0).to(torch.float32)
                        dist_target[pair1, tmpc] = self.sim(anchor_feat, compare_feat)
                        tmpc += 1
            dist_target = self.sfm(dist_target)

            rel_loss = self.kl_weight * self.kl_loss(torch.log(dist_target), dist_src).mean(
                dim=1)  # distance consistency loss
            training_stats.report('Loss/G/dist_const', rel_loss)

            return rel_loss

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain, is_subspace, adaptation):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1 = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws, feats_img = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl),
                                                         # May get synced by Gpl.
                                                         is_subspace=is_subspace,
                                                         return_feats=True)
                # Update pseudo data
                self.pseudo_data = gen_img.detach()
                gen_logits, _ = self.run_D(gen_img, gen_c, sync=False, is_subspace=is_subspace)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)  # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)

            loss_G_diversity = self.dist_consistency_loss(gen_z) if adaptation == "CDC" \
                else self.generator_contrastive_loss(feats_img, gen_z, gen_c)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_Gmain + loss_G_diversity).mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws, _ = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync,
                                                is_subspace=is_subspace)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = \
                        torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True,
                                            only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws, _ = self.run_G(gen_z, gen_c, sync=False, is_subspace=is_subspace)
                gen_logits, feats_gen = self.run_D(gen_img, gen_c, sync=False, # Gets synced by loss_Dreal.
                                                   is_subspace=is_subspace,
                                                   return_feats=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)  # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                # Apply Adaptive Pseudo Augmentation (APA) when --aug!='noaug'
                if self.augment_pipe is not None:
                    real_img_tmp = self.adaptive_pseudo_augmentation(real_img)
                else:
                    real_img_tmp = real_img
                real_img_tmp = real_img_tmp.detach().requires_grad_(do_Dr1)
                real_logits, feats_real = self.run_D(real_img_tmp, real_c, sync=sync, is_subspace=is_subspace,
                                                     return_feats=True)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                loss_D_diversity = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)  # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                    if adaptation == "DCL":
                        loss_D_diversity = self.discriminator_contrastive_loss(feats_gen, feats_real, gen_z, gen_c)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = \
                            torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True,
                                                only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                sum_feats = 0
                if adaptation == "DCL":
                    for f in feats_real:
                        sum_feats += f * 0

                (real_logits * 0 + sum_feats + loss_Dreal + loss_Dr1 + loss_D_diversity).mean().mul(gain).backward()
