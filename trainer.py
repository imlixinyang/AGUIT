"""
Training codes for Attribute Guided Unpaired Image-to-image Translation.
Author: Xinyang Li (imlixinyang@gmail.com)
"""
"""
Original Copyright:
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis, ContentDis
from utils import weights_init, get_model_list, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import os

class AGUIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(AGUIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.noise_dim = hyperparameters['gen']['noise_dim']
        self.attr_dim = len(hyperparameters['gen']['selected_attrs'])
        self.gen = AdaINGen(hyperparameters['input_dim'], hyperparameters['gen'])
        self.dis = MsImageDis(hyperparameters['input_dim'], self.attr_dim, hyperparameters['dis'])
        self.dis_content = ContentDis(hyperparameters['gen']['dim'] * (2 ** hyperparameters['gen']['n_downsample']), self.attr_dim)

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis.parameters()) + list(self.dis_content.parameters())
        gen_params = list(self.gen.parameters())

        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis.apply(weights_init('gaussian'))
        self.dis_content.apply(weights_init('gaussian'))

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def gen_update(self, x_l, x_u, l, hyperparameters):
        self.gen_opt.zero_grad()
        # l_s_rand = torch.randn_like(l_s)
        # l_s = torch.where(l_s == 0, l_s_rand, l_s)
        s_r = torch.cat([torch.randn(x_u.size(0), self.noise_dim).cuda(), l], 1)

        # encode
        c_l, s_l = self.gen.encode(x_l)
        c_u, s_u = self.gen.encode(x_u)

        # decode (within domain)
        x_u_recon = self.gen.decode(c_u, s_u)

        # decode (cross domain)
        x_ur = self.gen.decode(c_u, s_r)

        # encode again
        c_u_recon, s_r_recon = self.gen.encode(x_ur)

        x_u_cycle = self.gen.decode(c_u_recon, s_u)

        # additional KL-loss (optional)
        s_mean = s_l[:, 0:self.noise_dim].mean()
        s_std = s_l[:, 0:self.noise_dim].std()

        self.loss_gen_kld = (s_mean ** 2 + s_std.pow(2) - s_std.pow(2).log() - 1).mean() / 2

        self.loss_gen_adv_content = self.dis_content.calc_gen_loss(c_l, c_u, l)
        # reconstruction loss
        self.loss_gen_rec = self.recon_criterion(x_u_recon, x_u)
        self.loss_gen_rec_s = self.recon_criterion(s_r_recon, s_r)
        self.loss_gen_rec_c = self.recon_criterion(c_u_recon, c_u)

        self.loss_gen_cyc = self.recon_criterion(x_u_cycle, x_u)

        # GAN loss
        self.loss_gen_adv = self.dis.calc_gen_loss(x_ur, l)

        # label part loss
        self.loss_gen_cla = (s_l[:, self.noise_dim:self.noise_dim + self.attr_dim] - l).pow(2).mean()
        
        self.loss_gen_total = hyperparameters['adv_w'] * self.loss_gen_adv + \
                              hyperparameters['adv_c_w'] * self.loss_gen_adv_content + \
                              hyperparameters['rec_w'] * self.loss_gen_rec + \
                              hyperparameters['rec_s_w'] * self.loss_gen_rec_s + \
                              hyperparameters['rec_c_w'] * self.loss_gen_rec_c + \
                              hyperparameters['cla_w'] * self.loss_gen_cla + \
                              hyperparameters['kld_w'] * self.loss_gen_kld + \
                              hyperparameters['cyc_w'] * self.loss_gen_cyc

        self.loss_gen_total.backward()

        self.gen_opt.step()

        return self.loss_gen_total.detach()

    def sample(self, x_l, l):

        c_l, s_l = self.gen.encode(x_l)

        # decode (within domain)
        x_l_recon = self.gen.decode(c_l, s_l)

        out = [x_l, x_l_recon]
        for i in range(self.attr_dim):
            s_changed = s_l.clone()
            s_changed[:, self.noise_dim + i] = -l[:, i]
            out += [self.gen.decode(c_l, s_changed)]

        return out

    def dis_update(self, x_l, x_u, l, hyperparameters):
        self.dis_opt.zero_grad()

        s_r = torch.cat([torch.randn(x_u.size(0), self.noise_dim).cuda(), l], 1)

        # encode
        c_l, s_l = self.gen.encode(x_l)
        c_u, s_u = self.gen.encode(x_u)

        # decode (cross domain)
        x_ur = self.gen.decode(c_u, s_r)

        # D loss
        self.loss_dis_adv = self.dis.calc_dis_loss(x_ur.detach(), x_l, x_u, l)
        self.loss_dis_adv_content = self.dis_content.calc_dis_loss(c_l, c_u, l)
        self.loss_dis_total = hyperparameters['adv_w'] * self.loss_dis_adv + \
                              hyperparameters['adv_c_w'] * self.loss_dis_adv_content
        self.loss_dis_total.backward()
        self.dis_opt.step()

        return self.loss_dis_total.detach()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen.load_state_dict(state_dict['gen'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis.load_state_dict(state_dict['dis'])
        self.dis_content.load_state_dict(state_dict['dis_content'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'gen': self.gen.state_dict()}, gen_name)
        torch.save({'dis': self.dis_a.state_dict(), 'dis_content': self.dis_content.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
