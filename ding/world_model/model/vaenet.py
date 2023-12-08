import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from ding.torch_utils.network.nn_module import MLP
from ding.torch_utils.network.activation import build_activation
from ding.torch_utils.network.res_block import ResFCTemporalBlock
from ding.torch_utils.time_emb import GaussianFourierProjection, SinusoidalPosEmb



class DiffusionNet(nn.Module):
    
    def __init__(
        self,
        state_size: int,
        hidden_size: int,
        layer_num: int,
        activation: str = 'mish',
        norm_type: str = 'LN'
    ):
        super().__init__()
        
        assert layer_num % 2 == 1, f"The layer num of diffusion net must be odd, but got {layer_num}!"
        
        self.activation = activation
        
        # 1. Encoder & Decoder
        self.encoder = MLP(
            state_size, 
            hidden_size * 4, 
            hidden_size, 
            layer_num=3, 
            activation=build_activation(activation), 
            norm_type=norm_type,
        )
        self.decoder = MLP(
            hidden_size, 
            hidden_size * 4, 
            state_size, 
            layer_num=3, 
            activation=build_activation(activation), 
            norm_type=norm_type,
        )
        
        # 2. Modified U-net
        self.down_groups_1 = nn.ModuleList([])    # for miu
        self.down_groups_2 = nn.ModuleList([])    # for sigma
        self.up_groups = nn.ModuleList([])
        
        # 2.1. down groups
        for _ in range(layer_num // 2):
            group = nn.ModuleList([
                ResFCTemporalBlock(hidden_size, hidden_size, time_channels=hidden_size, activation=build_activation(activation), norm_type=norm_type),
                ResFCTemporalBlock(hidden_size, hidden_size, time_channels=hidden_size, activation=build_activation(activation), norm_type=norm_type),
            ])
            self.down_groups_1.append(group)
            group = nn.ModuleList([
                ResFCTemporalBlock(hidden_size, hidden_size, time_channels=hidden_size, activation=build_activation(activation), norm_type=norm_type),
                ResFCTemporalBlock(hidden_size, hidden_size, time_channels=hidden_size, activation=build_activation(activation), norm_type=norm_type),
            ])
            self.down_groups_2.append(group)
        
        # 2.2. mid group
        # for miu
        self.mid_resblock1_1 = ResFCTemporalBlock(hidden_size, hidden_size, time_channels=hidden_size, activation=build_activation(activation), norm_type=norm_type)
        self.mid_resblock2_1 = ResFCTemporalBlock(hidden_size, hidden_size, time_channels=hidden_size, activation=build_activation(activation), norm_type=norm_type)
        # for sigma
        self.mid_resblock1_2 = ResFCTemporalBlock(hidden_size, hidden_size, time_channels=hidden_size, activation=build_activation(activation), norm_type=norm_type)
        self.mid_resblock2_2 = ResFCTemporalBlock(hidden_size, hidden_size, time_channels=hidden_size, activation=build_activation(activation), norm_type=norm_type)
        
        # 2.3. up groups
        for _ in range(layer_num // 2):
            group = nn.ModuleList([
                ResFCTemporalBlock(hidden_size, hidden_size, time_channels=hidden_size, activation=build_activation(activation), norm_type=norm_type),
                ResFCTemporalBlock(hidden_size, hidden_size, time_channels=hidden_size, activation=build_activation(activation), norm_type=norm_type),
            ])
            self.up_groups.append(group)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, x):
        x = self.encoder(x)
        
        # vae encode
        # miu
        miu = x
        for resblock1, resblock2 in self.down_groups_1:
            miu = resblock1(miu, None)
            miu = resblock2(miu, None)
        miu = self.mid_resblock1_1(miu, None)
        miu = self.mid_resblock2_1(miu, None)
        # log_sigma
        log_sigma = x
        for resblock1, resblock2 in self.down_groups_2:
            log_sigma = resblock1(log_sigma, None)
            log_sigma = resblock2(log_sigma, None)
        log_sigma = self.mid_resblock1_2(log_sigma, None)
        log_sigma = self.mid_resblock2_2(log_sigma, None)
        
        z = self.reparameterize(miu, log_sigma)
        
        # vae decoder
        for resblock1, resblock2 in self.up_groups:
            z = resblock1(z, None)
            z = resblock2(z, None)
        
        recon_x = self.decoder(z)
        
        return recon_x, miu, log_sigma

    def loss_function(self, target_x, recon_x, miu, log_sigma):
        kld_weight = 1.
        recons_loss = F.mse_loss(recon_x, target_x)
        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_sigma - miu ** 2 - log_sigma.exp(), dim=1), dim=0)
        kld_loss = -0.5 * torch.sum(1 + log_sigma - miu ** 2 - log_sigma.exp(), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'reconstruction_loss': recons_loss, 'kld_loss': kld_loss}