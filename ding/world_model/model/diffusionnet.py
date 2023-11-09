import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from ding.torch_utils.network.nn_module import MLP
from ding.torch_utils.network.activation import build_activation
from ding.torch_utils.network.res_block import ResFCTemporalBlock
from ding.torch_utils.network.merge import FiLM
from ding.torch_utils.time_emb import GaussianFourierProjection, SinusoidalPosEmb



class DiffusionNet(nn.Module):
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int,
        n_timesteps: int,
        background_size: int,
        layer_num: int,
        activation: str = 'mish',
        norm_type: str = 'LN'
    ):
        super().__init__()
        
        assert layer_num % 2 == 1, f"The layer num of diffusion net must be odd, but got {layer_num}!"
        
        self.n_timesteps = n_timesteps
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
        self.down_groups = nn.ModuleList([])
        self.up_groups = nn.ModuleList([])
        
        # 2.1. down groups
        for _ in range(layer_num // 2):
            group = nn.ModuleList([
                ResFCTemporalBlock(hidden_size, hidden_size, time_channels=hidden_size, activation=build_activation(activation), norm_type=norm_type),
                ResFCTemporalBlock(hidden_size, hidden_size, time_channels=hidden_size, activation=build_activation(activation), norm_type=norm_type),
            ])
            self.down_groups.append(group)
        
        # 2.2. mid group
        self.mid_resblock1 = ResFCTemporalBlock(hidden_size, hidden_size, time_channels=hidden_size, activation=build_activation(activation), norm_type=norm_type)
        self.mid_resblock2 = ResFCTemporalBlock(hidden_size, hidden_size, time_channels=hidden_size, activation=build_activation(activation), norm_type=norm_type)
        
        # 2.3. up groups
        for _ in range(layer_num // 2):
            group = nn.ModuleList([
                ResFCTemporalBlock(hidden_size * 2, hidden_size, time_channels=hidden_size, activation=build_activation(activation), norm_type=norm_type),
                ResFCTemporalBlock(hidden_size, hidden_size, time_channels=hidden_size, activation=build_activation(activation), norm_type=norm_type),
            ])
            self.up_groups.append(group)
        
        # 3. temporal embedding
        self.time_mlp = nn.Sequential(
            GaussianFourierProjection(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),    # TODO
            build_activation(self.activation),
            nn.Linear(hidden_size * 4, hidden_size)
        )
    
    def forward(self, x, cond_a, cond_s, t, background):
        t = self.time_mlp(t / self.n_timesteps)
        crop_store = []
        
        x = self.encoder(x)
        
        for resblock1, resblock2 in self.down_groups:
            x = resblock1(x, t)
            x = resblock2(x, t)
            crop_store.append(x)
        
        x = self.mid_resblock1(x, t)
        x = self.mid_resblock2(x, t)
        
        for resblock1, resblock2 in self.up_groups:
            x = torch.cat((x, crop_store.pop()), dim=1)
            x = resblock1(x, t)
            x = resblock2(x, t)
        
        x = self.decoder(x)
        
        return x