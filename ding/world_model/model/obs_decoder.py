import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from ding.torch_utils import MLP



class HiddenDecoder(nn.Module):
    
    def __init__(
        self,
        output_size: int = 24,
        hidden_size: int = 128,
        layer_num: int = 1,
        activation: nn.Module = nn.Tanh(), # nn.ReLU(), 
        norm_type: str = None,
        discrete: list = [8, 13],
        discrete_activation: nn.Module = nn.Sigmoid(),
    ):
        super().__init__()
        self.discrete = discrete
        self.mlp = MLP(hidden_size, hidden_size, hidden_size, layer_num, activation=activation, norm_type=norm_type)
        self.out = nn.Sequential(nn.Linear(hidden_size, output_size-len(self.discrete)), activation)
        self.discrete_out = []
        for _ in self.discrete:
            self.discrete_out.append(nn.Sequential(nn.Linear(hidden_size, 1), discrete_activation))
    
    def forward(self, x):
        x = self.mlp(x)
        x_continuous = self.out(x)
        x_discrete = []
        for net in self.discrete_out:
            net.to('cuda')
            x_discrete.append(net(x))
        return {'continuous': x_continuous, 'discrete': x_discrete}
