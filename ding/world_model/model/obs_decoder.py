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
    ):
        super().__init__()
        self.mlp = MLP(hidden_size, hidden_size, hidden_size, layer_num, activation=activation, norm_type=norm_type)
        self.out = nn.Sequential(nn.Linear(hidden_size, output_size), activation)
    
    def forward(self, x):
        x = self.mlp(x)
        x = self.out(x)
        return x
