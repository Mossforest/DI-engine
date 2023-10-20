import math
import torch
from torch import nn
import numpy as np


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps. source: CEP"""  
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    
    def forward(self, x):
        x_proj = x[..., None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class SinusoidalPosEmb(nn.Module):
    """source: Plan Diffuser"""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

if __name__ == "__main__":
    embed = GaussianFourierProjection(embed_dim=32)
    t=torch.tensor([0.1,0.2,0.3,0.4,0.5])
    print(embed(t).shape)