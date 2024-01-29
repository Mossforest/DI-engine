import numpy as np
import copy
import math
import tqdm
import os
import random
import matplotlib.pyplot as plt

from pathlib import Path
# from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
import torch.nn.functional as F


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def add_dict(tmp_dict, name, v):
    if name in tmp_dict:
        tmp_dict[name].append(v)
    else:
        tmp_dict[name] = [v]

def draw(x, name, path):
    plt.figure()
    plt.plot(x)
    plt.title(name)
    plt.savefig(f'{path}/{name}.png')
    plt.close()


n_timesteps = 1000
s = 0.008
steps = n_timesteps + 1
x = torch.linspace(0, n_timesteps, steps, dtype = torch.float64)
# alphas_cumprod = torch.cos(((x / n_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
# alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
# betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
# betas = torch.clip(betas, 0, 0.999)


# alternative
# def cosine_schedule(t, start=0, end=1, tau=1, clip_min=1e-9):
#     # A gamma function based on cosine function.
#     v_start = math.cos(start * math.pi / 2) ** (2 * tau)
#     v_end = math.cos(end * math.pi / 2) ** (2 * tau)
#     output = torch.cos((t * (end - start) + start) * math.pi / 2) ** (2 * tau)
#     output = (v_end - output) / (v_end - v_start)
#     return torch.clip(output, clip_min, 1.)

# alphas_cumprod = cosine_schedule(x / n_timesteps, start=0.2, end=1, tau=1)


# alternative - sigmoid
def sigmoid_schedule(t, start=-3, end=3, tau=1.0, clip_min=1e-9):
    # A gamma function based on sigmoid function.
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    v_start = sigmoid(start / tau)
    v_end = sigmoid(end / tau)
    output = torch.sigmoid((t * (end - start) + start) / tau)
    output = (v_end - output) / (v_end - v_start)
    return torch.clip(output, clip_min, 1.)

alphas_cumprod = sigmoid_schedule(x / n_timesteps)

# alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
betas = torch.clip(betas, 0, 0.999)

alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

# helper function to register buffer from float64 to float32
register_buffer = lambda name, val: register_buffer(name, val.to(torch.float32))

betas = betas
alphas_cumprod = alphas_cumprod
alphas_cumprod_prev = alphas_cumprod_prev
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod)
sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
posterior_variance = posterior_variance
posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min =1e-20))
posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
loss_weight = betas ** 2 / (2 * posterior_log_variance_clipped ** 2 * alphas * (1 - alphas_cumprod))
loss_weight_simp = 0.5 * betas / (alphas - alphas_cumprod)


coef11 = np.array(sqrt_recip_alphas_cumprod)
coef12 = np.array(sqrt_recipm1_alphas_cumprod)
coef21 = np.array(posterior_mean_coef1)
coef22 = np.array(posterior_mean_coef2)
coef3 = np.array(torch.exp(0.5 * posterior_log_variance_clipped))

path = f'./eval/check_schedule_img/sigmoid_step{n_timesteps}'
if not os.path.exists(path):
    os.mkdir(path)
draw(coef11, 'coef11', path)
draw(coef12, 'coef12', path)
draw(coef21, 'coef21', path)
draw(coef22, 'coef22', path)
draw(coef3, 'coef3', path)
draw(np.array(alphas_cumprod), 'alphas_cumprod', path)
draw(np.array(alphas), 'alphas', path)
draw(np.sqrt(np.array(alphas)), 'alphas_sqrt', path)
draw(np.sqrt(np.array(1-alphas)), '1_alphas_sqrt', path)
draw(np.array(loss_weight_simp[:800]), 'loss_weight_simp_n2', path)
draw(np.array(loss_weight_simp), 'loss_weight_simp', path)
draw(np.array(loss_weight[:800]), 'loss_weight_n2', path)
draw(np.array(loss_weight), 'loss_weight', path)
print(loss_weight[-10:])
print(loss_weight_simp[-10:])
draw(np.array(betas), 'betas', path)