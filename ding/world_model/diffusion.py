import numpy as np
import copy
import math
import tqdm

from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.cuda.amp import autocast

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from ding.utils import WORLD_MODEL_REGISTRY, lists_to_dicts
from ding.utils.data import default_collate
from ding.worker import IBuffer
from ding.envs import BaseEnv
from ding.model import ConvEncoder
from ding.world_model.base_world_model import WorldModel
from ding.world_model.model.networks import RSSM, ConvDecoder
from ding.torch_utils import to_device
from ding.torch_utils.network.dreamer import DenseHead

# ddpm
Tuple = lambda *args: tuple(args)
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


@WORLD_MODEL_REGISTRY.register('diffusion')
class DiffusionWorldModel(WorldModel, nn.Module):
    r"""
    Overview:
        Single-step diffusion demo for world model

    Interfaces:
        should_train, should_eval, train, eval, step
    """
    config = dict(
        train_freq=250,  # w.r.t environment step
        eval_freq=250,  # w.r.t environment step
        cuda=True,
        n_timesteps=1000,
        rollout_length_scheduler=dict(
            type='linear',
            rollout_start_step=20000,
            rollout_end_step=150000,
            rollout_length_min=1,
            rollout_length_max=25,
        )
    )

    def __init__(self, cfg: dict, env: BaseEnv, tb_logger: 'SummaryWriter'):  # noqa
        self.state_size = cfg.state_size
        self.action_size = cfg.action_size
        self.reward_size = cfg.reward_size
        self.hidden_size = cfg.hidden_size
        self.batch_size = cfg.batch_size
        
        # diffusion schedule
        self.n_timesteps = cfg.n_timesteps
        if self.beta_schedule == 'linear':
            betas = linear_beta_schedule(self.n_timesteps)
        elif self.beta_schedule == 'cosine':
            betas = cosine_beta_schedule(self.n_timesteps)
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        
        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        
        self.model = DiffusionModelNet()  # TODO net class
        if self._cuda:
            self.cuda()
        

    def loss_fn(pred, targ):
        loss = F.mse_loss(pred, targ, reduction='none')
        info = {
            'loss': loss.mean().item(),
            'mean_pred': pred.mean().item(), 'mean_targ': targ.mean().item(),
            'min_pred': pred.min().item(), 'min_targ': targ.min().item(),
            'max_pred': pred.max().item(), 'max_targ': targ.max().item(),
        }
        return loss, info

    def q_sample(self, x_start, t, noise=None):
        if noise is None:   # TODO
            noise = torch.randn_like(x_start)
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start, cond_a, cond_s, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.model(x_noisy, cond_a, cond_s, t)
        assert x_start.shape == x_recon.shape

        if self.predict_epsilon:    # model outputs noise
            loss, info = self.loss_fn(x_recon, noise)
        else:                       # model outputs x_0
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def train(self, env_buffer: IBuffer, envstep: int, train_iter: int):
        r"""
        Overview:
            Train world model using data from env_buffer.

        Arguments:
            - env_buffer (:obj:`IBuffer`): the buffer which collects real environment steps
            - envstep (:obj:`int`): the current number of environment steps in real environment
            - train_iter (:obj:`int`): the current number of policy training iterations
        """
        data = env_buffer.sample(env_buffer.count(), train_iter)
        data = default_collate(data)
        data['done'] = data['done'].float()
        data['weight'] = data.get('weight', None)
        obs = data['obs']
        action = data['action']
        reward = data['reward']
        next_obs = data['next_obs']
        if len(reward.shape) == 1:
            reward = reward.unsqueeze(1)
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        # build train samples
        x_start = next_obs
        cond_a = action
        cond_s = obs
        if self._cuda:
            x_start = x_start.cuda()
            cond_a = cond_a.cuda()
            cond_s = cond_s.cuda()
        
        # sample and train
        t = torch.randint(0, self.n_timesteps, (self.batch_size,), device=x_start.device).long()
        loss, logvar = self.p_losses(x_start, cond_a,cond_s, t)
        
        self.model.train(loss)  # TODO: below in net
            # def train(self, loss: torch.Tensor):
            # self.optimizer.zero_grad()
            # loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
            # if self.use_decay:
            #     loss += self.get_decay_loss()
            # loss.backward()
            # self.optimizer.step()
        self.last_train_step = envstep
        # log
        if self.tb_logger is not None:
            for k, v in logvar.items():
                self.tb_logger.add_scalar('env_model_step/' + k, v, envstep)

    #------------------------------------------ eval ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise  # if not self.predict_noise, the model directly output x_0 which was named as 'noise'

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond_a, cond_s, t):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond_a, cond_s, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def default_sample_fn(self, x, cond_a, cond_s, t):
        model_mean, _, model_log_variance = self.p_mean_variance(x, cond_a, cond_s, t)
        model_std = torch.exp(0.5 * model_log_variance)
        # no noise when t == 0
        noise = torch.randn_like(x)
        noise[t == 0] = 0
        return model_mean + model_std * noise
    
    @torch.no_grad()
    def p_sample_loop(self, cond_a, cond_s, sample_fn=default_sample_fn):
        def make_timesteps(batch_size, i, device):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            return t
        
        device = self.betas.device  # TODO
        x = torch.randn(cond_s, device=device)
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(self.batch_size, i, device)
            x = sample_fn(self, x, cond_a, cond_s, t)
        return x

    @torch.no_grad()
    def p_eval(self, x_start, cond_a, cond_s):
        x_recon = self.p_sample_loop(self, cond_a, cond_s)
        loss, info = self.loss_fn(x_recon, x_start)
        return loss, info

    def eval(self, env_buffer: IBuffer, envstep: int, train_iter: int):
        r"""
        Overview:
            Evaluate world model using data from env_buffer.

        Arguments:
            - env_buffer (:obj:`IBuffer`): the buffer that collects real environment steps
            - envstep (:obj:`int`): the current number of environment steps in real environment
            - train_iter (:obj:`int`): the current number of policy training iterations
        """
        data = env_buffer.sample(env_buffer.count(), train_iter)
        data = default_collate(data)
        data['done'] = data['done'].float()
        data['weight'] = data.get('weight', None)
        obs = data['obs']
        action = data['action']
        reward = data['reward']
        next_obs = data['next_obs']
        if len(reward.shape) == 1:
            reward = reward.unsqueeze(1)
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        # build train samples
        x_start = next_obs
        cond_a = action
        cond_s = obs
        if self._cuda:
            x_start = x_start.cuda()
            cond_a = cond_a.cuda()
            cond_s = cond_s.cuda()
        
        loss, logvar = self.p_eval(self, x_start, cond_a, cond_s)
        
        self.last_train_step = envstep
        # log
        if self.tb_logger is not None:
            for k, v in logvar.items():
                self.tb_logger.add_scalar('env_model_step/' + k, v, envstep)

    def step(self, obs: Tensor, action: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        Overview:
            Take one step in world model.

        Arguments:
            - obs (:obj:`torch.Tensor`): current observations :math:`S_t`
            - action (:obj:`torch.Tensor`): current actions :math:`A_t`

        Returns:
            - reward (:obj:`torch.Tensor`): rewards :math:`R_t`
            - next_obs (:obj:`torch.Tensor`): next observations :math:`S_t+1`
            - done (:obj:`torch.Tensor`): whether the episodes ends

        Shapes:
            :math:`B`: batch size
            :math:`O`: observation dimension
            :math:`A`: action dimension

            - obs:      [B, O]
            - action:   [B, A]
            - reward:   [B, ]
            - next_obs: [B, O]
            - done:     [B, ]
        """
        raise NotImplementedError

