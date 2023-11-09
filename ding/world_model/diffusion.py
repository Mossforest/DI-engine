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
from ding.world_model.model.diffusionnet import DiffusionNet

# ddpm
Tuple = lambda *args: tuple(args)
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def add_dict(tmp_dict, name, v):
    if name in tmp_dict:
        tmp_dict[name].append(v)
    else:
        tmp_dict[name] = [v]

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
        clip_denoised=False,
        beta_schedule='linear',
        rollout_length_scheduler=dict(
            type='linear',
            rollout_start_step=20000,
            rollout_end_step=150000,
            rollout_length_min=1,
            rollout_length_max=25,
        ),
        learn=dict(
            data_path=None,
            train_epoch=30000,
            batch_size=256,
            learning_rate=3e-4,
        ),
    )

    def __init__(self, cfg: dict, env: BaseEnv, tb_logger: 'SummaryWriter'):  # noqa
        WorldModel.__init__(self, cfg, env, tb_logger)
        nn.Module.__init__(self)
        self._cfg = cfg
        self.state_size = self._cfg.model.state_size
        self.action_size = self._cfg.model.action_size
        self.hidden_size = self._cfg.model.hidden_size
        self.background_size = self._cfg.model.background_size
        self.batch_size = self._cfg.learn.batch_size
        
        self.env = env
        self.tb_logger = tb_logger
        self.log_dict = {}
        
        # diffusion schedule
        self.n_timesteps = self._cfg.n_timesteps
        if self._cfg.beta_schedule == 'linear':
            scale = 1000 / self.n_timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            betas = torch.linspace(beta_start, beta_end, self.n_timesteps, dtype = torch.float64)
        elif self._cfg.beta_schedule == 'cosine':
            # cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
            s = 0.008
            steps = self.n_timesteps + 1
            x = torch.linspace(0, self.n_timesteps, steps, dtype = torch.float64)
            alphas_cumprod = torch.cos(((x / self.n_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0, 0.999)
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        
        self.model = DiffusionNet(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_size=self.hidden_size,
            n_timesteps=self.n_timesteps,
            background_size=self.background_size,
            layer_num=self._cfg.model.layer_num,
            activation='mish',
            norm_type='LN',
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self._cfg.learn.learning_rate)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self._cfg.learn.train_epoch)
        
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
        
        
        if self._cuda:
            self.cuda()
        
        
        ########### debug
        # print(f'alphas_cumprod: {self.alphas_cumprod[-20:]}')
        # print(f'sqrt_alphas_cumprod: {self.sqrt_alphas_cumprod[-20:]}')
        # print(f'sqrt_one_minus_alphas_cumprod: {self.sqrt_one_minus_alphas_cumprod[-20:]}')
        # print(f'log_one_minus_alphas_cumprod: {self.log_one_minus_alphas_cumprod[-20:]}')
        # exit()
        

    def loss_fn(self, pred, targ):
        loss = F.mse_loss(pred, targ, reduction='none')
        info = {
            'loss': loss.mean().item(),
            # 'mean_pred': pred.mean().item(), 'mean_targ': targ.mean().item(),
            # 'min_pred': pred.min().item(), 'min_targ': targ.min().item(),
            # 'max_pred': pred.max().item(), 'max_targ': targ.max().item(),
        }
        return loss, info

    # model.train with dataset w.o. buffer
    def train(self, data: dict, epoch: int, step: int):
        r"""
        Overview:
            Train world model using data from env_buffer.

        Arguments:
            - env_buffer (:obj:`IBuffer`): the buffer which collects real environment steps
            - envstep (:obj:`int`): the current number of environment steps in real environment
            - train_iter (:obj:`int`): the current number of policy training iterations
        """
        data = default_collate(data)
        data['done'] = data['done'].float()
        data['weight'] = data.get('weight', None)
        obs = data['obs'].to(torch.float32)
        action = data['action'].to(torch.float32)
        next_obs = data['next_obs'].to(torch.float32)
        background = data['background'].to(torch.float32)
        
        # no action
        action = torch.full(action.shape, 0, dtype=torch.float32)
        background = torch.full(background.shape, 0, dtype=torch.float32)
        next_obs = torch.clone(obs)
        
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
            background = background.cuda()
        
        # sample and model
        if self.batch_size != x_start.shape[0]:
            return
        t = torch.randint(0, self.n_timesteps, (self.batch_size,), device=x_start.device).long()
        noise = torch.randn_like(x_start)
        x_noisy = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        x_recon = self.model(x_noisy, cond_a, cond_s, t, background)
        assert x_start.shape == x_recon.shape
        loss, logvar = self.loss_fn(x_recon, noise)
        
        
        # train with loss
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.log_grad(epoch, step)
        self.optimizer.step()
        lr = self.optimizer.param_groups[0]['lr']
        # log
        if self.tb_logger is not None:
            for k, v in logvar.items():
                name = 'train_model' + k
                add_dict(self.log_dict, name, v)
            add_dict(self.log_dict, 'train_model/learning_rate', lr)

    #--------------------------------------- eval ---------------------------------------#

    @torch.no_grad()
    def p_sample_fn(self, x, cond_a, cond_s, t, background):
        # p mean variance
        # 1. predict start from noise
        x_t = x
        noise = self.model(x_t, cond_a, cond_s, t, background)
        x_recon = (
                extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise
            )
        # if t[0] % 100==99:
        #     print(f'NOISE  : {noise[0]}')
        #     print(f't: {t[0]}')
        #     print(f'1: {extract(self.sqrt_recip_alphas_cumprod, t, x.shape)[0] }')
        #     print(f'X_RECON: {x_recon[0]}')
        if self._cfg.clip_denoised:  # TODO
            x_recon.clamp_(-1., 1.)

        # 2. q posterior
        x_start = x_recon
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        # p mean variance end
        
        posterior_std = torch.exp(0.5 * posterior_log_variance_clipped)
        # no added noise when t == 0
        noise_add = torch.randn_like(x)
        noise_add[t == 0] = 0
        return posterior_mean + posterior_std * noise_add


    def eval(self, data: dict, step: int):
        r"""
        Overview:
            Evaluate world model using data from env_buffer.

        Arguments:
            - env_buffer (:obj:`IBuffer`): the buffer that collects real environment steps
            - envstep (:obj:`int`): the current number of environment steps in real environment
            - train_iter (:obj:`int`): the current number of policy training iterations
        """
        data = default_collate(data)
        data['done'] = data['done'].float()
        data['weight'] = data.get('weight', None)
        obs = data['obs'].to(torch.float32)
        action = data['action'].to(torch.float32)
        next_obs = data['next_obs'].to(torch.float32)
        background = data['background'].to(torch.float32)
        
        # no action
        # action = torch.full(action.shape, 0, dtype=torch.float32)
        # background = torch.full(background.shape, 0, dtype=torch.float32)
        # next_obs = torch.clone(obs)
        
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
            background = background.cuda()
            
        if self.batch_size != x_start.shape[0]:
            return
        
        # evaluation
        with torch.no_grad():
            x = torch.randn(x_start.shape, device=x_start.device)
            for i in reversed(range(0, self.n_timesteps)):
                t = torch.full((self.batch_size,), i, dtype=torch.long, device=x_start.device)
                x = self.p_sample_fn(x, cond_a, cond_s, t, background)
            x_recon_overall = x
            loss, logvar = self.loss_fn(x_recon_overall, x_start)
        # log
        if self.tb_logger is not None:
            for k, v in logvar.items():
                name = 'eval_model/overall_' + k
                add_dict(self.log_dict, name, v)
        
        # [v0.4] new eval 1
        with torch.no_grad():
            t = torch.randint(0, self.n_timesteps, (self.batch_size,), device=x_start.device).long()
            t_end = torch.full((self.batch_size,), self.n_timesteps - 1, dtype=torch.long, device=x_start.device)
            noise = torch.randn_like(x_start)
            x_noisy = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
            )
            x_noisy_end = (
                extract(self.sqrt_alphas_cumprod, t_end, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t_end, x_start.shape) * noise
            )
            noise_recon = self.model(x_noisy, cond_a, cond_s, t, background)
            assert x_start.shape == noise_recon.shape
            loss, logvar = self.loss_fn(noise_recon, noise)
        if self.tb_logger is not None:
            for k, v in logvar.items():
                name = 'eval_model/noise_' + k
                add_dict(self.log_dict, name, v)
        
        # [v0.4] new eval 2
        with torch.no_grad():
            x = x_noisy_end
            for i in reversed(range(0, self.n_timesteps)):
                t = torch.full((self.batch_size,), i, dtype=torch.long, device=x_start.device)
                x = self.p_sample_fn(x, cond_a, cond_s, t, background)
            x_recon_fixgauss = x
            loss, logvar = self.loss_fn(x_recon_fixgauss, x_start)
        if self.tb_logger is not None:
            for k, v in logvar.items():
                name = 'eval_model/fixgauss_' + k
                add_dict(self.log_dict, name, v)
        
        # [v0.4] new eval 3
        with torch.no_grad():
            loss, logvar = self.loss_fn(x_recon_fixgauss, x_recon_overall)
        if self.tb_logger is not None:
            for k, v in logvar.items():
                name = 'eval_model/fixornot_' + k
                add_dict(self.log_dict, name, v)
        
        # [v0.4] new eval 4
        with torch.no_grad():
            x = torch.randn(x_start.shape, device=x_start.device)
            for i in reversed(range(0, self.n_timesteps)):
                t = torch.full((self.batch_size,), i, dtype=torch.long, device=x_start.device)
                x = self.p_sample_fn(x, cond_a, cond_s, t, background)
            x_recon_nofix = x
            loss, logvar = self.loss_fn(x_recon_overall, x_recon_nofix)
        if self.tb_logger is not None:
            for k, v in logvar.items():
                name = 'eval_model/nofix_' + k
                add_dict(self.log_dict, name, v)

    def step(self, obs: Tensor, action: Tensor):
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
    
    def epoch_log(self, epoch):
        for k in self.log_dict:
            v = torch.Tensor(self.log_dict[k])
            self.tb_logger.add_scalar(k, v.mean(), epoch)
        self.log_dict = {}

    def print_grad(self, step):
        print(f'\n\n\n==========================  {step}  ==========================\n')
        for idx, item in enumerate(self.model.named_parameters()):
            # h = item[1].register_hook(lambda grad: print(grad[:20]))
            try:
                grad = item[1].grad.data
                grad = grad.abs()
                print('{0}.{1:40} ==> {2:10.2}, {3:10.2}'.format(idx, item[0], grad.mean(), grad.std()))
            except AttributeError:
                continue
        print(f'\n==========================  {step}  ==========================\n\n\n')
    
    def log_grad(self, epoch, step):
        if self.tb_logger is not None and step % 500 == 0:
            for idx, item in enumerate(self.model.named_parameters()):
                try:
                    grad = item[1].grad.data
                    grad = grad.abs()
                    name = 'train_grad/' + item[0]
                    add_dict(self.log_dict, name, grad.mean())
                except AttributeError:
                    continue
        
        if self.tb_logger is not None and epoch % 50 == 0 and step % 500 == 0:
            for idx, item in enumerate(self.model.named_parameters()):
                # h = item[1].register_hook(lambda grad: print(grad[:20]))
                try:
                    grad = item[1].grad.data
                    grad = grad.abs()
                    self.tb_logger.add_scalar(f'train_grad/epoch_{epoch}', grad.mean(), idx)
                except AttributeError:
                    continue
    
    def save_model(self, file):
        torch.save(self.state_dict(), file)
    
    def load_model(self, file):
        self.load_state_dict(torch.load(file))