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
from ding.world_model.model.vaenet import DiffusionNet

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

@WORLD_MODEL_REGISTRY.register('diffusion_vae')
class DiffusionVAEModel(WorldModel, nn.Module):
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
        
        
        self.model = DiffusionNet(
            state_size=self.state_size,
            hidden_size=self.hidden_size,
            layer_num=self._cfg.model.layer_num,
            activation='mish',
            norm_type='LN',
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self._cfg.learn.learning_rate)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self._cfg.learn.train_epoch)
        
        if self._cuda:
            self.cuda()


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
        obs = data['obs'].to(torch.float32)
        # build train samples
        if self._cuda:
            obs = obs.cuda()
        
        # sample and model
        if self.batch_size != obs.shape[0]:
            return
        x_recon, miu, log_sigma = self.model(obs)
        assert obs.shape == x_recon.shape
        logvar = self.model.loss_function(obs, x_recon, miu, log_sigma)
        loss = logvar['loss']
        
        # train with loss
        self.optimizer.zero_grad()
        loss.backward()
        self.log_grad(epoch, step)
        self.optimizer.step()
        lr = self.optimizer.param_groups[0]['lr']
        # log
        if self.tb_logger is not None:
            for k, v in logvar.items():
                name = 'train_model/' + k
                add_dict(self.log_dict, name, v)
            add_dict(self.log_dict, 'train_model/learning_rate', lr)

    #--------------------------------------- eval ---------------------------------------#


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
        obs = data['obs'].to(torch.float32)
        # build train samples
        if self._cuda:
            obs = obs.cuda()
        
        # sample and model
        if self.batch_size != obs.shape[0]:
            return
        x_recon, miu, log_sigma = self.model(obs)
        assert obs.shape == x_recon.shape
        logvar = self.model.loss_function(obs, x_recon, miu, log_sigma)
        
        # log
        if self.tb_logger is not None:
            for k, v in logvar.items():
                name = 'eval_model/' + k
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