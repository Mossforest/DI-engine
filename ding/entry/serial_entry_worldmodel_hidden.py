from typing import Union, Tuple, Dict
import os
import time
import numpy as np
import random
import torch
import h5py
from functools import partial
from tensorboardX import SummaryWriter
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from ding.envs import get_vec_env_setting, create_env_manager
# from ding.worker import BaseLearner, InteractionSerialEvaluator
from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.world_model import create_world_model
from ding.utils import set_pkg_seed
from ding.utils.data import default_collate
from ding.torch_utils import to_ndarray

class HDF5Dataset(Dataset):

    def __init__(self, data_path, ignore_dim):
        # if 'dataset' in cfg:
        #     self.context_len = cfg.dataset.context_len
        # else:
        #     self.context_len = 0
        data = h5py.File(data_path, 'r')
        self._load_data(data)
        # self._norm_data()
        self._cal_statistics()
        
        # delete ignore_dim
        if ignore_dim == None:
            return
        ignore_dim = ignore_dim.copy()
        ignore_dim.reverse()
        for idx in ignore_dim:
            self._data['obs'] = np.delete(self._data['obs'], idx, axis=-1)
            self._data['next_obs'] = np.delete(self._data['next_obs'], idx, axis=-1)
        

    def __len__(self) -> int:
        return len(self._data['obs'])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {k: self._data[k][idx] for k in self._data.keys()}

    def _load_data(self, dataset: Dict[str, np.ndarray]) -> None:
        self._data = {}
        for k in dataset.keys():
            self._data[k] = dataset[k][:]

    def _norm_data(self):
        # renorm obs of bipedalwalker
        obs_high = np.array([3.14, 5., 5., 5., 3.14, 5., 3.14, 5., 5., 3.14, 5., 3.14, 5., 5., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        obs_low  = np.array([-3.14, -5., -5., -5., -3.14, -5., -3.14, -5., -0., -3.14, -5., -3.14, -5., -0., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])
        # press to [-1, 1]
        self._data['obs'] = (2 * self._data['obs'] - obs_high - obs_low) / (obs_high - obs_low)
        self._data['next_obs'] = (2 * self._data['next_obs'] - obs_high - obs_low) / (obs_high - obs_low)

    def _cal_statistics(self, eps=1e-3):
        self._mean = self._data['obs'].mean(0)
        self._std = self._data['obs'].std(0) + eps
        action_max = self._data['action'].max(0)
        action_min = self._data['action'].min(0)
        buffer = 0.05 * (action_max - action_min)
        action_max = action_max.astype(float) + buffer
        action_min = action_max.astype(float) - buffer
        self._action_bounds = np.stack([action_min, action_max], axis=0)

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def action_bounds(self) -> np.ndarray:
        return self._action_bounds


def draw(x, y, name, path):
    x = np.array(x)
    y = np.array(y)
    plt.figure()
    plt.scatter(x, y)
    plt.title(name)
    plt.savefig(f'{path}/{name}.png')
    plt.close()


def encoding(obs, policy):
    obs = torch.as_tensor(obs).to(dtype=torch.float32)
    inference_output = policy.forward({0: obs})
    output = [v for v in inference_output.values()]
    embed_obs = output[0]['embed_obs']
    return embed_obs


def serial_pipeline_worldmodel_hidden(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0
):
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = deepcopy(input_cfg)
    # create_cfg.world_model.type = create_cfg.world_model.type + '_command'
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)
    
    print(f'============== exp name: {cfg.exp_name}')
    
    # dataset
    train_dataset = HDF5Dataset(cfg.policy.collect.train_data_path, cfg.policy.collect.ignore_dim)
    eval_dataset = HDF5Dataset(cfg.policy.collect.eval_data_path, cfg.policy.collect.ignore_dim)
    train_dataloader = DataLoader(
        train_dataset,
        cfg.world_model.learn.batch_size,
        shuffle=True,
        sampler=None,
        collate_fn=lambda x: x,
        drop_last=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        cfg.world_model.test.batch_size,
        shuffle=False,
        sampler=None,
        collate_fn=lambda x: x,
        drop_last=True,
    )
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    
    # Env
    env_fn, _, _ = get_vec_env_setting(cfg.env, collect=False, eval_=False)
    # Random seed
    set_pkg_seed(cfg.seed, use_cuda=cfg.world_model.cuda)
    
    # agent
    policy = create_policy(cfg.policy, enable_field=['eval'])
    pre_policy = torch.load(cfg.policy.eval.state_dict_path, map_location=policy.device)
    policy.eval_mode.load_state_dict(pre_policy)
    policy = policy.eval_mode
    
    # world_model
    world_model = create_world_model(cfg.world_model, env_fn(cfg.env), tb_logger)

    print('start training...')
    for epoch in range(cfg.world_model.learn.train_epoch):
        t1 = time.time()
        print(f'length: {len(train_dataloader)}')
        for idx, train_data in enumerate(train_dataloader):
            train_data = default_collate(train_data)
            # 1. go through encoder
            train_data['obs'] = encoding(train_data['obs'], policy)
            train_data['next_obs'] = encoding(train_data['next_obs'], policy)
            # 2. go through world model
            world_model.train(train_data, epoch, idx)
        world_model.scheduler.step()
        
        if epoch % (cfg.world_model.learn.train_epoch // cfg.world_model.test.test_epoch) == 0:
            for idx, eval_data in enumerate(eval_dataloader):
                eval_data = default_collate(eval_data)
                # 1. go through encoder
                eval_data['obs'] = encoding(eval_data['obs'], policy)
                eval_data['next_obs'] = encoding(eval_data['next_obs'], policy)
                world_model.eval(eval_data, epoch)
                if idx % 100 == 0:
                    print(f'finish eval in epoch {epoch}')
        world_model.epoch_log(epoch)

        if epoch % 10 == 0:
            path = f'{cfg.exp_name}/model'
            if not os.path.exists(path):
                os.mkdir(path)
            world_model.save_model(f'{path}/epoch{epoch}')
        
        print(f'============== exp name: {cfg.exp_name}')
        print(f'finished epoch {epoch}, {time.time() - t1:.2f} sec.')

    print('Done.')
    return world_model
