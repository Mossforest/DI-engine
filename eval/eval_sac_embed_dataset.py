from typing import Union, Tuple, Dict
import os
import time
import numpy as np
import random
import torch
import treetensor.torch as ttorch
import h5py
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from functools import partial
from tensorboardX import SummaryWriter
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

from ding.envs import get_vec_env_setting, create_env_manager
# from ding.worker import BaseLearner, InteractionSerialEvaluator
from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.world_model import create_world_model
from ding.utils import set_pkg_seed
from ding.torch_utils import to_ndarray, get_shape0
from pathlib import Path
from ding.framework.middleware.functional.evaluator import VectorEvalMonitor

class HDF5Dataset(Dataset):

    def __init__(self, data_path):
        # if 'dataset' in cfg:
        #     self.context_len = cfg.dataset.context_len
        # else:
        #     self.context_len = 0
        data = h5py.File(data_path, 'r')
        self._load_data(data)
        self._norm_data()
        self._cal_statistics()

    def __len__(self) -> int:
        return len(self._data['obs'])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {k: self._data[k][idx] for k in self._data.keys()}

    def _load_data(self, dataset: Dict[str, np.ndarray]) -> None:
        self._data = {}
        for k in dataset.keys():
            self._data[k] = dataset[k][:]

    def _norm_data(self):    # TODO: also do norm in eval before world model, re-norm after model
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


def norm_data(obs, env='bipedalwalker'):
    if env == 'bipedalwalker':
        obs_high = torch.Tensor([3.14, 5., 5., 5., 3.14, 5., 3.14, 5., 5., 3.14, 5., 3.14, 5., 5., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]).to(obs.device)
        obs_low  = torch.Tensor([-3.14, -5., -5., -5., -3.14, -5., -3.14, -5., -0., -3.14, -5., -3.14, -5., -0., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.]).to(obs.device)
        # press to [-1, 1]
        obs = (2 * obs - obs_high - obs_low) / (obs_high - obs_low)
        return obs
    else:
        assert False, f"no support norm for {env}!"


def norm_data_restore(obs, env='bipedalwalker'):
    if env == 'bipedalwalker':
        obs_high = torch.Tensor([3.14, 5., 5., 5., 3.14, 5., 3.14, 5., 5., 3.14, 5., 3.14, 5., 5., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]).to(obs.device)
        obs_low  = torch.Tensor([-3.14, -5., -5., -5., -3.14, -5., -3.14, -5., -0., -3.14, -5., -3.14, -5., -0., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.]).to(obs.device)
        obs = ((obs_high - obs_low) * obs + obs_high + obs_low) / 2
        for i, val in enumerate(obs):
            obs[i] = min(max(obs_low[i], val), obs_high[i])
        return obs
    else:
        assert False, f"no support norm for {env}!"


def serial_pipeline(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0
):
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = deepcopy(input_cfg)
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)
    
    print(f'============== exp name: {cfg.exp_name}')
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    
    # agent
    policy = create_policy(cfg.policy, enable_field=['eval', 'command'])
    pre_policy = torch.load(cfg.policy.eval.state_dict_path, map_location=policy.device)
    policy.eval_mode.load_state_dict(pre_policy)
    policy = policy.eval_mode
    
    # dataset
    train_dataset = HDF5Dataset(cfg.policy.collect.train_data_path)
    # eval_dataset = HDF5Dataset(cfg.policy.collect.eval_data_path, cfg.policy.collect.ignore_dim.copy())
    
    
    embed_obs_traj = []
    embed_obs_box = []
    print('start eval...')
    for train_data in train_dataset:
        obs = train_data['obs']
        done = train_data['done']
        obs = torch.as_tensor(obs).to(dtype=torch.float32)
        inference_output = policy.forward({0: obs})
        output = [v for v in inference_output.values()]
        embed_obs = [to_ndarray(v['embed_obs']) for v in output][0]
        embed_obs_box.append(embed_obs)
        if done:
            embed_obs_traj.append(1)
            np.save(f'./{cfg.exp_name}/embed_obs_box_{len(embed_obs_traj)}', np.stack(embed_obs_box))
            if len(embed_obs_traj) == 50:
                break
    
    print('Done.')





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=10)
    parser.add_argument('--config', '-c', type=str, default='eval_sac_embed_dataset_config.py')
    args = parser.parse_args()
    config = Path(__file__).absolute().parent / 'config' / args.config
    config = read_config(str(config))
    config[0].exp_name = config[0].exp_name.replace('0', str(args.seed))
    serial_pipeline(config, seed=args.seed)
