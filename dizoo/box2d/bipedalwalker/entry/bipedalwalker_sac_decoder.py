from typing import Union, Tuple, Dict
import os
import time
import numpy as np
import random
import torch
import h5py
from functools import partial
from easydict import EasyDict
from tensorboardX import SummaryWriter
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ding.envs import get_vec_env_setting, create_env_manager
# from ding.worker import BaseLearner, InteractionSerialEvaluator
from ding.config import read_config, compile_config
from ding.world_model import create_world_model
from ding.utils import set_pkg_seed
from ding.torch_utils import Adam
from ding.utils.data import create_dataset
from ding.world_model.model.obs_decoder import HiddenDecoder


bipedalwalker_sac_config = dict(
    exp_name='exp_sac_decoder_train',
    cuda=True,
    policy=dict(),
    learn=dict(
        batch_size=256,
        learning_rate=0.0003,
        train_epoch=500,
    ),
    train_data_path='bipedalwalker_data_hidden/processed_hidden_train.npy',
)
bipedalwalker_sac_config = EasyDict(bipedalwalker_sac_config)
main_config = bipedalwalker_sac_config
bipedalwalker_sac_create_config = dict(
    env=dict(
        type='bipedalwalker',
        import_names=['dizoo.box2d.bipedalwalker.envs.bipedalwalker_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='sac', ),
    replay_buffer=dict(type='naive', ),
)
bipedalwalker_sac_create_config = EasyDict(bipedalwalker_sac_create_config)
create_config = bipedalwalker_sac_create_config




class HDF5Dataset(Dataset):

    def __init__(self, data_path):
        # data = h5py.File(data_path, 'r')
        data = np.load(data_path, allow_pickle=True)
        self._data = {k: np.stack([d[k] for d in data]) for k in data[0]}
        self._norm_data()

    def __len__(self) -> int:
        return len(self._data['obs'])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {k: self._data[k][idx] for k in self._data.keys()}

    def _norm_data(self):
        obs_high = np.array([3.14, 5., 5., 5., 3.14, 5., 3.14, 5., 5., 3.14, 5., 3.14, 5., 5., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        obs_low  = np.array([-3.14, -5., -5., -5., -3.14, -5., -3.14, -5., -0., -3.14, -5., -3.14, -5., -0., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])
        # press to [-1, 1]
        self._data['obs'] = (2 * self._data['obs'] - obs_high - obs_low) / (obs_high - obs_low)
        self._data['next_obs'] = (2 * self._data['next_obs'] - obs_high - obs_low) / (obs_high - obs_low)

def collate_fn(batch):
    merged_dict = {}
    for sample in batch:
        for key, value in sample.items():
            if key in merged_dict:
                merged_dict[key].append(value)
            else:
                merged_dict[key] = [value]
    for key in merged_dict.keys():
        merged_dict[key] = torch.Tensor(merged_dict[key])
    return merged_dict


if __name__ == "__main__":
    cfg, create_cfg = main_config, create_config
    cfg = compile_config(cfg, seed=3, auto=True, create_cfg=create_cfg)
    
    print(f'============== exp name: {cfg.exp_name}')
    
    # dataset
    train_dataset = HDF5Dataset(cfg.train_data_path)
    # eval_dataset = HDF5Dataset(cfg.eval_data_path)
    train_dataloader = DataLoader(
        train_dataset,
        cfg.learn.batch_size,
        shuffle=True,
        sampler=None,
        collate_fn=collate_fn,
        drop_last=True,
    )
    # eval_dataloader = DataLoader(
    #     eval_dataset,
    #     cfg.test.batch_size,
    #     shuffle=False,
    #     sampler=None,
    #     collate_fn=collate_fn,
    #     drop_last=True,
    # )
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    
    # Random seed
    set_pkg_seed(cfg.seed, use_cuda=cfg.cuda)
    decoder = HiddenDecoder()
    if torch.cuda.is_available() and cfg.cuda:
        decoder = decoder.to('cuda')
    optimizer = Adam(decoder.parameters(), lr=cfg.learn.learning_rate)
    
    print('start training...')
    for epoch in range(cfg.learn.train_epoch):
        t1 = time.time()
        print(f'length: {len(train_dataloader)}')
        for idx, train_data in enumerate(train_dataloader):
            hidden = train_data['hidden_obs']
            real = train_data['obs']
            if torch.cuda.is_available() and cfg.cuda:
                hidden = hidden.to('cuda')
                real = real.to('cuda')
            pred = decoder.forward(hidden)
            
            loss = F.mse_loss(pred, real)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tb_logger.add_scalar('train/loss', loss.detach().cpu().item(), epoch)

        if epoch % 20 == 0:
            path = f'{cfg.exp_name}/model'
            if not os.path.exists(path):
                os.mkdir(path)
            torch.save(decoder.state_dict(), f'{path}/epoch{epoch}')
        
        print(f'finished epoch {epoch}, {time.time() - t1:.2f} sec.')

    print('Done.')

