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

    def __init__(self, data_path, ignore_dim):
        # if 'dataset' in cfg:
        #     self.context_len = cfg.dataset.context_len
        # else:
        #     self.context_len = 0
        data = h5py.File(data_path, 'r')
        self._load_data(data)
        self._norm_data()
        self._cal_statistics()
        
        # delete ignore_dim
        ignore_dim.reverse()
        print(ignore_dim)
        for idx in ignore_dim:
            print(f'delete {ignore_dim}, ')
            self._data['obs'] = np.delete(self._data['obs'], idx, axis=-1)
            self._data['next_obs'] = np.delete(self._data['next_obs'], idx, axis=-1)
        
        print(f'--debug--: data shape: {self._data["obs"].shape}')

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
    
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    
    # part 1. =========== agent & real env ===========
    
    # Env_1
    env_fn, _, evaluator_env_cfg = get_vec_env_setting(deepcopy(cfg.env), collect=False, eval_=True)
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    
    # agent
    policy = create_policy(cfg.policy, enable_field=['eval', 'command'])
    pre_policy = torch.load(cfg.policy.eval.state_dict_path, map_location=policy.device)
    policy.eval_mode.load_state_dict(pre_policy)
    policy = policy.eval_mode
    
    # world_model
    world_model = create_world_model(cfg.world_model, env_fn(cfg.env), tb_logger)
    world_model.load_model(cfg.world_model.test.state_dict_path)
    
    print('start eval...')
    for _ in range(cfg.policy.eval.test_epoch):
        track_time = 0
        instep_draw_time = 0
        # 1. real world env
        if evaluator_env.closed:
            evaluator_env.launch()
        else:
            evaluator_env.reset()
        eval_monitor = VectorEvalMonitor(evaluator_env.env_num, cfg.env.n_evaluator_episode)
        vae_loss_var = {'real_obs': [], 'diffusion_obs': []}

        obs = torch.as_tensor(evaluator_env.ready_obs[0]).to(dtype=torch.float32)
        vae_loss_var['diffusion_obs'].append(obs.cpu())   # to align
        while not eval_monitor.is_finished():
            obs = torch.as_tensor(evaluator_env.ready_obs[0]).to(dtype=torch.float32)
            vae_loss_var['real_obs'].append(obs.cpu())
            inference_output = policy.forward({0: obs})
            if cfg.env.render:
                eval_monitor.update_video(evaluator_env.ready_imgs)
                eval_monitor.update_output(inference_output)
            output = [v for v in inference_output.values()]
            action = [to_ndarray(v['action']) for v in output][0]  # TBD
            timesteps = evaluator_env.step({0: action})
            
            
            # world_model inference [next] obs
            world_action = torch.Tensor(action)
            world_normed_obs = norm_data(obs)
            
            # ignore_dim
            tmp = []
            for idx in range(world_normed_obs.shape[0]):
                if idx not in cfg.world_model.model.ignore_dim:
                    tmp.append(world_normed_obs[idx].item())
            world_normed_obs = torch.Tensor(tmp)
            # done
            
            world_next_obs, obs_box = world_model.step(world_normed_obs, world_action, instep=True)
            tmp = []
            for idx in range(world_next_obs.shape[0]):
                while len(tmp) in cfg.world_model.model.ignore_dim:
                    tmp.append(0)
                tmp.append(world_next_obs[idx].item())
            while len(tmp) in cfg.world_model.model.ignore_dim:
                tmp.append(0)
            world_next_obs = torch.Tensor(tmp)
            # done
            
            # print('debug\n\n\n')
            # obs_box = np.stack(obs_box)
            # for i in range(obs_box.shape[1]):
            #     print(f'dim{i}: max {np.max(obs_box[:, i])}, min {np.min(obs_box[:, i])}')
            # exit()
            
            # draw in_step diffusion process (normed)
            real_draw = torch.as_tensor(evaluator_env.ready_obs[0])
            real_draw = np.array(norm_data(real_draw))
            diffusion_draw = np.array(world_next_obs)
            
            randomset = random.randint(0, 50)
            if randomset == 27:  # the collapse step    and track_time >= 100
                obs_box = np.stack(obs_box)
                # ignored_dim
                for idx in cfg.world_model.model.ignore_dim:
                    obs_box = np.insert(obs_box, idx, 0, axis=1)
                
                obs_high = np.array([3.14, 5., 5., 5., 3.14, 5., 3.14, 5., 5., 3.14, 5., 3.14, 5., 5., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
                obs_low  = np.array([-3.14, -5., -5., -5., -3.14, -5., -3.14, -5., -0., -3.14, -5., -3.14, -5., -0., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])
                for idx in range(obs_box.shape[1]):
                    base = real_draw[idx].item()
                    baseline = np.ones(obs_box[:, idx].shape) * base
                    diff = diffusion_draw[idx].item()
                    diffline = np.ones(obs_box[:, idx].shape) * diff
                    plt.figure()
                    plt.plot(baseline)
                    plt.plot(obs_box[:, idx])
                    plt.plot(diffline)
                    plt.legend(['real', 'diffusion', 'diff_end'])
                    plt.title(f'obs {idx}: [{obs_low[idx]}, {obs_high[idx]}]')
                    plt.savefig(f'./{cfg.exp_name}/obs_indiffusion_step{track_time}_{idx}.png')
                    plt.close()
                instep_draw_time += 1
                if instep_draw_time == 3:
                    exit()
            
            
            world_next_obs = norm_data_restore(world_next_obs)  # (s, a, bg) -> s'
            vae_loss_var['diffusion_obs'].append(world_next_obs.cpu())
            
            for env_id, timestep in timesteps.items():
                if timestep.done:
                    reward = timestep.info['eval_episode_return']
                    eval_monitor.update_reward(env_id, reward)
                    if 'episode_info' in timestep.info:
                        eval_monitor.update_info(env_id, timestep.info)
            track_time += 1
        
        episode_return_real = eval_monitor.get_episode_return()
        
        print(f'\n\n\n ===================   real timestpe: {track_time}   =================== \n\n\n')
        
        # metric: traj & reward
        real_obs = vae_loss_var['real_obs']
        vae_obs = vae_loss_var['diffusion_obs']
        max_step = min(len(real_obs), len(vae_obs))
        # traj state loss, but in normed version (to align training)
        for i in range(max_step):
            normed_real = norm_data(real_obs[i])
            normed_vae = norm_data(vae_obs[i])
            val = torch.nn.functional.mse_loss(normed_real, normed_vae)
            tb_logger.add_scalar(f'both_env/traj_loss', val, i)
        for step, rew in enumerate(episode_return_real):
            tb_logger.add_scalar(f'real_env/reward', rew, step)
        
        # plot obs change
        real_obs = vae_loss_var['real_obs']
        real_obs = torch.stack(real_obs)
        real_obs = torch.Tensor(real_obs).cpu()
        torch.save(real_obs, f'./{cfg.exp_name}/data_real.pt')
        real_obs = np.array(real_obs)
        plt.plot(real_obs)
        plt.savefig(f'./{cfg.exp_name}/real_obs.png')
        
        plt.figure()
        diffusion_obs = vae_loss_var['diffusion_obs']
        diffusion_obs = torch.stack(diffusion_obs)
        diffusion_obs = torch.Tensor(diffusion_obs).cpu()
        torch.save(diffusion_obs, f'./{cfg.exp_name}/data_diffusion.pt')
        diffusion_obs = np.array(diffusion_obs)
        plt.plot(diffusion_obs)
        plt.savefig(f'./{cfg.exp_name}/diffusion_obs.png')
        break


    print('Done.')
    return world_model





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=10)
    parser.add_argument('--config', '-c', type=str, default='eval_traj_real_diffusion_config.py')
    args = parser.parse_args()
    config = Path(__file__).absolute().parent / 'config' / args.config
    config = read_config(str(config))
    config[0].exp_name = config[0].exp_name.replace('0', str(args.seed))
    serial_pipeline(config, seed=args.seed)
