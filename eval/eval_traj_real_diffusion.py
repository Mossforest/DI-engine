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
    
    # Env_2
    env_fn, _, evaluator_env_cfg = get_vec_env_setting(deepcopy(cfg.env), collect=False, eval_=True)
    vae_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    vae_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    
    print('start eval...')
    for _ in range(cfg.policy.eval.test_epoch):
        track_time = 0
        # 1. real world env
        if evaluator_env.closed:
            evaluator_env.launch()
        else:
            evaluator_env.reset()
        eval_monitor = VectorEvalMonitor(evaluator_env.env_num, cfg.env.n_evaluator_episode)
        vae_loss_var = {'real_obs': [], 'diffusion_obs': []}

        obs = torch.as_tensor(evaluator_env.ready_obs[0]).to(dtype=torch.float32)
        vae_loss_var['diffusion_obs'].append(obs.cpu())
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
            world_next_obs = world_model.step(world_normed_obs, world_action)
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
        # episode_return_min = np.min(episode_return)
        # episode_return_max = np.max(episode_return)
        # episode_return_std = np.std(episode_return)
        # episode_return = np.mean(episode_return)
        
        # print('Evaluation: Eval Iter({})\tEval Reward({:.3f})'.format(epoch, episode_return))
        # tb_logger.add_scalar(f'real_env/eval_value', episode_return, epoch)
        # tb_logger.add_scalar(f'real_env/eval_value_min', episode_return_min, epoch)
        # tb_logger.add_scalar(f'real_env/eval_value_max', episode_return_max, epoch)
        # tb_logger.add_scalar(f'real_env/eval_value_std', episode_return_std, epoch)
        if cfg.env.render:
            real_replay_video = eval_monitor.get_episode_video()  # [N, T, C, H, W]
            real_replay_video = real_replay_video.squeeze().transpose(0, 2, 3, 1)  # [T, H, W, C]
            
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            size = (real_replay_video.shape[1], real_replay_video.shape[2])
            vw = cv2.VideoWriter(f'./{cfg.exp_name}/video_real.avi', fourcc=fourcc, fps=25, frameSize=size)
            
            index=0
            for img in real_replay_video:
                f_img = Image.fromarray(img)
                f_out = np.array(f_img, dtype=np.uint8)
                assert (f_out.shape[0], f_out.shape[1]) == size
                cv2.imwrite(f'./{cfg.exp_name}/real_{index}.jpg', f_out)
                index+=1
                vw.write(f_out)
            vw.release()
        
        print(f'\n\n\n ===================   real timestpe: {track_time}   =================== \n\n\n')
        
        
        # # 2. vae obs env
        # run_time = 0
        # if vae_env.closed:
        #     vae_env.launch()
        # else:
        #     vae_env.reset()
        # obs = torch.as_tensor(vae_env.ready_obs[0]).to(dtype=torch.float32)
        # while True:
        #     vae_loss_var['diffusion_obs'].append(obs.cpu())
        #     inference_output = policy.forward({0: obs})
        #     output = [v for v in inference_output.values()]
        #     action = [to_ndarray(v['action']) for v in output][0]  # TBD
        #     action = torch.Tensor(action)
        #     normed_obs = norm_data(obs)
        #     next_obs = world_model.step(normed_obs, action)
        #     obs = norm_data_restore(next_obs)  # (s, a, bg) -> s'
        #     run_time += 1
        #     if run_time % 5 == 0:
        #         print(f'\n\n\n ===================   diffusion timestpe: {run_time}   =================== \n\n\n')
        #     if run_time >= track_time:
        #         break
        
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
