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
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ding.envs import get_vec_env_setting, create_env_manager
# from ding.worker import BaseLearner, InteractionSerialEvaluator
from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.world_model import create_world_model
from ding.utils import set_pkg_seed
from ding.torch_utils import Adam, to_ndarray
from ding.utils.data import create_dataset
from ding.world_model.model.obs_decoder import HiddenDecoder
from ding.framework.middleware.functional.evaluator import VectorEvalMonitor


bipedalwalker_sac_config = dict(
    exp_name="exp_v1.12_hidden_diffuse_sync_debug",
    seed=369,
    cuda=True,
    env=dict(
        env_id='BipedalWalker-v3',
        evaluator_env_num=1,
        n_evaluator_episode=1,
        # (bool) Scale output action into legal range.
        act_scale=True,
        rew_clip=True,
        hardcore=False,
        render=True,
    ),
    policy=dict(
        cuda=True,
        # random_collect_size=10000,
        model=dict(
            obs_shape=24,
            action_shape=4,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
        ),
        eval=dict(
            n_sample=64,
            test_epoch=10,
            state_dict_path='./bipedalwalker_hardcore_pretrain/ckpt/ckpt_best.pth.tar',
        ),
    ),
    world_model=dict(
        cuda=True,
        n_timesteps=100,
        beta_schedule='cosine',
        clip_denoised=True,
        model=dict(
            state_size=128,
            action_size=4,
            background_size=3,
            hidden_size=512,
            layer_num=5,
        ),
        test=dict(
            data_path=None,
            test_epoch=100,
            batch_size=10000,
            state_dict_path='./exp_v1.12_hidden_diffuse_nonorm/model/epoch960',
        ),
    ),
    obs_decoder=dict(
        state_dict_path='./exp_sac_head_train_240202_100345/model/epoch980',
        discrete=[8,13],
    ),
    eval_step=50,
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
    world_model=dict(
        type='diffusion',
        import_names=['ding.world_model.diffusion'],
    ),
    replay_buffer=dict(type='naive', ),
)
bipedalwalker_sac_create_config = EasyDict(bipedalwalker_sac_create_config)
create_config = bipedalwalker_sac_create_config


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

def norm_data_restore_discrete(obs, env='bipedalwalker'):
    if env == 'bipedalwalker':
        obs_high = torch.Tensor([3.14, 5., 5., 5., 3.14, 5., 3.14, 5., 1., 3.14, 5., 3.14, 5., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]).to(obs.device)
        obs_low  = torch.Tensor([-3.14, -5., -5., -5., -3.14, -5., -3.14, -5., -0., -3.14, -5., -3.14, -5., -0., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.]).to(obs.device)
        obs = ((obs_high - obs_low) * obs + obs_high + obs_low) / 2
        for i, val in enumerate(obs):
            obs[i] = min(max(obs_low[i], val), obs_high[i])
        return obs
    else:
        assert False, f"no support norm for {env}!"


if __name__ == "__main__":
    cfg, create_cfg = main_config, create_config
    cfg = compile_config(cfg, seed=3, auto=True, create_cfg=create_cfg)
    
    print(f'============== exp name: {cfg.exp_name}')
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    
    # part 1. =========== agent & real env ===========
    
    # Env_1
    env_fn, _, evaluator_env_cfg = get_vec_env_setting(deepcopy(cfg.env), collect=False, eval_=True)
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    
    # agent
    policy = create_policy(cfg.policy, enable_field=['eval'])
    pre_policy = torch.load(cfg.policy.eval.state_dict_path, map_location=policy.device)
    device = policy.device
    policy.eval_mode.load_state_dict(pre_policy)
    policy = policy.eval_mode
    
    # world_model
    world_model = create_world_model(cfg.world_model, env_fn(cfg.env), tb_logger)
    world_model.load_model(cfg.world_model.test.state_dict_path)
    
    # decoder
    obs_decoder = HiddenDecoder(discrete=cfg.obs_decoder.discrete)
    pre_decoder = torch.load(cfg.obs_decoder.state_dict_path, map_location=device)
    obs_decoder.load_state_dict(pre_decoder)
    if torch.cuda.is_available() and cfg.cuda:
        obs_decoder = obs_decoder.to('cuda')
    
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

        # obs = torch.as_tensor(evaluator_env.ready_obs[0]).to(dtype=torch.float32)
        # vae_loss_var['diffusion_obs'].append(obs.cpu())
        while not eval_monitor.is_finished():
            obs = torch.as_tensor(evaluator_env.ready_obs[0]).to(dtype=torch.float32)
            vae_loss_var['real_obs'].append(obs.cpu())
            inference_output = policy.forward({0: obs})
            if cfg.env.render:
                eval_monitor.update_video(evaluator_env.ready_imgs)
                eval_monitor.update_output(inference_output)
            output = [v for v in inference_output.values()]
            # encoder get this obs
            embed_obs = output[0]['embed_obs']
            action = [to_ndarray(v['action']) for v in output][0]  # TBD
            timesteps = evaluator_env.step({0: action})
            
            # world_model inference [next] obs
            world_action = torch.Tensor(action)
            hidden_next_obs = world_model.step(embed_obs, world_action)
            
            # decoder next_obs from world_model
            with torch.no_grad():
                world_next_obs = obs_decoder.forward(hidden_next_obs)
            next_continuous = world_next_obs['continuous'].reshape(-1)
            world_next_obs['discrete'] = torch.stack(world_next_obs['discrete'])
            next_discrete = world_next_obs['discrete'].reshape(-1)
            for idx, val in zip(cfg.obs_decoder.discrete, next_discrete):
                val = val.unsqueeze(0)
                next_continuous = torch.cat((next_continuous[:idx], val, next_continuous[idx:]))
            world_next_obs = next_continuous
            
            world_next_obs = norm_data_restore(world_next_obs)
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
        # torch.save(real_obs, f'./{cfg.exp_name}/data_real.pt')
        real_obs = np.array(real_obs)
        plt.plot(real_obs)
        plt.savefig(f'./{cfg.exp_name}/real_obs.png')
        
        plt.figure()
        diffusion_obs = vae_loss_var['diffusion_obs']
        diffusion_obs = torch.stack(diffusion_obs)
        diffusion_obs = torch.Tensor(diffusion_obs).cpu()
        # torch.save(diffusion_obs, f'./{cfg.exp_name}/data_diffusion.pt')
        diffusion_obs = np.array(diffusion_obs)
        plt.plot(diffusion_obs)
        plt.savefig(f'./{cfg.exp_name}/diffusion_obs.png')
        
        # draw each dim
        real = real_obs
        diff = diffusion_obs

        real = np.array(real)
        diff = np.array(diff)

        obs_high = np.array([3.14, 5., 5., 5., 3.14, 5., 3.14, 5., 5., 3.14, 5., 3.14, 5., 5., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        obs_low  = np.array([-3.14, -5., -5., -5., -3.14, -5., -3.14, -5., -0., -3.14, -5., -3.14, -5., -0., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])

        assert real.shape[1] == diff.shape[1]
        for idx in range(real.shape[1]):
            plt.figure()
            plt.plot(real[:, idx])
            plt.plot(diff[:, idx])
            plt.legend(['real', 'diffusion'])
            plt.title(f'obs {idx}: [{obs_low[idx]}, {obs_high[idx]}]')
            plt.savefig(f'./{cfg.exp_name}/obs_{idx}.png')
            plt.close()
        
        break


    print('Done.')


