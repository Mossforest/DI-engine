from copy import deepcopy
import gym
import wandb
from ditk import logging

from ding.config import compile_config
from ding.envs import DingEnvWrapper, BaseEnvManagerV2

from ding.model import DQN
from ding.policy import AveragedDQNPolicy
from ding.data import DequeBuffer
from dizoo.atari.envs import AtariEnv

from ding.framework import task, ding_init
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, data_pusher, eps_greedy_handler, online_logger, wandb_online_logger, CkptSaver
from ding.utils import set_pkg_seed

from easydict import EasyDict
from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--prime', default=5, type=int)
    args = parser.parse_args()
    return args


def make_config(args):
    breakout_averaged_dqn_config = dict(
        exp_name=f'adqn_prime{args.prime}_fix_seed{args.seed}',
        seed=args.seed,
        env=dict(
            collector_env_num=8,
            evaluator_env_num=8,
            n_evaluator_episode=8,
            stop_value=20000,
            env_id='AsterixNoFrameskip-v0',
            #'ALE/SpaceInvaders-v5' is available. But special setting is needed after gym make.
            frame_stack=4,
        ),
        policy=dict(
            cuda=True,
            priority=False,
            model=dict(
                obs_shape=[4, 84, 84],
                action_shape=9,
                encoder_hidden_size_list=[128, 128, 512],
                dueling=False,
            ),
            num_of_prime=args.prime,
            nstep=1,
            discount_factor=0.99,
            learn=dict(
                train_iterations=40000000,
                update_per_collect=10,
                batch_size=32,
                learning_rate=0.0001*args.prime,
                target_update_freq=500,
                learner=dict(hook=dict(save_ckpt_after_iter=1000000, ))
            ),
            collect=dict(n_sample=100, ),
            eval=dict(evaluator=dict(eval_freq=4000, )),
            other=dict(
                eps=dict(
                    type='exp',
                    start=1.,
                    end=0.05,
                    decay=1000000,
                ),
                replay_buffer=dict(replay_buffer_size=400000, ),
            ),
        ),
    )
    breakout_averaged_dqn_config = EasyDict(breakout_averaged_dqn_config)
    main_config = breakout_averaged_dqn_config
    breakout_averaged_dqn_create_config = dict(
        env=dict(
            type='atari',
            import_names=['dizoo.atari.envs.atari_env'],
        ),
        env_manager=dict(type='subprocess'),        
        policy=dict(type='averaged_dqn'),
    )
    breakout_averaged_dqn_create_config = EasyDict(breakout_averaged_dqn_create_config)
    create_config = breakout_averaged_dqn_create_config
    return main_config, create_config



def main(main_config, create_config):
    filename = '{}/log.txt'.format(main_config.exp_name)
    logging.getLogger(with_files=[filename]).setLevel(logging.INFO)
    
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    ding_init(cfg)
    
    # pipeline
    with task.start(async_mode=False, ctx=OnlineRLContext()):
    
        # environment
        collector_cfg = deepcopy(cfg.env)
        collector_cfg.is_train = True
        evaluator_cfg = deepcopy(cfg.env)
        evaluator_cfg.is_train = False
        collector_env = BaseEnvManagerV2(
            env_fn=[lambda: AtariEnv(collector_cfg) for _ in range(cfg.env.collector_env_num)],
            cfg=cfg.env.manager
        )
        evaluator_env = BaseEnvManagerV2(
            env_fn=[lambda: AtariEnv(evaluator_cfg) for _ in range(cfg.env.collector_env_num)],
            cfg=cfg.env.manager
        )
        
        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
        
        # policy
        model = DQN(**cfg.policy.model)
        buffer = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
        policy = AveragedDQNPolicy(cfg.policy, model=model)
        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(eps_greedy_handler(cfg))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        # task.use(nstep_reward_enhancer(cfg))
        task.use(data_pusher(cfg, buffer))
        task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer))
        metric_list = ['cur_lr', 'total_loss', 'q_value', 'target_q_value', 'priority']
        task.use(wandb_online_logger(project_name='asterix_exp_1', exp_name=cfg.exp_name, metric_list=metric_list))
        task.use(online_logger(train_show_freq=10000))
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=1000000))
        # termination_checker
        task.run()


if __name__ == '__main__':
    args = make_args()
    main_config, create_config = make_config(args)
    main(main_config, create_config)