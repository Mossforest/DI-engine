from copy import deepcopy
import gym
import wandb
from ditk import logging

from ding.config import compile_config
from ding.envs import DingEnvWrapper, BaseEnvManagerV2

from ding.model import DQN
from ding.policy import MDQNPolicy
from ding.data import DequeBuffer
from dizoo.atari.envs import AtariEnv

from ding.framework import task, ding_init
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, data_pusher, eps_greedy_handler, online_logger, wandb_online_logger, CkptSaver
from ding.utils import set_pkg_seed

from asterix_mdqn_config import main_config, create_config



def main():
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
        policy = MDQNPolicy(cfg.policy, model=model)
    
        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(eps_greedy_handler(cfg))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        # task.use(nstep_reward_enhancer(cfg))
        task.use(data_pusher(cfg, buffer))
        task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer))
        task.use(wandb_online_logger(project_name='asterix_adqn_exp', exp_name=cfg.exp_name))
        task.use(online_logger(train_show_freq=10000))
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=10000000))
        task.run()


if __name__ == '__main__':
    main()