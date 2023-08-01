import gym
from ditk import logging
from ding.model import IQLNetwork
from ding.policy import IQLPolicy
from ding.envs import DingEnvWrapper, BaseEnvManagerV2
from ding.data import create_dataset
from ding.config import compile_config
from ding.framework import task, ding_init
from ding.framework.context import OfflineRLContext
from ding.framework.middleware import interaction_evaluator, trainer, CkptSaver, offline_data_fetcher, offline_logger, wandb_offline_logger
from ding.utils import set_pkg_seed
from dizoo.d4rl.envs import D4RLEnv
# from halfcheetah_medium_iql_config import main_config, create_config

from argparse import ArgumentParser

def make_args():
    parser = ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--tau', default=0.7, type=float)
    parser.add_argument('--config', '-c', type=str, default='halfcheetah_medium_iql_config.py')
    parser.add_argument('--name', '-n', type=str, default=None)
    args = parser.parse_args()
    return args


def main(main_config, create_config, args):
    # main_config.exp_name = main_config.exp_name.replace('0', str(args.seed))
    if args.name:
        main_config.exp_name = args.name
    main_config.seed = args.seed
    main_config.policy.learn.expectile = args.tau
    
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    ding_init(cfg)
    with task.start(async_mode=False, ctx=OfflineRLContext()):
        evaluator_env = BaseEnvManagerV2(
            env_fn=[lambda: D4RLEnv(cfg.env) for _ in range(cfg.env.evaluator_env_num)], cfg=cfg.env.manager
        )

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        dataset = create_dataset(cfg)
        model = IQLNetwork(**cfg.policy.model)
        policy = IQLPolicy(cfg.policy, model=model)

        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(offline_data_fetcher(cfg, dataset))
        task.use(trainer(cfg, policy.learn_mode))
        metric_list = ['policy_loss', 'q1_loss', 'q2_loss', 'value_loss', 'cur_lr_q', 'cur_lr_p', 'cur_lr_v', 'target_q_value',
            'value', 'q_value', 'td_error', 'priority', 'action_mu', 'action_sigma']
        task.use(wandb_offline_logger(project_name='iql', run_name=cfg.exp_name, metric_list=metric_list))
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=100))
        task.use(offline_logger())
        task.run()


if __name__ == "__main__":
    args = make_args()
    args.config = args.config.rstrip('.py')
    exec(f'from {args.config} import main_config, create_config')
    main(main_config, create_config, args)
