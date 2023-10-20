# You can conduct Experiments on D4RL with this config file through the following command:
# cd ../entry && python d4rl_cql_main.py
from easydict import EasyDict

main_config = dict(
    exp_name="bipedalwalker_diffusion_worldmodel",
    env=dict(
        env_name='BipedalWalker-v3',
        collector_env_num=8,
        evaluator_env_num=5,
        act_scale=True,
        n_evaluator_episode=5,
        rew_clip=True,  # reward clip
        replay_path=None,
    ),
    world_model=dict(
        cuda=True,
        model=dict(
            state_size=24,
            action_size=4,
            background_size=3,
        ),
        learn=dict(
            data_path=None,
            train_epoch=30000,
            batch_size=256,
            learning_rate_q=3e-4,
            learning_rate_policy=1e-4,
            learning_rate_alpha=1e-4,
            alpha=0.2,
            auto_alpha=False,
            lagrange_thresh=-1.0,
            min_q_weight=5.0,
        ),
    ),
)

main_config = EasyDict(main_config)
main_config = main_config

create_config = dict(
    env=dict(
        type='d4rl',
        import_names=['dizoo.d4rl.envs.d4rl_env'],
    ),
    env_manager=dict(type='base'),
    world_model=dict(
        type='diffusion',
        import_names=['ding.world_model.diffusion'],
    ),
)
create_config = EasyDict(create_config)
create_config = create_config
