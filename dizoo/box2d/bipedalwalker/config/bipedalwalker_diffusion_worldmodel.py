# You can conduct Experiments on D4RL with this config file through the following command:
# cd ../entry && python d4rl_cql_main.py
from easydict import EasyDict

main_config = dict(
    exp_name="exp_v1.6.2",
    env=dict(
        env_name='BipedalWalker-v3',
        act_scale=True,
        rew_clip=True,
        replay_path=None,
        hardcore=True,
    ),
    policy=dict(
        cuda=True,
        collect=dict(
            data_type='hdf5',
            # offline data path
            train_data_path='./bipedalwalker_data/bipedalwalker_normal_smoother_collect/processed_train.hdf5',
            eval_data_path='./bipedalwalker_data/bipedalwalker_normal_smoother_collect/processed_eval.hdf5',
            ignore_dim=[8,13,22,23],
        ),
    ),
    world_model=dict(
        cuda=True,
        n_timesteps=1000,
        beta_schedule='cosine',
        clip_denoised=True,
        model=dict(
            state_size=20,
            action_size=4,
            background_size=3,
            hidden_size=512,
            layer_num=5,
        ),
        learn=dict(
            data_path=None,
            train_epoch=5000,
            batch_size=256,
            learning_rate=3e-4,
        ),
        test=dict(
            data_path=None,
            test_epoch=100,
            batch_size=10000,
        ),
    ),
)

main_config = EasyDict(main_config)
main_config = main_config

create_config = dict(
    env=dict(
        type='bipedalwalker',
        import_names=['dizoo.box2d.bipedalwalker.envs.bipedalwalker_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='cql',
        import_names=['ding.policy.cql'],
    ),
    world_model=dict(
        type='diffusion',
        import_names=['ding.world_model.diffusion'],
    ),
)
create_config = EasyDict(create_config)
create_config = create_config
