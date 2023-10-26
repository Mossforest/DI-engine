# You can conduct Experiments on D4RL with this config file through the following command:
# cd ../entry && python d4rl_cql_main.py
from easydict import EasyDict

main_config = dict(
    exp_name="bipedalwalker_diffusion_worldmodel_seed0",
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
            data_path='/mnt/nfs/chenxinyan/DI-engine/bipedalwalker_data/friction_known_train.hdf5',
        ),
    ),
    world_model=dict(
        cuda=True,
        n_timesteps=1000,
        beta_schedule='linear',
        clip_denoised=False,
        model=dict(
            state_size=24,
            action_size=4,
            background_size=3,
            hidden_size=512,
        ),
        learn=dict(
            data_path=None,
            train_epoch=30000,
            batch_size=256,
            learning_rate=3e-4,
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
