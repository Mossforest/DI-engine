# You can conduct Experiments on D4RL with this config file through the following command:
# cd ../entry && python d4rl_cql_main.py
from easydict import EasyDict

main_config = dict(
    exp_name="exp_v1.12_hidden_diffuse_nonorm",
    env=dict(
        env_name='BipedalWalker-v3',
        act_scale=True,
        rew_clip=True,
        replay_path=None,
        hardcore=True,
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
        collect=dict(
            data_type='hdf5',
            # offline data path
            train_data_path='./bipedalwalker_data/bipedalwalker_normal_smoother_collect/processed_train.hdf5',
            eval_data_path='./bipedalwalker_data/bipedalwalker_normal_smoother_collect/processed_eval.hdf5',
            ignore_dim=None,
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
        learn=dict(
            data_path=None,
            train_epoch=5000,
            batch_size=256,
            learning_rate=1e-3,
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
    policy=dict(type='sac',),
    world_model=dict(
        type='diffusion',
        import_names=['ding.world_model.diffusion'],
    ),
)
create_config = EasyDict(create_config)
create_config = create_config
