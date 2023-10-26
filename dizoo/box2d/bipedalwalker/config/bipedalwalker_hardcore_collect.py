from easydict import EasyDict

bipedalwalker_sac_config = dict(
    exp_name='bipedalwalker_hardcore_collect',
    seed=123,
    env=dict(
        env_id='BipedalWalker-v3',
        collector_env_num=8,
        # (bool) Scale output action into legal range.
        act_scale=True,
        rew_clip=True,
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
        collect=dict(
            n_sample=10000,
            save_path='./bipedalwalker_hardcore_collect/expert.pkl',
            data_type='hdf5',
            state_dict_path='./bipedalwalker_hardcore_pretrain/ckpt/ckpt_best.pth.tar',
        ),
    ),
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
    replay_buffer=dict(type='naive', ),
)
bipedalwalker_sac_create_config = EasyDict(bipedalwalker_sac_create_config)
create_config = bipedalwalker_sac_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c bipedalwalker_sac_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0, max_env_step=int(1e5))
