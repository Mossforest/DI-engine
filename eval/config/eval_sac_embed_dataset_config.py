from easydict import EasyDict

bipedalwalker_sac_config = dict(
    exp_name='exp_demo_sac_embed_dataset',
    seed=123,
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
        collect=dict(
            data_type='hdf5',
            # offline data path
            train_data_path='./bipedalwalker_data/bipedalwalker_normal_smoother_collect/processed_train.hdf5',
            eval_data_path='./bipedalwalker_data/bipedalwalker_normal_smoother_collect/processed_eval.hdf5',
        ),
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
    replay_buffer=dict(type='naive', ),
)
bipedalwalker_sac_create_config = EasyDict(bipedalwalker_sac_create_config)
create_config = bipedalwalker_sac_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c bipedalwalker_sac_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0, max_env_step=int(1e5))
