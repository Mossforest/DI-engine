from easydict import EasyDict

bipedalwalker_sac_config = dict(
    exp_name='exp_eval_traj_real_diffusion',
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
    ),
    world_model=dict(
        cuda=True,
        n_timesteps=1000,
        beta_schedule='cosine',
        clip_denoised=True,
        model=dict(
            state_size=24,
            action_size=4,
            background_size=3,
            hidden_size=1024,
            layer_num=9,
        ),
        test=dict(
            data_path=None,
            test_epoch=100,
            batch_size=10000,
            state_dict_path='./bipedalwalker_diffusion_worldmodel_seed10_231208_124505/model/epoch300',
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
    world_model=dict(
        type='diffusion',
        import_names=['ding.world_model.diffusion'],
    ),
    replay_buffer=dict(type='naive', ),
)
bipedalwalker_sac_create_config = EasyDict(bipedalwalker_sac_create_config)
create_config = bipedalwalker_sac_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c bipedalwalker_sac_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0, max_env_step=int(1e5))
