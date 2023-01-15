from easydict import EasyDict

qbert_dqn_config = dict(
    exp_name='qbert_dqn_normal',
    env=dict(
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        stop_value=30000,
        env_id='QbertNoFrameskip-v4',
        frame_stack=4
    ),
    policy=dict(
        cuda=True,
        priority=False,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[128, 128, 512],
        ),
        nstep=3,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=10,
            batch_size=32,
            learning_rate=0.0001,
            target_update_freq=500,
            reward_scale_function='normal',
        ),
        collect=dict(n_sample=100, ),
        logger=dict(record_path='./video_qbert_dqn', gradient_logger=True, plot_logger=True, action_logger='q_value'),
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
qbert_dqn_config = EasyDict(qbert_dqn_config)
main_config = qbert_dqn_config
qbert_dqn_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn'),
)
qbert_dqn_create_config = EasyDict(qbert_dqn_create_config)
create_config = qbert_dqn_create_config

if __name__ == '__main__':
    # or you can enter ding -m serial -c qbert_dqn_config.py -s 0
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)