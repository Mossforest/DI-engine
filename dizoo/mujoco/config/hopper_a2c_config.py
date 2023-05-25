from easydict import EasyDict

hopper_a2c_config = dict(
    exp_name='hopper_a2c_seed0',
    env=dict(
        env_id='Hopper-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=6000,
    ),
    policy=dict(
        cuda=True,
        action_space='continuous',
        model=dict(
            obs_shape=11,
            action_shape=3,
            action_space='continuous',
        ),
        learn=dict(
            # (int) the number of data for a train iteration
            batch_size=256,
            learning_rate=0.0003,
            # (float) loss weight of the value network, the weight of policy network is set to 1
            value_weight=0.5,
            # (float) loss weight of the entropy regularization, the weight of policy network is set to 1
            entropy_weight=0.001,
            # (float) discount factor for future reward, defaults int [0, 1]
            discount_factor=0.999,
            adv_norm=True,
        ),
        collect=dict(
            n_sample=256,
            collector=dict(collect_print_freq=100, ),
        ),
        command=dict(),
        eval=dict(evaluator=dict(eval_freq=100, )),
    ),
)

hopper_a2c_config = EasyDict(hopper_a2c_config)
main_config = hopper_a2c_config

hopper_a2c_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='a2c',
        import_names=['ding.policy.a2c'],
    ),
    replay_buffer=dict(type='naive', ),
)
hopper_a2c_create_config = EasyDict(hopper_a2c_create_config)
create_config = hopper_a2c_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c hopper_a2c_config.py -s 0`
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy([main_config, create_config], seed=0)