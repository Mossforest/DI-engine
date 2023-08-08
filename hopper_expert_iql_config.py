# You can conduct Experiments on D4RL with this config file through the following command:
# cd ../entry && python d4rl_cql_main.py
from easydict import EasyDict

main_config = dict(
    exp_name="hopper_expert_iql_seed0-obsnorm-fix",
    env=dict(
        env_id='hopper-expert-v2',
        collector_env_num=1,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=11,
            action_shape=3,
            actor_head_hidden_size=256,
            actor_head_layer_num=2,
            critic_head_hidden_size=256,
            critic_head_layer_num=2,
        ),
        learn=dict(
            data_path=None,
            train_epoch=30000,
            batch_size=256,
            learning_rate_q=3e-4,
            learning_rate_policy=3e-4,
            learning_rate_value=3e-4,
            expectile=0.7,  # The actual tau for expectiles.
            beta=3.0,
            clip_score=100,
        ),
        collect=dict(data_type='d4rl', ),
        eval=dict(evaluator=dict(eval_freq=500, )),
    ),
    seed=0,
)

main_config = EasyDict(main_config)
main_config = main_config

create_config = dict(
    env=dict(
        type='d4rl',
        import_names=['dizoo.d4rl.envs.d4rl_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='iql',
        import_names=['ding.policy.iql'],
    ),
)
create_config = EasyDict(create_config)
create_config = create_config
