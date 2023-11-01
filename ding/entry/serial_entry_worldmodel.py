from typing import Union, Optional, List, Any, Tuple
import os
import time
import random
import torch
from functools import partial
from tensorboardX import SummaryWriter
from copy import deepcopy
from torch.utils.data import DataLoader

from ding.envs import get_vec_env_setting, create_env_manager
# from ding.worker import BaseLearner, InteractionSerialEvaluator
from ding.config import read_config, compile_config
from ding.world_model import create_world_model
from ding.utils import set_pkg_seed
from ding.utils.data import create_dataset


def serial_pipeline_worldmodel(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0
) -> 'Policy':
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = deepcopy(input_cfg)
    # create_cfg.world_model.type = create_cfg.world_model.type + '_command'
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)

    dataset = create_dataset(cfg)
    sampler, shuffle = None, True
    dataloader = DataLoader(
        dataset,
        cfg.world_model.learn.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=lambda x: x,
        pin_memory=cfg.world_model.cuda,
    )
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    
    # Env
    try:
        if cfg.env.norm_obs.use_norm and cfg.env.norm_obs.offline_stats.use_offline_stats:
            cfg.env.norm_obs.offline_stats.update({'mean': dataset.mean, 'std': dataset.std})
    except (KeyError, AttributeError):
        pass
    env_fn, _, _ = get_vec_env_setting(cfg.env, collect=False, eval_=False)
    # Random seed
    set_pkg_seed(cfg.seed, use_cuda=cfg.world_model.cuda)
    
    # world_model
    world_model = create_world_model(cfg.world_model, env_fn(cfg.env), tb_logger)

    # evaluator = InteractionSerialEvaluator(
    #     cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    # )
    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    stop = False
    
    i = 0
    t = time.time()
    printed = False
    eval_data = None
    batch_num = None
    print('start training...')
    for epoch in range(cfg.world_model.learn.train_epoch):
        if printed:
            kk = random.randint(0, batch_num-1)
            i0 = i
        
        for train_data in dataloader:
            world_model.train(train_data, i, debug=(i%100==0))
            i += 1
            if printed and i - i0 == kk:
            # if i%100==0:
                eval_data = train_data.copy()
                # world_model.eval(eval_data, epoch)
        
        if printed:
            world_model.eval(eval_data, epoch)
        if not printed:
            print(f'---- One epoch has {i} batch, with time {time.time() - t} sec. ----')
            batch_num = i
            printed = True
        
        print(f'finished: epoch {epoch}')

        # # Evaluate policy at most once per epoch.
        # if evaluator.should_eval(learner.train_iter):
        #     stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter)

    # print('final reward is: {}'.format(reward))
    return world_model, stop
