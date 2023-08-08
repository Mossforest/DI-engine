from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from torch.optim.lr_scheduler import CosineAnnealingLR

from ding.torch_utils import Adam, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample, \
    qrdqn_nstep_td_data, qrdqn_nstep_td_error, get_nstep_return_data
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('iql')
class IQLPolicy(Policy):
    """
       Overview:
           Policy class of IQL algorithm (arXiv 2110.06169).

       Config:
           == ====================  ========    =============  ================================= =======================
           ID Symbol                Type        Default Value  Description                       Other(Shape)
           == ====================  ========    =============  ================================= =======================
           1  ``type``              str         td3            | RL policy register name, refer  | this arg is optional,
                                                               | to registry ``POLICY_REGISTRY`` | a placeholder
           2  ``cuda``              bool        True           | Whether to use cuda for network |
           3  | ``random_``         int         10000          | Number of randomly collected    | Default to 10000 for
              | ``collect_size``                               | training samples in replay      | SAC, 25000 for DDPG/
              |                                                | buffer when training starts.    | TD3.
           4  | ``model.policy_``   int         256            | Linear layer size for policy    |
              | ``embedding_size``                             | network.                        |
           5  | ``model.soft_q_``   int         256            | Linear layer size for soft q    |
              | ``embedding_size``                             | network.                        |
           6  | ``model.value_``    int         256            | Linear layer size for value     | Defalut to None when
              | ``embedding_size``                             | network.                        | model.value_network
              |                                                |                                 | is False.
           7  | ``learn.learning``  float       3e-4           | Learning rate for soft q        | Defalut to 1e-3, when
              | ``_rate_q``                                    | network.                        | model.value_network
              |                                                |                                 | is True.
           8  | ``learn.learning``  float       3e-4           | Learning rate for policy        | Defalut to 1e-3, when
              | ``_rate_policy``                               | network.                        | model.value_network
              |                                                |                                 | is True.
           9  | ``learn.learning``  float       3e-4           | Learning rate for policy        | Defalut to None when
              | ``_rate_value``                                | network.                        | model.value_network
              |                                                |                                 | is False.
           10 | ``learn.alpha``     float       0.2            | Entropy regularization          | alpha is initiali-
              |                                                | coefficient.                    | zation for auto
              |                                                |                                 | `alpha`, when
              |                                                |                                 | auto_alpha is True
           11 | ``learn.repara_``   bool        True           | Determine whether to use        |
              | ``meterization``                               | reparameterization trick.       |
           12 | ``learn.``          bool        False          | Determine whether to use        | beta parameter
              | ``auto_alpha``                                 | auto beta parameter      | determines the
              |                                                | `alpha`.                        | relative importance
              |                                                |                                 | of the entropy term
              |                                                |                                 | against the reward.
           13 | ``learn.-``         bool        False          | Determine whether to ignore     | Use ignore_done only
              | ``ignore_done``                                | done flag.                      | in halfcheetah env.
           14 | ``learn.-``         float       0.005          | Used for soft update of the     | aka. Interpolation
              | ``target_theta``                               | target network.                 | factor in polyak aver
              |                                                |                                 | aging for target
              |                                                |                                 | networks.
           == ====================  ========    =============  ================================= =======================
       """

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='sac',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool type) on_policy: Determine whether on-policy or off-policy.
        # on-policy setting influences the behaviour of buffer.
        # Default False in SAC.
        on_policy=False,
        multi_agent=False,
        # (bool type) priority: Determine whether to use priority in buffer sample.
        # Default False in SAC.
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (int) Number of training samples(randomly collected) in replay buffer when training starts.
        # Default 10000 in SAC.
        random_collect_size=10000,
        model=dict(
            # (bool type) twin_critic: Determine whether to use double-soft-q-net for target q computation.
            # Please refer to TD3 about Clipped Double-Q Learning trick, which learns two Q-functions instead of one .
            # The IQL policy set twin critic to True.
            twin_critic=True,

            # (str type) action_space: Use reparameterization trick for continous action
            action_space='reparameterization',

            # (int) Hidden size for actor network head.
            actor_head_hidden_size=256,

            # (int) Hidden size for critic network head.
            critic_head_hidden_size=256,
        ),
        learn=dict(

            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=1,
            # (int) Minibatch size for gradient descent.
            batch_size=256,

            # (float type) learning_rate_q: Learning rate for soft q network.
            # Default to 1e-3, when model.value_network is True.
            learning_rate_q=1e-3,
            # (float type) learning_rate_policy: Learning rate for policy network.
            # Default to 1e-3, when model.value_network is True.
            learning_rate_policy=1e-3,
            # (int type) 
            max_steps=int(2.5e6),
            # (float type) learning_rate_value: Learning rate for value network.
            # `learning_rate_value` should be initialized, when model.value_network is True.
            # Please set to 3e-4, when model.value_network is True.
            learning_rate_value=3e-4,

            # (float type) target_theta: Used for soft update of the target network,
            # aka. Interpolation factor in polyak averaging for target networks.
            # Default to 0.005.
            target_theta=0.005,
            # (float) discount factor for the discounted sum of rewards, aka. gamma.
            discount_factor=0.99,
            
            # IQL parameters
            expectile=0.7,  # The actual tau for expectiles.
            beta=3.0,
            clip_score=100,

            # (bool) Whether ignore done(usually for max step termination env. e.g. pendulum)
            # Note: Gym wraps the MuJoCo envs by default with TimeLimit environment wrappers.
            # These limit HalfCheetah, and several other MuJoCo envs, to max length of 1000.
            # However, interaction with HalfCheetah always gets done with done is False,
            # Since we inplace done==True with done==False to keep
            # TD-error accurate computation(``gamma * (1 - done) * next_v + reward``),
            # when the episode step is greater than max episode step.
            ignore_done=False,
            # (float) Weight uniform initialization range in the last output layer
            init_w=3e-3,
            # (int) The numbers of action sample each at every state s from a uniform-at-random
            num_actions=10,
        ),
        collect=dict(),
        eval=dict(),
        other=dict(),
    )
    
    def default_model(self) -> Tuple[str, List[str]]:
            return 'iql', ['ding.model.template.iql']

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init q, value and policy's optimizers, algorithm config, main and target models.
        """
        # Init
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self._num_actions = self._cfg.learn.num_actions

        # Weight Init
        init_w = self._cfg.learn.init_w
        # actor policy
        self._model.actor[-1].mu.weight.data.uniform_(-init_w, init_w)
        self._model.actor[-1].mu.bias.data.uniform_(-init_w, init_w)
        # self._model.actor[-1].log_sigma_layer.weight.data.uniform_(-init_w, init_w)
        # self._model.actor[-1].log_sigma_layer.bias.data.uniform_(-init_w, init_w)
        # critic
        self._model.critic[0][-1].last.weight.data.uniform_(-init_w, init_w)
        self._model.critic[0][-1].last.bias.data.uniform_(-init_w, init_w)
        self._model.critic[1][-1].last.weight.data.uniform_(-init_w, init_w)
        self._model.critic[1][-1].last.bias.data.uniform_(-init_w, init_w)
        # value critic
        self._model.value_critic[-1].last.weight.data.uniform_(-init_w, init_w)
        self._model.value_critic[-1].last.bias.data.uniform_(-init_w, init_w)

        # Optimizers
        self._optimizer_value = Adam(
            self._model.value_critic.parameters(),
            lr=self._cfg.learn.learning_rate_value,
        )
        self._optimizer_q1 = Adam(
            self._model.critic[0].parameters(),
            lr=self._cfg.learn.learning_rate_q,
        )
        self._optimizer_q2 = Adam(
            self._model.critic[1].parameters(),
            lr=self._cfg.learn.learning_rate_q,
        )
        self._optimizer_policy = Adam(
            self._model.actor.parameters(),
            lr=self._cfg.learn.learning_rate_policy,
        )
        self._scheduler_policy = CosineAnnealingLR(self._optimizer_policy, self._cfg.learn.max_steps, eta_min=0, last_epoch=- 1, verbose=False)

        # Algorithm config
        self._gamma = self._cfg.learn.discount_factor
        self._expectile = self._cfg.learn.expectile
        self._beta = self._cfg.learn.beta
        self._clip_score = self._cfg.learn.clip_score

        # Main and target models
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.learn.target_theta}
        )
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()
        self._target_model.reset()

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including current lr and loss.
        """
        loss_dict = {}
        data = default_preprocess_learn(
            data,
            use_priority=self._priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=False
        )
        if len(data.get('action').shape) == 1:
            data['action'] = data['action'].reshape(-1, 1)

        if self._cuda:
            data = to_device(data, self._device)

        self._learn_model.train()
        self._target_model.train()
        action = data['action']
        obs = data['obs']
        next_obs = data['next_obs']
        reward = data['reward']
        done = data['done']
        
        # normalize observation
        mean = torch.tensor(self._cfg.dataset_mean).to(self._device)
        std = torch.tensor(self._cfg.dataset_std).to(self._device)
        obs = (obs - mean) / std
        next_obs = (next_obs - mean) / std
        
        # print(f'====== obs normalized ======')
        # print(f'obs: {torch.mean(obs)}, {torch.std(obs)}')
        
        
        # # print(f'====== reward normalized ======')
        # print(f'before reward: {torch.min(reward)}, {torch.max(reward)}')
        
        # # normalize reward
        # reward_bounds = self._cfg.dataset_reward
        # reward = (reward - reward_bounds[0]) / (reward_bounds[1] - reward_bounds[0]) * 1000
        
        # # print(f'====== reward normalized ======')
        # print(f'bounds: {reward_bounds}')
        # print(f'reward: {torch.min(reward)}, {torch.max(reward)}')

        value_forward = lambda input: self._learn_model.forward(input, mode='compute_value_critic')['v_value']
        q_forward = lambda input: self._learn_model.forward(input, mode='compute_critic')['q_value']
        policy_forward = lambda input: self._learn_model.forward(input, mode='compute_actor')['logit']
        target_forward = lambda input: self._target_model.forward(input, mode='compute_critic')['q_value']

        # 1. compute value loss
        q1, q2 = target_forward(data)
        target_q = torch.min(q1, q2).detach()
        value = value_forward(obs)
        value_err = value - target_q
        value_sign = (value_err > 0).float()
        value_weight = (1 - value_sign) * self._expectile + value_sign * (1 - self._expectile)
        value_loss = (value_weight * (value_err ** 2)).mean()
        loss_dict['value_loss'] = value_loss

        # 2. compute policy loss, aka AWR policy extraction update
        (mu, sigma) = policy_forward(obs)
        policy_dist = Independent(Normal(mu, sigma), 1)
        policy_log = policy_dist.log_prob(action)
        exp_adv = torch.exp((target_q - value) / self._beta)
        if self._clip_score:
            exp_adv = torch.clamp(exp_adv, max=self._clip_score)
        weights = exp_adv.detach()  # TODO: ? from torch version
        policy_loss = (-policy_log * weights).mean()
        loss_dict['policy_loss'] = policy_loss

        # 3. compute q loss
        q1, q2 = q_forward(data)
        next_value = value_forward(next_obs).detach()
        q_data1 = v_1step_td_data(q1, next_value, reward, done, data['weight'])
        q1_loss, td_error_per_sample1 = v_1step_td_error(q_data1, self._gamma)
        q_data2 = v_1step_td_data(q2, next_value, reward, done, data['weight'])
        q2_loss, td_error_per_sample2 = v_1step_td_error(q_data2, self._gamma)
        td_error_per_sample = (td_error_per_sample1 + td_error_per_sample2) / 2
        loss_dict['q1_loss'] = q1_loss
        loss_dict['q2_loss'] = q2_loss
        
        # 4. update network
        self._optimizer_q1.zero_grad()
        loss_dict['q1_loss'].backward()
        self._optimizer_q1.step()
        self._optimizer_q2.zero_grad()
        loss_dict['q2_loss'].backward()
        self._optimizer_q2.step()
        
        self._optimizer_value.zero_grad()
        loss_dict['value_loss'].backward()
        self._optimizer_value.step()

        self._optimizer_policy.zero_grad()
        loss_dict['policy_loss'].backward()
        self._optimizer_policy.step()
        lr = self._scheduler_policy.get_lr()[0]
        self._scheduler_policy.step()

        loss_dict['total_loss'] = sum(loss_dict.values())

        # =============
        # after update
        # =============
        # target update
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr_q': self._optimizer_q1.defaults['lr'],
            'cur_lr_p': lr,
            'cur_lr_v': self._optimizer_value.defaults['lr'],
            'priority': td_error_per_sample.abs().tolist(),
            'td_error': td_error_per_sample.detach().mean().item(),
            'q_value': torch.min(q1, q2).detach().mean().item(),
            'target_q_value': target_q.detach().mean().item(),
            'value': value.detach().mean().item(),
            'action_mu': mu.detach().mean().item(),
            'action_sigma': sigma.detach().mean().item(),
            **loss_dict
        }
        
    def _state_dict_learn(self) -> Dict[str, Any]:
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer_q1': self._optimizer_q1.state_dict(),
            'optimizer_q2': self._optimizer_q2.state_dict(),
            'optimizer_policy': self._optimizer_policy.state_dict(),
            'optimizer_value': self._optimizer_value.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer_q1.load_state_dict(state_dict['optimizer_q1'])
        self._optimizer_q2.load_state_dict(state_dict['optimizer_q2'])
        self._optimizer_policy.load_state_dict(state_dict['optimizer_policy'])
        self._optimizer_value.load_state_dict(state_dict['optimizer_value'])


    def _init_collect(self) -> None:
        # raise Exception("Offline policy hasn't collect mode!")
        pass

    def _forward_collect(self, data: dict) -> dict:
        # raise Exception("Offline policy hasn't collect mode!")
        pass

    def _process_transition(self, obs: Any, policy_output: dict, timestep: namedtuple) -> dict:
        # raise Exception("Offline policy hasn't collect mode!")
        pass

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        # raise Exception("Offline policy hasn't collect mode!")
        pass

    def _init_eval(self) -> None:
        self._eval_model = model_wrap(self._model, wrapper_name='base')
        self._eval_model.reset()

    def _forward_eval(self, data: dict) -> dict:
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        
        # normalize observation
        mean = torch.tensor(self._cfg.dataset_mean).to(self._device)
        std = torch.tensor(self._cfg.dataset_std).to(self._device)
        data = (data - mean) / std
        
        with torch.no_grad():
            (mu, sigma) = self._eval_model.forward(data, mode='compute_actor')['logit']
            action = torch.tanh(mu)  # deterministic_eval
            output = {'action': action}
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _monitor_vars_learn(self) -> List[str]:
        return super()._monitor_vars_learn() + [
            'policy_loss', 'q1_loss', 'q2_loss', 'value_loss', 'cur_lr_q', 'cur_lr_p', 'cur_lr_v', 'target_q_value',
            'value', 'q_value', 'td_error', 'priority'
        ]
