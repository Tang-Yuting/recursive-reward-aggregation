from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from scipy.special import logsumexp

from recursive_stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from recursive_stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from recursive_stable_baselines3.recursive_td3.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, TD3Policy


from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update

SelfTD3 = TypeVar("SelfTD3", bound="TD3")


class TD3(OffPolicyAlgorithm):
    """
    Twin Delayed DDPG (TD3)
    Addressing Function Approximation Error in Actor-Critic Methods.

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/abs/1802.09477
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`td3_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: TD3Policy
    actor: Actor
    actor_target: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        policy: Union[str, type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1e6,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        recursive_type: Union[th.device, str] = "dsum",
        output_number: int = 1,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
            recursive_type=recursive_type,
            output_number=output_number,
        )

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.actor_batch_norm_stats = get_parameters_by_name(self.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.actor_batch_norm_stats_target = get_parameters_by_name(self.actor_target, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def softplus(self, x):
        return th.nn.functional.softplus(th.tensor(x, dtype=th.float32))

    def update(self, rewards, tau, recursive_type: Any) -> th.Tensor:
        if recursive_type == "dsum":
            # print("rewards", rewards.shape, "tau", tau.shape)
            update_tau = rewards + self.gamma * tau
            return update_tau
        elif recursive_type == "dmax":
            update_tau = th.max(rewards, self.gamma * tau)
            return update_tau
        elif recursive_type == "min":
            update_tau = th.min(rewards, tau)
            return update_tau
        elif recursive_type == "log-sum-exp":
            update_tau = logsumexp([rewards, tau])
            return update_tau

        elif recursive_type == "dsum_dmax":
            update_tau = tau
            tau_dsum, tau_dmax = tau[:, 0], tau[:, 1]
            rewards = rewards.squeeze()
            update_tau[:, 0] = rewards + self.gamma * tau_dsum
            update_tau[:, 1] = th.maximum(rewards, self.gamma * tau_dmax)
            # print("sum", update_tau[:, 0], "max", update_tau[:, 1])
            return update_tau

        elif recursive_type == "max_min_weight":
            update_tau = tau
            tau_max, tau_min = tau[:, 0], tau[:, 1]
            rewards = rewards.squeeze()
            update_tau[:, 0] = th.maximum(rewards, tau_max)
            update_tau[:, 1] = th.min(rewards, tau_max)
            return update_tau

        elif recursive_type == "mean":
            update_tau = tau
            # print("update_tau", update_tau.shape)
            tau_sum, tau_length = tau[:,0], tau[:,1]
            tau_length = self.softplus(tau_length)
            # print("reward", rewards.shape, rewards)  # torch.Size([256,1])
            # print("tau_sum", tau_sum.shape, tau_sum)  # torch.Size([256])
            rewards = rewards.squeeze()
            update_tau[:, 0] = rewards + tau_sum
            update_tau[:, 1] = 1 + tau_length
            return update_tau

        elif recursive_type == "range_multi_output":
            update_tau = tau
            tau_max, tau_min = tau[:, 0], tau[:, 1]
            update_tau[:, 0] = max(rewards, tau_max)
            update_tau[:, 1] = min(rewards, tau_min)
            return update_tau

        elif recursive_type == "p-mean_multi_output":
            update_tau = tau
            tau_p_mean, tau_length = tau[:, 0], tau[:, 1]
            update_tau[:, 1] = 1 + tau_length
            update_tau[:, 0] = (((rewards**2) + tau_length * (tau_p_mean**2)) / update_tau[:, 1])**(1/2)
            return update_tau

        elif recursive_type == "dsum_variance":
            update_tau = tau
            tau_dsum, tau_mean, tau_variance, tau_length = tau[:, 0], tau[:, 1], tau[:, 2], tau[:, 3]
            tau_length = self.softplus(tau_length)
            rewards = rewards.squeeze()
            update_tau[:, 3] = 1 + tau_length
            update_tau[:, 0] = rewards + self.gamma * tau_dsum
            update_tau[:, 1] = tau_mean + ((rewards - tau_mean) / update_tau[:, 3])
            update_tau[:, 2] = tau_variance + (
                        (rewards - update_tau[:, 1]) * (rewards - tau_mean) - tau_variance) / update_tau[:, 3]
            return update_tau

        elif recursive_type == "sum_range_multi_output":
            update_tau = tau
            tau_sum, tau_max, tau_min = tau[:, 0], tau[:, 1], tau[:, 2]
            update_tau[:, 0] = rewards + self.gamma * tau_sum
            update_tau[:, 1] = max(rewards, tau_max)
            update_tau[:, 2] = min(rewards, tau_min)
            return update_tau

        elif recursive_type == "sharpe_3":
            update_tau = tau
            tau_mean, tau_variance, tau_length = tau[:, 0], tau[:, 1], tau[:, 2]
            tau_length = self.softplus(tau_length)
            update_tau[:, 2] = 1 + tau_length
            update_tau[:, 0] = tau_mean + ((rewards - tau_mean) / update_tau[:, 2])
            update_tau[:, 1] = tau_variance + ((rewards - update_tau[:, 0]) * (rewards - tau_mean) - tau_variance) / update_tau[:, 2]
            return update_tau

        elif recursive_type == "sharpe_3_multi_env":
            update_tau = tau
            tau_mean, tau_variance, tau_length = tau[:, 0], tau[:, 1], tau[:, 2]
            tau_length = self.softplus(tau_length)
            # print("tau_length", tau_length)
            update_tau[:, 2] = 1 + tau_length
            update_tau[:, 0] = tau_mean + ((rewards - tau_mean) / update_tau[:, 2])
            update_tau[:, 1] = tau_variance + ((rewards - update_tau[:, 0]) * (rewards - tau_mean) - tau_variance) / update_tau[:, 2]
            return update_tau


    def post(self, tau, recursive_type: Any) -> th.Tensor:
        if recursive_type == "dsum":
            # print("dsum_post")
            post_tau = tau
            return post_tau
        elif recursive_type == "dmax":
            post_tau = tau
            return post_tau
        elif recursive_type == "min":
            post_tau = tau
            return post_tau
        elif recursive_type == "log-sum-exp":
            post_tau = tau
            return post_tau
        elif recursive_type == "dsum_dmax":
            tau_dsum, tau_dmax = tau[..., 0], tau[..., 1]
            post_tau = tau_dsum + tau_dmax
            return post_tau

        elif recursive_type == "max_min_weight":
            tau_max, tau_min = tau[..., 0], tau[..., 1]
            post_tau = tau_max + tau_min
            return post_tau

        elif recursive_type == "mean":
            tau_sum, tau_length = tau[...,0], tau[...,1]
            post_tau = tau_sum / tau_length
            return post_tau

        elif recursive_type == "range_multi_output":
            tau_max, tau_min = tau[...,0], tau[...,1]
            post_tau = tau_max - tau_min
            return post_tau

        elif recursive_type == "p-mean_multi_output":
            post_tau = tau[...,0]
            return post_tau

        elif recursive_type == "dsum_variance":
            tau_dsum, tau_mean, tau_variance, tau_length = tau[..., 0], tau[..., 1], tau[..., 2], tau[..., 3]
            tau_variance = self.softplus(tau_variance)
            post_tau = tau_dsum - tau_variance
            return post_tau

        elif recursive_type == "sum_range_multi_output":
            tau_sum, tau_max, tau_min = tau[..., 0], tau[..., 1], tau[..., 2]
            post_tau_sum = tau_sum
            post_tau_range = tau_max - tau_min
            post_tau = post_tau_sum - 0.2 * post_tau_range
            return post_tau

        elif recursive_type == "sharpe_3":
            tau_mean, tau_variance, tau_length = tau[..., 0], tau[..., 1], tau[..., 2]
            tau_variance = self.softplus(tau_variance)
            post_tau = tau_mean / np.sqrt(tau_variance + 1e-8)
            return post_tau

        elif recursive_type == "sharpe_3_multi_env":
            tau_mean, tau_variance, tau_length = tau[..., 0], tau[..., 1], tau[..., 2]
            tau_variance = self.softplus(tau_variance)
            post_tau = tau_mean / np.sqrt(tau_variance + 1e-8)
            return post_tau


    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        # print("output num", self.output_number)

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next tau-values: min over all critics targets
                tau_1_next, tau_2_next = self.critic_target(replay_data.next_observations, next_actions)  # (batch_size, 3)
                tau_1_next_update = self.post(self.update(replay_data.rewards, (1 - replay_data.dones) * tau_1_next, self.recursive_type), self.recursive_type)
                tau_2_next_update = self.post(self.update(replay_data.rewards, (1 - replay_data.dones) * tau_2_next, self.recursive_type), self.recursive_type)
                target_tau_values = th.min(tau_1_next_update, tau_2_next_update)

            # Get current Q-values estimates for each critic network
            current_tau_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(self.post(current_tau, self.recursive_type), target_tau_values) for current_tau in current_tau_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                tau_for_actor = self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations))
                actor_loss = -self.post(tau_for_actor, self.recursive_type).mean()

                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

    def learn(
        self: SelfTD3,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "TD3",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTD3:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []
