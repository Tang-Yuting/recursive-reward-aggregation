import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces

from scipy.special import logsumexp

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.vec_env import VecNormalize

from recursive_stable_baselines3.recursive_common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
    RolloutBufferSamples_multi_output,
)
from recursive_stable_baselines3.recursive_common.utils import get_device

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Space
    obs_shape: tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, dict[str, np.ndarray]],
        env: Optional[VecNormalize] = None,
    ) -> Union[np.ndarray, dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )

            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    taus: np.ndarray
    count_step: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.count_step = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)  # for mean
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.taus = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super().reset()


    def compute_returns_and_advantage_recursive_mc(self, last_taus: th.Tensor, dones: np.ndarray, recursive_type: Any) -> None:
        last_taus = last_taus.clone().cpu().numpy().flatten()
        G_t = last_taus
        # print("last_taus", last_taus, "G_t", G_t)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                G_t = self.update(self.rewards[step], G_t * (1 - dones.astype(np.float32)), recursive_type)
                # print("type_1", G_t, dones.astype(np.float32))
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                G_t = self.update(self.rewards[step], G_t * next_non_terminal, recursive_type)
                # print("type_3", G_t, self.rewards[step])

            self.returns[step] = self.post(G_t, recursive_type)
            # print("self.returns[step]", step, self.returns[step])
            self.advantages[step] = self.post(G_t, recursive_type) - self.post(self.taus[step], recursive_type)  # Monte Carlo Advantage
            # print("self.advantages[step]", step, self.advantages[step])
        print("gae_mc", "returns", self.returns, "advantage", self.advantages)


    def compute_returns_and_advantage_recursive_gae(self, last_taus: th.Tensor, dones: np.ndarray, recursive_type: Any) -> None:
        last_taus = last_taus.clone().cpu().numpy().flatten()
        adv = np.zeros((self.advantages.shape[0], self.advantages.shape[0]), dtype=np.float32)
        taus_to_go = self.taus.copy().astype(np.float32)

        start_indices = np.where(self.episode_starts)[0]
        start_indices = np.append(start_indices, self.buffer_size)
        i = 0
        episode_end = start_indices[i]

        for step in range(self.buffer_size):
            if step >= episode_end:
                i = i + 1
                if i == len(start_indices):
                    break
            episode_end = start_indices[i]
            for n in range(0, episode_end - step):
                if n == self.buffer_size - 1 - step:
                    taus_to_go[n] = self.update(self.rewards[n], last_taus * dones.astype(np.float32), recursive_type)
                elif n == episode_end - 1 - step:
                    taus_to_go[n] = self.update(self.rewards[n], taus_to_go[n+1]*0, recursive_type)
                    # print("episode_end - step", n, episode_end - 1 - step)
                else:
                    taus_to_go[n] = self.update(self.rewards[n], taus_to_go[n+1], recursive_type)
                if i-1 >= 0:
                    adv[step][start_indices[i-1] + n] = self.post(taus_to_go[n], recursive_type) - self.post(self.taus[n], recursive_type)
                    adv[step][start_indices[i-1] + n] = adv[step][start_indices[i-1] + n] * (1 - self.gae_lambda) * (self.gae_lambda ** (step - start_indices[i-1]))
                    # print("self.gae_lambda ** (step - start_indices[i-1])", step - start_indices[i-1], self.gae_lambda ** (step - start_indices[i-1]))
                else:
                    adv[step][n] = self.post(taus_to_go[n], recursive_type) - self.post(self.taus[n], recursive_type)
                    adv[step][n] = adv[step][n] * (1 - self.gae_lambda) * (self.gae_lambda ** (step))

        self.advantages = adv.sum(axis=0).reshape(-1, 1).astype(np.float32)
        self.returns = self.advantages + self.post(self.taus, recursive_type).astype(np.float32)
        print("gae_gae", "returns", self.returns, "advantage", self.advantages)


    def compute_returns_and_advantage_recursive(self, last_taus: th.Tensor, dones: np.ndarray, recursive_type: Any) -> None:
        last_taus = last_taus.clone().cpu().numpy().flatten()

        if recursive_type == "dsum":
            print("dsum")
            last_gae_lam = 0
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones.astype(np.float32)
                    next_tau = last_taus
                else:
                    next_non_terminal = 1.0 - self.episode_starts[step + 1]
                    next_tau = self.taus[step + 1]

                delta = (self.post(self.update(self.rewards[step], next_tau * next_non_terminal, recursive_type), recursive_type)
                         - self.post(self.taus[step], recursive_type))
                # last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam   # gamma
                # last_gae_lam = delta + self.gae_lambda * next_non_terminal * last_gae_lam
                last_gae_lam = delta + next_non_terminal * last_gae_lam
                self.advantages[step] = last_gae_lam
            self.returns = self.advantages + self.taus
            print("gae_original", "returns", self.returns, "advantage", self.advantages)

        elif recursive_type == "dmax":
            print("dmax")
            last_gae_lam = 0
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones.astype(np.float32)
                    next_tau = last_taus
                else:
                    next_non_terminal = 1.0 - self.episode_starts[step + 1]
                    next_tau = self.taus[step + 1]

                delta = self.post(self.update(self.rewards[step], next_tau * next_non_terminal, recursive_type), recursive_type) - self.post(self.taus[step], recursive_type)
                # last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam   # gamma
                last_gae_lam = delta + self.gae_lambda * next_non_terminal * last_gae_lam
                self.advantages[step] = last_gae_lam
            self.returns = self.advantages + self.taus

        elif recursive_type == "min":
            print("min")
            last_gae_lam = 0
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones.astype(np.float32)
                    next_tau = last_taus
                else:
                    next_non_terminal = 1.0 - self.episode_starts[step + 1]
                    next_tau = self.taus[step + 1]

                delta = self.post(self.update(self.rewards[step], next_tau * next_non_terminal, recursive_type), recursive_type) - self.post(self.taus[step], recursive_type)
                # last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam   # gamma
                last_gae_lam = delta + self.gae_lambda * next_non_terminal * last_gae_lam
                self.advantages[step] = last_gae_lam
            self.returns = self.advantages + self.taus

        elif recursive_type == "log-sum-exp":
            print("log-sum-exp")
            last_gae_lam = 0
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones.astype(np.float32)
                    next_tau = last_taus
                else:
                    next_non_terminal = 1.0 - self.episode_starts[step + 1]
                    next_tau = self.taus[step + 1]

                delta = self.post(self.update(self.rewards[step], next_tau * next_non_terminal, recursive_type),
                                  recursive_type) - self.post(self.taus[step], recursive_type)
                # last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam   # gamma
                last_gae_lam = delta + self.gae_lambda * next_non_terminal * last_gae_lam
                self.advantages[step] = last_gae_lam
            self.returns = self.advantages + self.taus

        elif recursive_type == "mean":
            print("mean")
            last_gae_lam = 0
            count_step = 10
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones.astype(np.float32)
                    next_tau = last_taus
                    count_step += 1
                    self.count_step[step] = count_step
                else:
                    next_non_terminal = 1.0 - self.episode_starts[step + 1]
                    next_tau = self.taus[step + 1]
                    if next_non_terminal:
                        count_step += 1
                        self.count_step[step] = count_step
                    else:
                        count_step = 1
                        self.count_step[step] = count_step

                delta = self.post(self.update(self.rewards[step], next_tau * next_non_terminal, recursive_type, count_step),
                                  recursive_type, count_step) - self.post(self.taus[step], recursive_type, count_step)
                # last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam   # gamma
                last_gae_lam = delta + self.gae_lambda * next_non_terminal * last_gae_lam
                self.advantages[step] = last_gae_lam
            self.returns = self.advantages + self.taus



    def update(self, rewards, tau, recursive_type: Any, count_step: int = 1) -> th.Tensor:
        if recursive_type == "dsum":
            update_tau = rewards + self.gamma * tau
            return update_tau
        elif recursive_type == "dmax":
            update_tau = max(rewards, self.gamma * tau)
            return update_tau
        elif recursive_type == "min":
            update_tau = min(rewards, tau)
            return update_tau
        elif recursive_type == "log-sum-exp":
            update_tau = logsumexp([rewards, tau])
            return update_tau
        elif recursive_type == "mean":
            sum_reward = rewards + tau * (count_step - 1)
            update_tau = sum_reward / count_step
            return update_tau


    def post(self, tau, recursive_type: Any, count_step: int = 1):
        if recursive_type == "dsum":
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
        elif recursive_type == "mean":
            post_tau = tau
            return post_tau


    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        tau: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.taus[self.pos] = tau.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "taus",
                "log_probs",
                "advantages",
                "returns",
                "count_step",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.taus[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.count_step[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))


class RolloutBuffer_multi_output(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    taus: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        output_feature_num: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.output_feature_num = output_feature_num
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.taus = np.zeros((self.buffer_size, self.n_envs, self.output_feature_num), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super().reset()


    def compute_returns_and_advantage_recursive(self, last_taus: th.Tensor, dones: np.ndarray, recursive_type: Any) -> None:
        last_taus = last_taus.clone().cpu().numpy()

        if recursive_type == "mean_multi_output":
            print("mean_multi_output")
            last_gae_lam = 0
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones.astype(np.float32)
                    next_tau = last_taus
                else:
                    next_non_terminal = 1.0 - self.episode_starts[step + 1]
                    next_tau = self.taus[step + 1]

                delta = self.post(self.update(self.rewards[step], next_tau * next_non_terminal, recursive_type),
                                  recursive_type) - self.post(self.taus[step], recursive_type)
                # last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam   # gamma
                last_gae_lam = delta + self.gae_lambda * next_non_terminal * last_gae_lam
                self.advantages[step] = last_gae_lam
            # print("self.taus", self.taus.shape)
            self.returns = self.advantages + self.post(self.taus, recursive_type)

        elif recursive_type == "range_multi_output":
            print("range_multi_output")
            last_gae_lam = 0
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones.astype(np.float32)
                    next_tau = last_taus
                else:
                    next_non_terminal = 1.0 - self.episode_starts[step + 1]
                    next_tau = self.taus[step + 1]

                delta = self.post(self.update(self.rewards[step], next_tau * next_non_terminal, recursive_type),
                                  recursive_type) - self.post(self.taus[step], recursive_type)
                # last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam   # gamma
                last_gae_lam = delta + self.gae_lambda * next_non_terminal * last_gae_lam
                self.advantages[step] = last_gae_lam
            # print("self.taus", self.taus.shape)
            self.returns = self.advantages + self.post(self.taus, recursive_type)

        elif recursive_type == "p-mean_multi_output":  # p = 2
            print("p-mean_multi_output")
            last_gae_lam = 0
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones.astype(np.float32)
                    next_tau = last_taus
                else:
                    next_non_terminal = 1.0 - self.episode_starts[step + 1]
                    next_tau = self.taus[step + 1]

                delta = self.post(self.update(self.rewards[step], next_tau * next_non_terminal, recursive_type),
                                  recursive_type) - self.post(self.taus[step], recursive_type)
                # last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam   # gamma
                last_gae_lam = delta + self.gae_lambda * next_non_terminal * last_gae_lam
                self.advantages[step] = last_gae_lam
            # print("self.taus", self.taus.shape)
            self.returns = self.advantages + self.post(self.taus, recursive_type)

        elif recursive_type == "sum_variance_multi_output":
            print("dsum_variance_multi_output")
            last_gae_lam = 0
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones.astype(np.float32)
                    next_tau = last_taus
                else:
                    next_non_terminal = 1.0 - self.episode_starts[step + 1]
                    next_tau = self.taus[step + 1]
                # print(f"Shape of next_tau * next_non_terminal: {next_tau}, {next_non_terminal}")

                delta = self.post(self.update(self.rewards[step], next_tau * next_non_terminal, recursive_type),
                                  recursive_type) - self.post(self.taus[step], recursive_type)
                # last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam   # gamma
                last_gae_lam = delta + self.gae_lambda * next_non_terminal * last_gae_lam
                self.advantages[step] = last_gae_lam
            # print("self.taus", self.taus.shape)
            self.returns = self.advantages + self.post(self.taus, recursive_type)

        elif recursive_type == "sum_range_multi_output":
            print("dsum_range_multi_output")
            last_gae_lam = 0
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones.astype(np.float32)
                    next_tau = last_taus
                else:
                    next_non_terminal = 1.0 - self.episode_starts[step + 1]
                    next_tau = self.taus[step + 1]

                delta = self.post(self.update(self.rewards[step], next_tau * next_non_terminal, recursive_type),
                                  recursive_type) - self.post(self.taus[step], recursive_type)
                # last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam   # gamma
                last_gae_lam = delta + self.gae_lambda * next_non_terminal * last_gae_lam
                self.advantages[step] = last_gae_lam
            # print("self.taus", self.taus.shape)
            self.returns = self.advantages + self.post(self.taus, recursive_type)

        elif recursive_type == "sharpe_3":
            print("sharpe_3")
            last_gae_lam = 0
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones.astype(np.float32)
                    next_tau = last_taus
                else:
                    next_non_terminal = 1.0 - self.episode_starts[step + 1]
                    next_tau = self.taus[step + 1]

                # print(f"Shape of next_tau * next_non_terminal: {next_tau}, {next_non_terminal}")
                # next_non_terminal = next_non_terminal[:, None]
                # print(f"Shape of next_tau * next_non_terminal: {next_tau}, {next_non_terminal}")

                delta = self.post(self.update(self.rewards[step], next_tau * next_non_terminal, recursive_type),
                                  recursive_type) - self.post(self.taus[step], recursive_type)
                # print("delta", delta.shape, delta)
                # print("next_non_terminal", next_non_terminal.shape, next_non_terminal)
                # print("last_gae_lam", last_gae_lam.shape, last_gae_lam)
                # last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam   # gamma
                last_gae_lam = delta + self.gae_lambda * next_non_terminal * last_gae_lam
                self.advantages[step] = last_gae_lam
            # print("self.taus", self.taus.shape)
            self.returns = self.advantages + self.post(self.taus, recursive_type)

        elif recursive_type == "sharpe_3_multi_env":
            print("sharpe_3_multi_env")
            last_gae_lam = np.zeros(10)
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones.astype(np.float32)
                    next_tau = last_taus
                else:
                    next_non_terminal = 1.0 - self.episode_starts[step + 1]
                    next_tau = self.taus[step + 1]

                next_non_terminal_ = next_non_terminal[:, None]

                delta = self.post(self.update(self.rewards[step], next_tau * next_non_terminal_, recursive_type),
                                  recursive_type) - self.post(self.taus[step], recursive_type)
                # print("delta", delta.shape, delta)
                # print("next_non_terminal", next_non_terminal.shape, next_non_terminal)
                # print("last_gae_lam", last_gae_lam.shape, last_gae_lam)
                # last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam   # gamma
                last_gae_lam = delta + self.gae_lambda * next_non_terminal * last_gae_lam
                self.advantages[step] = last_gae_lam
            # print("self.taus", self.taus.shape)
            self.returns = self.advantages + self.post(self.taus, recursive_type)


    def compute_returns_and_advantage_recursive_mc(self, last_taus: th.Tensor, dones: np.ndarray, recursive_type: Any) -> None:
        print("compute_returns_and_advantage_recursive_mc, sharpe")
        last_taus = last_taus.clone().cpu().numpy()
        G_t = last_taus
        # print("last_taus", last_taus, "G_t", G_t)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1 - dones.astype(np.float32)
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
            next_non_terminal_ = next_non_terminal[:, None]

            # print("G_t_1", G_t)
            # G_t = self.update(self.rewards[step], G_t * next_non_terminal_, recursive_type)
            G_t = np.clip(self.update(self.rewards[step], G_t * next_non_terminal_, recursive_type), -1e6, 1e6)
            # if step >= self.buffer_size-100:
                # print("G_t_2", step, G_t.shape, G_t)
            # print("self.rewards[step]", self.rewards[step].shape, self.rewards[step])
            # print("next_non_terminal_", next_non_terminal_.shape, next_non_terminal_)

            self.returns[step] = self.post(G_t, recursive_type)
            self.advantages[step] = self.post(G_t, recursive_type) - self.post(self.taus[step], recursive_type)  # Monte Carlo Advantage
        # print("gae_mc", "returns", self.returns, "advantage", self.advantages)


    def update(self, rewards, tau, recursive_type: Any) -> th.Tensor:
        if recursive_type == "mean_multi_output":
            update_tau = tau
            tau_sum, tau_length = tau[:,0], tau[:,1]
            update_tau[:, 0] = rewards + tau_sum
            update_tau[:, 1] = 1 + tau_length
            return update_tau

        elif recursive_type == "range_multi_output":
            update_tau = tau
            tau_max, tau_min = tau[:, 0], tau[:, 1]
            update_tau[:, 0] = max(rewards, tau_max)
            update_tau[:, 1] = max(rewards, tau_min)
            return update_tau

        elif recursive_type == "p-mean_multi_output":
            update_tau = tau
            tau_p_mean, tau_length = tau[:, 0], tau[:, 1]
            update_tau[:, 1] = 1 + tau_length
            update_tau[:, 0] = (((rewards**2) + tau_length * (tau_p_mean**2)) / update_tau[:, 1])**(1/2)
            return update_tau

        elif recursive_type == "sum_variance_multi_output":
            update_tau = tau
            tau_sum, tau_sum_square, tau_length = tau[:, 0], tau[:, 1], tau[:, 2]
            update_tau[:, 0] = rewards + self.gamma * tau_sum
            update_tau[:, 1] = rewards**2 + tau_sum_square
            update_tau[:, 2] = 1 + tau_length
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
            tau_length = softplus(tau_length)
            update_tau[:, 2] = 1 + tau_length
            update_tau[:, 0] = tau_mean + ((rewards - tau_mean) / update_tau[:, 2])
            update_tau[:, 1] = tau_variance + ((rewards - update_tau[:, 0]) * (rewards - tau_mean) - tau_variance) / update_tau[:, 2]
            return update_tau

        elif recursive_type == "sharpe_3_multi_env":
            update_tau = tau
            tau_mean, tau_variance, tau_length = tau[:, 0], tau[:, 1], tau[:, 2]
            tau_length = softplus(tau_length)
            # print("tau_length", tau_length)
            update_tau[:, 2] = 1 + tau_length
            update_tau[:, 0] = tau_mean + ((rewards - tau_mean) / update_tau[:, 2])
            update_tau[:, 1] = tau_variance + ((rewards - update_tau[:, 0]) * (rewards - tau_mean) - tau_variance) / update_tau[:, 2]
            return update_tau


    def post(self, tau, recursive_type: Any):
        if recursive_type == "mean_multi_output":
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

        elif recursive_type == "sum_variance_multi_output":
            tau_sum, tau_sum_square, tau_length = tau[..., 0], tau[..., 1], tau[..., 2]
            post_tau_sum = tau_sum
            post_tau_variance = (tau_sum_square/tau_length) - (tau_sum/tau_length)**2
            post_tau = post_tau_sum - 0.2 * post_tau_variance
            return post_tau

        elif recursive_type == "sum_range_multi_output":
            tau_sum, tau_max, tau_min = tau[..., 0], tau[..., 1], tau[..., 2]
            post_tau_sum = tau_sum
            post_tau_range = tau_max - tau_min
            post_tau = post_tau_sum - 0.2 * post_tau_range
            return post_tau

        elif recursive_type == "sharpe_3":
            tau_mean, tau_variance, tau_length = tau[..., 0], tau[..., 1], tau[..., 2]
            tau_variance = softplus(tau_variance)
            post_tau = tau_mean / np.sqrt(tau_variance + 1e-8)
            return post_tau

        elif recursive_type == "sharpe_3_multi_env":
            tau_mean, tau_variance, tau_length = tau[..., 0], tau[..., 1], tau[..., 2]
            tau_variance = softplus(tau_variance)
            post_tau = tau_mean / np.sqrt(tau_variance + 1e-8)
            return post_tau


    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        tau: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.taus[self.pos] = tau.clone().cpu().numpy()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples_multi_output, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "taus",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples_multi_output:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.taus[batch_inds],
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples_multi_output(*tuple(map(self.to_torch, data)))



class DictReplayBuffer(ReplayBuffer):
    """
    Dict Replay buffer used in off-policy algorithms like SAC/TD3.
    Extends the ReplayBuffer to use dictionary observations

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observation_space: spaces.Dict
    obs_shape: dict[str, tuple[int, ...]]  # type: ignore[assignment]
    observations: dict[str, np.ndarray]  # type: ignore[assignment]
    next_observations: dict[str, np.ndarray]  # type: ignore[assignment]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        assert not optimize_memory_usage, "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = {
            key: np.zeros((self.buffer_size, self.n_envs, *_obs_shape), dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        self.next_observations = {
            key: np.zeros((self.buffer_size, self.n_envs, *_obs_shape), dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            obs_nbytes = 0
            for _, obs in self.observations.items():
                obs_nbytes += obs.nbytes

            total_memory_usage: float = obs_nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            if not optimize_memory_usage:
                next_obs_nbytes = 0
                for _, obs in self.observations.items():
                    next_obs_nbytes += obs.nbytes
                total_memory_usage += next_obs_nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(  # type: ignore[override]
        self,
        obs: dict[str, np.ndarray],
        next_obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = np.array(obs[key])

        for key in self.next_observations.keys():
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.next_observations[key][self.pos] = np.array(next_obs[key])

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(  # type: ignore[override]
        self,
        batch_size: int,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        return super(ReplayBuffer, self).sample(batch_size=batch_size, env=env)

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()}, env)
        next_obs_ = self._normalize_obs(
            {key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()}, env
        )

        assert isinstance(obs_, dict)
        assert isinstance(next_obs_, dict)
        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(
                -1, 1
            ),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
        )


class DictRolloutBuffer(RolloutBuffer):
    """
    Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
    Extends the RolloutBuffer to use dictionary observations

    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to Monte-Carlo advantage estimate when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Dict
    obs_shape: dict[str, tuple[int, ...]]  # type: ignore[assignment]
    observations: dict[str, np.ndarray]  # type: ignore[assignment]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super(RolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictRolloutBuffer must be used with Dict obs space only"

        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            self.observations[key] = np.zeros((self.buffer_size, self.n_envs, *obs_input_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def add(  # type: ignore[override]
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            obs_ = np.array(obs[key])
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = obs_

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(  # type: ignore[override]
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[DictRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = ["actions", "values", "log_probs", "advantages", "returns"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictRolloutBufferSamples:
        return DictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
        )

# def softplus(x):
#     return np.log(1 + np.exp(x))

def softplus(x):
    return th.nn.functional.softplus(th.tensor(x, dtype=th.float32)).numpy()