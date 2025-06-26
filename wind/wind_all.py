from typing import NamedTuple, Tuple, Optional
import functools as ft
from pathlib import Path
from matplotlib import pyplot as plt
import imageio
from IPython.display import Image

import argparse

# numerical computing
import numpy as np
import jax
import jax.numpy as jnp
import chex

# neural network
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax import struct

# optimization
import optax

# distribution
import distrax

# environment
import gymnax
from gymnax.environments import spaces
from gymnax.environments.environment import Environment, EnvParams


fig_path = Path('figures')
data_path = Path('wind')

jnp.set_printoptions(precision=2, suppress=True)


@struct.dataclass
# State for the environment at a given timestep
class EnvState:
    pos: float
    last_pos: float
    time: int

# State used during rollout execution
class RunnerState(NamedTuple):
    env_state: gymnax.EnvState
    observation: jnp.ndarray

# Complete agent state, including both policy (actor) and value (critic) networks
class AgentState(NamedTuple):
    actor_state: TrainState
    critic_state: TrainState

# One step of trajectory data collected during rollout
class Step(NamedTuple):
    observation: jnp.ndarray
    action: jnp.ndarray
    log_prob: jnp.ndarray
    done: jnp.ndarray
    reward: jnp.ndarray
    statistic: jnp.ndarray
    value: jnp.ndarray
    info: jnp.ndarray

# A full training batch
class Batch(NamedTuple):
    trajectory: jnp.ndarray
    advantage: jnp.ndarray

# Output of the loss function for logging
class LossInfo(NamedTuple):
    loss: jnp.ndarray
    value_loss: jnp.ndarray
    policy_loss: jnp.ndarray
    entropy: jnp.ndarray


""" #################################### Hyperparameters #################################### """

def parse_args():
    parser = argparse.ArgumentParser()

    # Environment config
    parser.add_argument("--ENV_NAME", type=str, default="XY_WIND")
    parser.add_argument("--NUM_ENVS", type=int, default=64)
    parser.add_argument("--NUM_STEPS", type=int, default=128)
    parser.add_argument("--DIM_ACTION", type=int, default=2)
    parser.add_argument("--ACTION_SCALE", type=float, default=0.5)
    parser.add_argument("--SPACE_SIZE", type=float, default=2.0)

    # Network config
    parser.add_argument("--INIT_LOGSCALE", type=float, default=-1.0)
    parser.add_argument("--MIN_LOGSCALE", type=float, default=-4.0)

    # Optimization config
    parser.add_argument("--LR", type=float, default=2.5e-4)
    parser.add_argument("--MAX_GRAD_NORM", type=float, default=0.5)
    parser.add_argument("--CLIP_EPS", type=float, default=0.2)
    parser.add_argument("--VF_COEF", type=float, default=0.5)
    parser.add_argument("--ENT_COEF", type=float, default=0.01)

    # Training config
    parser.add_argument("--TOTAL_TIMESTEPS", type=int, default=int(5e5))
    parser.add_argument("--NUM_UPDATE_EPOCHS", type=int, default=10)
    parser.add_argument("--NUM_MINIBATCHES", type=int, default=4)

    # Recursive method
    parser.add_argument("--RECURSIVE_TYPE", type=str, default="max")

    args = parser.parse_args()

    # Compute derived fields manually
    args.BATCH_SIZE = args.NUM_ENVS * args.NUM_STEPS
    args.NUM_UPDATES = args.TOTAL_TIMESTEPS // args.BATCH_SIZE

    return args


def duplicate(x):
    return x, x

# forward :: (c -> (c, b)) -> c -> (c, [b])
def forward(f, init, length):
    return jax.lax.scan(f=lambda c, _: f(c), init=init, xs=None, length=length)

# scanr :: (a -> b -> b) -> b -> [a] -> [b]
def scanr(f, init, xs):
    return jax.lax.scan(f=lambda b, a: duplicate(f(a, b)), init=init, xs=xs, reverse=True)[1]


""" #################################### Aggregate Function #################################### """

def squeeze(statistic):
    return jnp.squeeze(statistic, axis=-1)


def dsum(discount):
    last = None

    def update(reward, sum):
        return reward[..., None] + discount * sum

    init = jnp.array([0.0])
    post = squeeze

    return f"sum_{discount}" if discount != 1 else "sum", last, update, init, post


def dmax(discount):
    last = None

    def update(reward, max):
        return jnp.maximum(reward[..., None], discount * max)

    init = jnp.array([-jnp.inf])
    post = squeeze

    return f"max_{discount}" if discount != 1 else "max", last, update, init, post


def min():
    last = None

    def update(reward, statistic):
        return jnp.minimum(reward[..., None], statistic)

    init = jnp.array([jnp.inf])
    post = squeeze

    return "min", last, update, init, post


def max_min(a, b):
    last = None

    def update(reward, statistic):
        max, min = statistic[..., 0], statistic[..., 1]
        return jnp.stack([jnp.maximum(reward, max), jnp.minimum(reward, min)], axis=-1)

    init = jnp.array([-jnp.inf, jnp.inf])

    def post(statistic):
        max, min = statistic[..., 0], statistic[..., 1]
        return a * max + b * min

    return f"{a}max{b:+}min", last, update, init, post


def mean():
    def last(output):
        length = jax.nn.softplus(output[..., 0])
        sum = output[..., 1]
        return jnp.stack([length, sum], axis=-1)

    def update(reward, statistic):
        return statistic + jnp.stack([jnp.ones_like(reward), reward], axis=-1)

    init = jnp.array([0.0, 0.0])

    def post(statistic):
        length = statistic[..., 0]
        sum = statistic[..., 1]
        return sum / length

    return "mean", last, update, init, post


def mean_variance(a, b):
    def last(output):
        length = jax.nn.softplus(output[..., 0])
        sum = output[..., 1]
        sum2 = jax.nn.softplus(output[..., 2])
        return jnp.stack([length, sum, sum2], axis=-1)

    def update(reward, statistic):
        return statistic + jnp.stack([jnp.ones_like(reward), reward, reward**2], axis=-1)

    init = jnp.array([0.0, 0.0, 0.0])

    def post(statistic):
        length, sum, sum2 = statistic[..., 0], statistic[..., 1], statistic[..., 2]
        mean = sum / length
        var = (sum2 / length - (mean) ** 2).clip(min=1e-8)
        return a * mean + b * var

    return f"{a}mean{b:+}var", last, update, init, post


def mean_std(a, b):
    def last(output):
        length = jax.nn.softplus(output[..., 0])
        sum = output[..., 1]
        sum2 = jax.nn.softplus(output[..., 2])
        return jnp.stack([length, sum, sum2], axis=-1)

    def update(reward, statistic):
        return statistic + jnp.stack([jnp.ones_like(reward), reward, reward**2], axis=-1)

    init = jnp.array([0.0, 0.0, 0.0])

    def post(statistic):
        length, sum, sum2 = statistic[..., 0], statistic[..., 1], statistic[..., 2]
        mean = sum / length
        var = (sum2 / length - (mean) ** 2).clip(min=1e-8)
        std = jnp.sqrt(var)
        return a * mean + b * std

    return f"{a}mean{b:+}std", last, update, init, post


def sharpe_ratio():
    def last(output):
        length = jax.nn.softplus(output[..., 0])
        sum = output[..., 1]
        sum2 = jax.nn.softplus(output[..., 2])
        return jnp.stack([length, sum, sum2], axis=-1)

    def update(reward, statistic):
        return statistic + jnp.stack([jnp.ones_like(reward), reward, reward**2], axis=-1)

    init = jnp.array([0.0, 0.0, 0.0])

    def post(statistic):
        length, sum, sum2 = statistic[..., 0], statistic[..., 1], statistic[..., 2]
        mean = sum / length
        var = (sum2 / length - (mean) ** 2).clip(min=1e-8)
        std = jnp.sqrt(var)
        return mean / std

    return "sharpe", last, update, init, post


""" #################################### Get Recursive Type #################################### """

def get_recursive_method(name: str):
    if name == "max":
        return dmax(1.0)
    elif name == "mean":
        return mean()
    elif name == "mean_var":
        return mean_variance(1.0, -0.2)
    elif name == "min":
        return min()
    else:
        raise ValueError(f"Unknown recursive method: {name}")


""" #################################### Env #################################### """

def wind_fn(x, y):
    """
    Compute the wind vector (u, v) at position (x, y).

    The wind consists of two components:
    - A rotational field: produces circular wind around the origin
      (e.g., u = -y, v = +x).
    - A weak "bounce" or attractive force toward the origin, scaled
      inversely with squared distance.

    Args:
        x (float): x-coordinate in 2D space
        y (float): y-coordinate in 2D space

    Returns:
        (u, v): Tuple of wind vector components at position (x, y)
    """
    d = jnp.sqrt(x**2 + y**2) + 1e-6
    e_x = x / d
    e_y = y / d
    wind = 0.3
    bounce = 0.1
    u_wind = -y
    v_wind = +x
    u_bounce = e_x / d**2
    v_bounce = e_y / d**2
    u = wind * u_wind + bounce * u_bounce
    v = wind * v_wind + bounce * v_bounce
    return u, v

def reward_fn(x, y):
    """
    Compute the reward at position (x, y) based on distance to the origin.

    The reward is highest at the origin and decays exponentially with distance.
    This encourages the agent to move toward the center (0, 0).

    Args:
        x (float): x-coordinate in 2D space
        y (float): y-coordinate in 2D space

    Returns:
        float: scalar reward value
    """
    d = jnp.sqrt(x**2 + y**2)
    reward = 100 * jnp.exp(-d)
    return reward

def plot_wind_environment(wind_fn, reward_fn, space_size, fig_path, filename="wind_env.pdf"):
    """
    Visualize a 2D wind environment with streamlines and reward contours.

    This function generates a plot of the vector field defined by `wind_fn`,
    overlaid with the reward landscape defined by `reward_fn`. The wind is
    visualized as streamlines and the reward as a filled contour map.

    Args:
        wind_fn (callable): Function taking (x, y) and returning wind vectors (u, v)
        reward_fn (callable): Function taking (x, y) and returning scalar reward
        space_size (float): Half-width/height of the square grid, i.e., plot range is [-space_size, space_size]
        fig_path (Path or str): Directory where the figure will be saved
        filename (str): Name of the output PDF file (default: "wind_env.pdf")
    """
    w = space_size
    x, y = np.meshgrid(np.linspace(-w, w, 101), np.linspace(-w, w, 101))
    u, v = wind_fn(x, y)
    r = reward_fn(x, y)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-w, w)
    ax.set_ylim(-w, w)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    # Plot reward as a filled contour map
    contour = ax.contourf(x, y, r, levels=100, cmap="RdYlGn")

    # Plot wind field as streamlines
    ax.streamplot(x, y, u, v, density=1, color="k", linewidth=1)

    fig.colorbar(contour, ax=ax, label="Reward")
    fig.tight_layout()

    fig.savefig(fig_path / filename)
    plt.close(fig)


class XYWindEnv(Environment):
    # List of parameter names used for environment variants or sweep
    parameter_names = ["phi"]

    # Enables jittable rendering in some Gymnax environments
    jittable_render = True

    def __init__(self, action_scale, space_size):
        """
        Create a 2D XY environment with wind dynamics and bounded space.

        Args:
            action_scale (float): Scale multiplier for the agent's action
            space_size (float): Half-width of the environment (defines bounds as [-space_size, space_size])
        """
        super().__init__()
        self.max_action = 1.0             # Maximum magnitude for each action component (before scaling)
        self.action_scale = action_scale  # Scaling factor applied to agent action
        self.space_size = space_size      # Physical limit of the 2D space

    def get_obs(self, state: EnvState) -> chex.Array:
        """
        Extract the observation from the environment state.
        In this case, the position is the full observation.

        Args:
            state (EnvState): current environment state

        Returns:
            jnp.ndarray: agent's current position
        """
        return jnp.array(state.pos)

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """
        Define the 2D continuous action space: x and y directional forces.

        Returns:
            gym.Space: Box(-1, 1)^2
        """
        return spaces.Box(low=-self.max_action, high=self.max_action, shape=(2,), dtype=jnp.float32)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """
        Define the observation space: the 2D position bounded within the environment.

        Returns:
            gym.Space: Box([-space_size, -space_size], [space_size, space_size])
        """
        high = jnp.array([self.space_size], dtype=jnp.float32)
        return spaces.Box(-high, high, shape=(2,), dtype=jnp.float32)

    def state_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """
        Define the state space used internally by the environment.

        Returns:
            Dict Space containing:
                - pos: current position
                - last_pos: previous position
                - time: current timestep
        """
        return spaces.Dict(
            {
                "pos": spaces.Box(-jnp.finfo(jnp.float32).max, jnp.finfo(jnp.float32).max, (2,), jnp.float32),
                "last_pos": spaces.Box(-jnp.finfo(jnp.float32).max, jnp.finfo(jnp.float32).max, (2,), jnp.float32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: float, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """
        Advance the environment one step given an action.

        The action is scaled, combined with wind vector at the current position,
        and used to update the agent's position. The new position is clipped
        within bounds. A reward is computed based on distance to the origin.

        Returns:
            obs (jnp.ndarray): next observation (position)
            state (EnvState): updated environment state
            reward (float): reward based on new position
            done (bool): whether episode has ended
            info (dict): auxiliary info (empty here)
        """
        action = jnp.clip(action, -self.max_action, self.max_action)
        action = self.action_scale * action

        reward = reward_fn(*state.pos)         # Reward based on proximity to origin
        wind = jnp.stack(wind_fn(*state.pos))  # Wind vector at current position

        new_pos = state.pos + action + wind    # New position with wind and action
        new_pos = jnp.clip(new_pos, -self.space_size, self.space_size)

        state = EnvState(new_pos, state.pos, state.time + 1)
        done = self.is_terminal(state, params)

        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            jax.lax.stop_gradient(reward),
            jax.lax.stop_gradient(done),
            {},
        )

    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        """
        Reset the environment to the initial state.

        Returns:
            obs (jnp.ndarray): initial observation (position)
            state (EnvState): initial environment state
        """
        # pos = jax.random.uniform(...)  # Can be used for randomized reset
        pos = jnp.array([0.8, 0.0])  # Fixed start near top-right
        state = EnvState(pos=pos, last_pos=pos, time=0)

        return jax.lax.stop_gradient(self.get_obs(state)), jax.lax.stop_gradient(state)

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """
        Check whether the episode should terminate.

        Returns:
            bool: True if episode is over
        """
        return state.time >= params.max_steps_in_episode


def init_jax_runner_state(key):
    env, env_params = XYWindEnv(args.ACTION_SCALE, args.SPACE_SIZE), EnvParams(max_steps_in_episode=128)

    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, args.NUM_ENVS)
    observation, env_state = jax.vmap(
        env.reset,
        in_axes=(0, None),
    )(reset_keys, env_params)

    runner_state = RunnerState(
        env_state=env_state,
        observation=observation,
    )
    return runner_state, env, env_params, key


""" #################################### Policy #################################### """

# Actor network: maps observations to action distributions (for continuous control).
class Actor(nn.Module):
    @nn.compact
    def __call__(self, x):
        activation = nn.tanh
        # First hidden layer with orthogonal weight initialization
        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        # Second hidden layer
        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        # Output layer: maps to action mean (loc) for a Normal distribution
        x = nn.Dense(args.DIM_ACTION, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)

        # Learnable log standard deviation (shared across actions)
        log_scale = self.param("log_scale", nn.initializers.constant(args.INIT_LOGSCALE), ())
        log_scale = jnp.clip(log_scale, args.MIN_LOGSCALE, 0.0)

        # Construct normal distribution for action sampling
        x = distrax.Normal(loc=x, scale=jnp.ones_like(x) * jnp.exp(log_scale))
        return x


# Sample an action and compute log-probability for PPO / policy gradient methods
def sample_action(d_action, key):
    """
    Sample an action from the policy distribution and compute log-probability.

    Args:
        d_action (distrax.Distribution): action distribution from Actor
        key (jax.random.PRNGKey): random key

    Returns:
        action (jnp.ndarray): sampled action
        log_prob (jnp.ndarray): log-prob of sampled action
        key (jax.random.PRNGKey): updated key
    """
    key, action_key = jax.random.split(key)
    action = d_action.sample(seed=action_key)
    log_prob = d_action.log_prob(action)
    return action, log_prob, key


# Critic network: maps observations to value estimates (used in advantage computation)
class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        activation = nn.tanh
        # First hidden layer
        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        # Second hidden layer
        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        # Output layer: returns a statistic (value estimate) for training objective
        x = nn.Dense(DIM_STATISTIC, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        # Optionally apply a final transformation (e.g., aggregation or normalization)
        if last is not None:
            x = last(x)
        return x


# Instantiate the networks
actor, critic = Actor(), Critic()


def linear_schedule(count):
    """
    Linearly decaying learning rate schedule.

    Args:
        count (int): current training step, typically incremented per minibatch update

    Returns:
        float: scaled learning rate based on training progress
    """
    # Compute fraction of total updates completed
    frac = 1.0 - (count // (args.NUM_MINIBATCHES * args.NUM_UPDATE_EPOCHS)) / args.NUM_UPDATES
    return args.LR * frac


def init_agent_state(init_observation, key):
    """
    Initialize the actor-critic agent's parameters and optimizer states.

    Args:
        init_observation (jnp.ndarray): initial environment observation, used for shape inference
        key (jax.random.PRNGKey): PRNG key for network initialization

    Returns:
        agent_state (AgentState): container of TrainStates for actor and critic
        key (jax.random.PRNGKey): updated random key
    """
    # Split random key for actor/critic init
    key, init_key = jax.random.split(key)

    # Initialize network parameters with dummy input
    actor_params = actor.init(init_key, init_observation)
    critic_params = critic.init(init_key, init_observation)

    # Define optimizer with gradient clipping and linear learning rate schedule
    tx = optax.chain(
        optax.clip_by_global_norm(args.MAX_GRAD_NORM),  # Clip large gradients for stability
        optax.adam(learning_rate=linear_schedule, eps=1e-5),  # Adam optimizer with scheduling
    )

    # Wrap parameters and optimizer state into TrainState objects
    actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=tx)
    critic_state = TrainState.create(apply_fn=critic.apply, params=critic_params, tx=tx)

    # Combine into unified agent state
    agent_state = AgentState(actor_state, critic_state)
    return agent_state, key


def jax_step(env, env_params, agent_state: AgentState, runner_state_key: Tuple[RunnerState, chex.PRNGKey]):
    # unpack
    actor_state, critic_state = agent_state
    (env_state, observation), key = runner_state_key

    # actor-critic
    d_action = actor.apply(actor_state.params, observation)
    action, log_prob, key = sample_action(d_action, key)
    statistic = critic.apply(critic_state.params, observation)
    value = post(statistic)

    # step
    key, step_key = jax.random.split(key)
    step_keys = jax.random.split(step_key, args.NUM_ENVS)
    next_observation, env_state, reward, done, info = jax.vmap(
        env.step,
        in_axes=(0, 0, 0, None),
    )(step_keys, env_state, action, env_params)

    # pack
    runner_state = RunnerState(env_state, next_observation)
    step = Step(observation, action, log_prob, done, reward, statistic, value, info)
    return (runner_state, key), step


def jax_trajectory(env, env_params, runner_state: RunnerState, agent_state: AgentState, key):
    (runner_state, key), trajectory = forward(
        f=ft.partial(jax_step, env, env_params, agent_state),
        init=(runner_state, key),
        length=args.NUM_STEPS,
    )
    return runner_state, trajectory, key


def update_statistic_with_termination(reward_done, statistic):
    reward, done = reward_done
    return update(reward, jnp.where(done[:, None], init, statistic))


def get_advantage(init, trajectory):
    advantage = post(
        scanr(
            f=update_statistic_with_termination,
            init=init,
            xs=(trajectory.reward, trajectory.done),
        )
    ) - trajectory.value
    return advantage

def actor_loss_fn(log_prob_pred, log_prob, advantage):
    # ratio = jnp.exp(log_prob_pred - log_prob)
    ratio = jnp.exp(jnp.sum(log_prob_pred - log_prob, axis=-1))
    ratio_clipped = jnp.clip(ratio, 1.0 - args.CLIP_EPS, 1.0 + args.CLIP_EPS)
    advantages = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    actor_loss = -jnp.minimum(ratio, ratio_clipped) * advantages
    actor_loss = actor_loss.mean()
    return actor_loss


def value_loss_fn(value_pred, value, target):
    value_clipped = value + (value_pred - value).clip(-args.CLIP_EPS, args.CLIP_EPS)
    value_losses = jnp.square(value_pred - target)
    value_losses_clipped = jnp.square(value_clipped - target)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    return value_loss


def loss_fn(actor_params, critic_params, batch):
    trajectory, advantage = batch

    d_action = actor.apply(actor_params, trajectory.observation)
    log_prob = d_action.log_prob(trajectory.action)
    statistic = critic.apply(critic_params, trajectory.observation)
    value = post(statistic)

    actor_loss = actor_loss_fn(log_prob, trajectory.log_prob, advantage)
    value_loss = value_loss_fn(value, trajectory.value, advantage + trajectory.value)
    entropy = d_action.entropy().mean()
    loss = actor_loss + args.VF_COEF * value_loss - args.ENT_COEF * entropy
    return loss, LossInfo(loss, value_loss, actor_loss, entropy)


def prepare_minibatch(batch, key):
    # batch
    batch = jax.tree_util.tree_map(
        lambda x: x.reshape((args.BATCH_SIZE,) + x.shape[2:]),
        batch,
    )
    # shuffle
    key, permutation_key = jax.random.split(key)
    permutation = jax.random.permutation(permutation_key, args.BATCH_SIZE)
    shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
    # minibatch
    minibatches = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, [args.NUM_MINIBATCHES, -1] + list(x.shape[1:])),
        shuffled_batch,
    )
    return minibatches, key


def update_agent(agent_state, minibatch):
    actor_state, critic_state = agent_state

    # gradient
    grad_fn = jax.grad(loss_fn, argnums=[0, 1], has_aux=True)
    (actor_grads, critic_grads), loss_info = grad_fn(actor_state.params, critic_state.params, minibatch)
    # update
    actor_state = actor_state.apply_gradients(grads=actor_grads)
    critic_state = critic_state.apply_gradients(grads=critic_grads)

    return AgentState(actor_state, critic_state), loss_info


def update_step(trajectory_fn, update_state_key):
    runner_state, agent_state, key = update_state_key

    runner_state, trajectory, key = trajectory_fn(runner_state, agent_state, key)
    last_statistic = critic.apply(agent_state.critic_state.params, runner_state.observation)
    advantage = get_advantage(last_statistic, trajectory)
    batch = Batch(trajectory, advantage)
    minibatches, key = prepare_minibatch(batch, key)
    agent_state, loss_info = forward(
        f=lambda agent_state: jax.lax.scan(
            f=update_agent,
            init=agent_state,
            xs=minibatches,
        ),
        init=agent_state,
        length=args.NUM_UPDATE_EPOCHS,
    )
    return (runner_state, agent_state, key), loss_info

# Run the full training loop using JAX scan to unroll updates.
def jax_train(runner_state, agent_state, trajectory_fn, key):
    (runner_state, agent_state, key), loss_info = forward(
        f=ft.partial(update_step, trajectory_fn),
        init=(runner_state, agent_state, key),
        length=args.NUM_UPDATES,
    )
    return runner_state, agent_state, loss_info, key

# Render the current environment state as an RGB image with agent and target positions.
def render(space_size, state: EnvState) -> jax.Array:
    size = 256
    radius = 5
    rgb_array = jnp.full([size, size, 3], 64, dtype=jnp.uint8)

    def draw_circle(edit_array, pos, radius, color):
        color = jnp.array(color, dtype=jnp.uint8)
        color = jnp.tile(color, [256, 256, 1])
        pos = (pos + space_size) / (2 * space_size) * size
        y, x = jnp.mgrid[:size, :size]
        y_diff = y - pos[0]
        x_diff = x - pos[1]
        pixel_dists = jnp.sqrt(x_diff**2 + y_diff**2)
        pixel_dists = jnp.repeat(pixel_dists[:, :, None], 3, 2)
        return jnp.where(pixel_dists < radius, color, edit_array)

    rgb_array = draw_circle(rgb_array, state.last_pos, radius, [0, 125, 0])
    rgb_array = draw_circle(rgb_array, state.pos, radius, [0, 255, 0])
    rgb_array = draw_circle(rgb_array, jnp.array([0, 0]), radius, [0, 0, 255])

    return rgb_array

# Visualize agent behavior over one rollout episode and collect images and rewards.
def vis(actor_state, env_params):
    vis_env = XYWindEnv(args.ACTION_SCALE, args.SPACE_SIZE)
    key = jax.random.PRNGKey(42)
    key, reset_key = jax.random.split(key)
    observation, env_state = vis_env.reset(reset_key)
    rewards = []
    positions = []
    images = []
    for _ in range(101):
        key, act_key, step_key = jax.random.split(key, 3)
        action = actor.apply(actor_state.params, observation).sample(seed=act_key)
        observation, env_state, reward, done, info = vis_env.step(step_key, env_state, action, env_params)
        rewards.append(reward)
        positions.append(env_state.pos)
        images.append(render(args.SPACE_SIZE, env_state))
    return rewards, np.stack(positions), images


if __name__ == "__main__":
    args = parse_args()

    name, last, update, init, post = get_recursive_method(args.RECURSIVE_TYPE)
    DIM_STATISTIC = len(init)

    key = jax.random.PRNGKey(42)
    runner_state, env, env_params, key = init_jax_runner_state(key)
    agent_state, key = init_agent_state(runner_state.observation, key)
    trajectory_fn = ft.partial(jax_trajectory, env, env_params)
    runner_state, agent_state, loss_info, key = jax_train(runner_state, agent_state, trajectory_fn, key)

    rewards, positions, images = vis(agent_state.actor_state, env_params)

    data_path = Path("wind")
    fig_path = Path("figures") / name
    data_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)

    # save data
    jnp.save(data_path / f"rewards_{name}.npy", rewards)
    jnp.save(data_path / f"positions_{name}.npy", positions)

    # ================= traj =================
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    ax1.set_title(name)
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_aspect("equal")

    ax1.plot(*positions.T)
    ax1.scatter(*positions.T, alpha=np.linspace(0.1, 1.0, len(positions)))
    fig1.tight_layout()
    fig1.savefig(fig_path / f"trajectory_{name}.pdf")
    fig1.savefig(fig_path / f"trajectory_{name}.png", dpi=300)
    plt.close(fig1)

    # ================= reward =================
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.set_title(name)
    ax2.set_xlabel("step")
    ax2.set_ylabel("reward")
    ax2.set_ylim(0, 100)

    ax2.plot(rewards)
    fig2.tight_layout()
    fig2.savefig(fig_path / f"rewards_{name}.pdf")
    fig2.savefig(fig_path / f"rewards_{name}.png", dpi=300)
    plt.close(fig2)

    # ================= gif =================
    gif_path = fig_path / f"wind_{name}.gif"
    imageio.mimsave(gif_path, images, loop=0)

