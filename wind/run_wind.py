from typing import NamedTuple, Tuple, Optional
import functools as ft
from pathlib import Path

from matplotlib import pyplot as plt


import imageio
from IPython.display import Image


import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

import distrax

import jax
import jax.numpy as jnp
import numpy as np
import chex

import optax

from wind_env import *
from utils.functional import *
from utils.recursive_function import *

jnp.set_printoptions(precision=2, suppress=True)

fig_path = Path('figures')
data_path = Path('wind')


import argparse

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




class Actor(nn.Module):
    @nn.compact
    def __call__(self, x):
        activation = nn.tanh
        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(args.DIM_ACTION, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        log_scale = self.param("log_scale", nn.initializers.constant(args.INIT_LOGSCALE), ())
        log_scale = jnp.clip(log_scale, args.MIN_LOGSCALE, 0.0)
        x = distrax.Normal(loc=x, scale=jnp.ones_like(x) * jnp.exp(log_scale))
        return x


def sample_action(d_action, key):
    key, action_key = jax.random.split(key)
    action = d_action.sample(seed=action_key)
    log_prob = d_action.log_prob(action)
    return action, log_prob, key


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        activation = nn.tanh
        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(DIM_STATISTIC, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        if last is not None:
            x = last(x)
        return x


actor, critic = Actor(), Critic()

def linear_schedule(count):
    frac = 1.0 - (count // (args.NUM_MINIBATCHES * args.NUM_UPDATE_EPOCHS)) / args.NUM_UPDATES
    return args.LR * frac


def init_agent_state(init_observation, key):
    key, init_key = jax.random.split(key)
    actor_params = actor.init(init_key, init_observation)
    critic_params = critic.init(init_key, init_observation)

    tx = optax.chain(
        optax.clip_by_global_norm(args.MAX_GRAD_NORM),
        optax.adam(learning_rate=linear_schedule, eps=1e-5),
    )
    actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=tx)
    critic_state = TrainState.create(apply_fn=critic.apply, params=critic_params, tx=tx)

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


def jax_train(runner_state, agent_state, trajectory_fn, key):
    (runner_state, agent_state, key), loss_info = forward(
        f=ft.partial(update_step, trajectory_fn),
        init=(runner_state, agent_state, key),
        length=args.NUM_UPDATES,
    )
    return runner_state, agent_state, loss_info, key


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

    runner_state, env, env_params, key = init_jax_runner_state(args, key)
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