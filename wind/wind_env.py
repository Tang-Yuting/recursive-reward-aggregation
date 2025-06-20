import jax
import jax.numpy as jnp

import gymnax
from gymnax.environments import spaces
from gymnax.environments.environment import Environment, EnvParams

from typing import NamedTuple, Tuple, Optional
import functools as ft
from pathlib import Path
import chex

from structures import *

def wind_fn(x, y):
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
    d = jnp.sqrt(x**2 + y**2)
    # reward = -d
    # reward = 1 / d
    # reward = 1 / (d + 0.1)
    reward = 100 * jnp.exp(-d)
    # reward = 100 * jnp.exp(-(d**2))
    return reward


class XYWindEnv(Environment):
    parameter_names = ["phi"]
    jittable_render = True

    def __init__(self, action_scale, space_size):
        super().__init__()
        self.max_action = 1.0
        self.action_scale = action_scale
        self.space_size = space_size

    def get_obs(self, state: EnvState) -> chex.Array:
        return jnp.array(state.pos)

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        return spaces.Box(low=-self.max_action, high=self.max_action, shape=(2,), dtype=jnp.float32)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        high = jnp.array([self.space_size], dtype=jnp.float32)
        return spaces.Box(-high, high, shape=(2,), dtype=jnp.float32)

    def state_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        return spaces.Dict(
            {
                "pos": spaces.Box(-jnp.finfo(jnp.float32).max, jnp.finfo(jnp.float32).max, (2), jnp.float32),
                "last_pos": spaces.Box(-jnp.finfo(jnp.float32).max, jnp.finfo(jnp.float32).max, (2), jnp.float32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: float, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        action = jnp.clip(action, -self.max_action, self.max_action)
        action = self.action_scale * action

        reward = reward_fn(*state.pos)
        wind = jnp.stack(wind_fn(*state.pos))

        new_pos = state.pos + action + wind
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
        # pos = jax.random.uniform(key, shape=(2,), minval=-self.space_size, maxval=self.space_size)
        pos = jnp.array([0.8, 0.0])
        state = EnvState(pos=pos, last_pos=pos, time=0)

        return jax.lax.stop_gradient(self.get_obs(state)), jax.lax.stop_gradient(state)

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        done = state.time >= params.max_steps_in_episode
        return done

def init_jax_runner_state(args, key):
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


