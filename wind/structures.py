from typing import NamedTuple
from flax.training.train_state import TrainState
from flax import struct
import jax.numpy as jnp
import gymnax


@struct.dataclass
class EnvState:
    """Custom environment state used to track simulation status."""
    pos: float
    last_pos: float
    time: int


class RunnerState(NamedTuple):
    """Stores the environment state and agent observation at each time step."""
    env_state: gymnax.EnvState
    observation: jnp.ndarray


class AgentState(NamedTuple):
    """Holds both actor and critic training states (parameters + optimizer)."""
    actor_state: TrainState
    critic_state: TrainState


class Step(NamedTuple):
    """
    Stores information for one time step of agent-environment interaction.
    """
    observation: jnp.ndarray
    action: jnp.ndarray
    log_prob: jnp.ndarray
    done: jnp.ndarray
    reward: jnp.ndarray
    statistic: jnp.ndarray
    value: jnp.ndarray
    info: jnp.ndarray


class Batch(NamedTuple):
    """
    A batch of trajectory and corresponding advantage estimates used for training.
    """
    trajectory: jnp.ndarray
    advantage: jnp.ndarray


class LossInfo(NamedTuple):
    """
    Stores components of the total loss for logging and optimization.
    """
    loss: jnp.ndarray
    value_loss: jnp.ndarray
    policy_loss: jnp.ndarray
    entropy: jnp.ndarray
