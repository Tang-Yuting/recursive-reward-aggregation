import jax
import jax.numpy as jnp

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