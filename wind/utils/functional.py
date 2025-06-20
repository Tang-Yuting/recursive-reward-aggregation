"""
Functional utilities for working with JAX lax.scan, including:

- forward: run a function f(c) repeatedly for fixed steps
- scanr: right-to-left cumulative scan over a list
- duplicate: helper for (x, x) tuple generation in scan

"""

import jax

# jax.lax.scan :: (c -> a -> (c, b)) -> c -> [a] -> (c, [b])
def duplicate(x):
    return x, x


# forward :: (c -> (c, b)) -> c -> (c, [b])
def forward(f, init, length):
    return jax.lax.scan(f=lambda c, _: f(c), init=init, xs=None, length=length)


# scanr :: (a -> b -> b) -> b -> [a] -> [b]
def scanr(f, init, xs):
    return jax.lax.scan(f=lambda b, a: duplicate(f(a, b)), init=init, xs=xs, reverse=True)[1]
