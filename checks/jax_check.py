import os
os.environ['JAX_PLATFORMS']= 'cpu'

import time
import jax
import jax.numpy as jnp




t1 = time.time()
x = jax.random.uniform(jax.random.PRNGKey(0), (10000, 10000), dtype=jnp.float32)
y = jax.random.uniform(jax.random.PRNGKey(0), (10000, 10000), dtype=jnp.float32)
z = jnp.dot(x, y)
print(f"CPU JAX Time: {time.time() - t1} seconds")

del os.environ['JAX_PLATFORMS']

t1 = time.time()
x = jax.random.uniform(jax.random.PRNGKey(0), (10000, 10000), dtype=jnp.float32)
y = jax.random.uniform(jax.random.PRNGKey(0), (10000, 10000), dtype=jnp.float32)
z = jnp.dot(x, y)
print(f"GPU JAX Time: {time.time() - t1} seconds")


