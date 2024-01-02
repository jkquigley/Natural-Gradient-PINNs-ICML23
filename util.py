import jax.numpy as jnp
from jax import vmap


def save(eq, name, n, u_star, v_model, params):
    x = jnp.linspace(0, 1, n)
    y = jnp.linspace(0, 1, n)
    XX, YY = jnp.meshgrid(x, y)
    XX = XX.flatten()
    YY = YY.flatten()

    XY = jnp.column_stack((XX, YY))

    Z = v_model(params, XY)
    z = Z.reshape((n,n))
    u = vmap(u_star)(XY).reshape((n,n))

    jnp.save(f"data/{eq}/{name}/u-star.npy", u)
    jnp.save(f"data/{eq}/{name}/u-theta.npy", z)