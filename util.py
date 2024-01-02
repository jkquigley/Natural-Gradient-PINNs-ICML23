import matplotlib.pylab as plt
import jax.numpy as jnp
from jax import vmap


def save(name, n, u_star, v_model, params):
    x = jnp.linspace(0, 1, n)
    y = jnp.linspace(0, 1, n)
    XX, YY = jnp.meshgrid(x, y)
    XX = XX.flatten()
    YY = YY.flatten()

    XY = jnp.column_stack((XX, YY))

    Z = v_model(params, XY)
    z = Z.reshape((n,n))
    u = vmap(u_star)(XY).reshape((n,n))
    error = jnp.abs(u - z)

    jnp.save("data/" + name +  "/u-star.npy", u)
    jnp.save("data/" + name + "/u-theta.npy", z)
    jnp.save("data/" + name + "/error.npy", z)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12,10))

    extent = [0,1,0,1]

    im_gt = axs[0].imshow(u, vmin=0, vmax=1, cmap='hot', extent=extent)
    axs[0].set_title(r"$u\left(x,y\right)$")

    im_pred = axs[1].imshow(z, vmin=0, vmax=1, cmap='hot', extent=extent)
    axs[1].set_title(r"$u_{\theta}\left(x,y\right)$")

    im_err = axs[2].imshow(error, cmap='hot', extent=extent)
    axs[2].set_title(r"$|u\left(x,y\right) - u_{\theta}\left(x,y\right)|$")

    plt.colorbar(im_pred, ax=axs[:-1], shrink=0.25, pad=0.1)
    plt.colorbar(im_err, ax=axs[-1], shrink=0.25, pad=0.2)

    plt.savefig("data/" + name + "/plots.pdf")
