"""
Adam Optimization.
Two dimensional Poisson equation example. Solution given by

u(x,y) = sin(pi*x) * sin(py*y).

"""
import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
import optax

from ngrad.models import init_params, mlp
from ngrad.domains import Square, SquareBoundary
from ngrad.integrators import DeterministicIntegrator
from ngrad.utility import laplace

jax.config.update("jax_enable_x64", True)

# random seed
seed = 0

# domains
interior = Square(1.)
boundary = SquareBoundary(1.)

# integrators
interior_integrator = DeterministicIntegrator(interior, 30)
boundary_integrator = DeterministicIntegrator(boundary, 30)
eval_integrator = DeterministicIntegrator(interior, 200)

# model
activation = lambda x : jnp.tanh(x)
layer_sizes = [2, 32, 1]
params = init_params(layer_sizes, random.PRNGKey(seed))
model = mlp(activation)
v_model = vmap(model, (None, 0))

# solution
@jit
def u_star(x):
    return jnp.product(jnp.sin(jnp.pi * x))

v_u_star = vmap(u_star, (0))
v_grad_u_star = vmap(
    lambda x: jnp.dot(grad(u_star)(x), grad(u_star)(x))**0.5, (0)
    )

# rhs
@jit
def f(x):
    return 2. * jnp.pi**2 * u_star(x)

# compute residual
laplace_model = lambda params: laplace(lambda x: model(params, x))
residual = lambda params, x: (laplace_model(params)(x) + f(x))**2.
v_residual =  jit(vmap(residual, (None, 0)))

# loss
@jit
def interior_loss(params):
    return interior_integrator(lambda x: v_residual(params, x))

@jit
def boundary_loss(params):
    return boundary_integrator(lambda x: v_model(params, x)**2)

@jit
def loss(params):
    return interior_loss(params) + boundary_loss(params)

# errors
error = lambda x: model(params, x) - u_star(x)
v_error = vmap(error, (0))
v_error_abs_grad = vmap(
        lambda x: jnp.dot(grad(error)(x), grad(error)(x))**0.5
        )

def l2_norm(f, integrator):
    return integrator(lambda x: (f(x))**2)**0.5


norm_sol_l2 = l2_norm(v_u_star, eval_integrator)
norm_sol_h1 = norm_sol_l2 + l2_norm(v_grad_u_star, eval_integrator)

interior_optimizer = optax.adam(learning_rate=0.001)
interior_opt_state = interior_optimizer.init(params)

boundary_optimizer = optax.adam(learning_rate=0.001)
boundary_opt_state = boundary_optimizer.init(params)
   
# adam gradient descent with line search
iterations = 200000
save_freq = 100

import numpy as np
data = np.empty((iterations // save_freq + 1, 5))

for iteration in range(iterations + 1):
    interior_grads = grad(interior_loss)(params)
    interior_updates, interior_opt_state = interior_optimizer.update(interior_grads, interior_opt_state)

    boundary_grads = grad(boundary_loss)(params)
    boundary_updates, boundary_opt_state = boundary_optimizer.update(boundary_grads, boundary_opt_state)

    updates = jax.tree_util.tree_map(
        lambda x, y: (x + y) / 2,
        interior_updates,
        boundary_updates,
    )
    params = optax.apply_updates(params, updates)
    
    if iteration % save_freq == 0:
        # errors
        l2_error = l2_norm(v_error, eval_integrator)
        h1_error = l2_error + l2_norm(v_error_abs_grad, eval_integrator)

        data[iteration // save_freq, :] = [
            iteration,
            interior_loss,
            boundary_loss,
            l2_error,
            h1_error,
        ]

        print(
            f'Seed: {seed} MultiAdam Iteration: {iteration}'
            f'\n  with loss: {loss(params)} = {interior_loss(params)} + {boundary_loss(params)}'
            f'\n  with relative L2 error: {l2_error / norm_sol_l2}'
            f'\n  with relative H1 error: {h1_error / norm_sol_h1}'
        )

np.save("data/multiadam/poisson-2d.npy")
