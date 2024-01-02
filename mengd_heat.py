"""
Multi-ENGD Optimization.
One dimensional heat equation example. Solution given by

u(t,x) = exp(pi**2 * t * 0.25) * sin(pi * x).

"""
import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
from jax.flatten_util import ravel_pytree

from ngrad.models import mlp, init_params
from ngrad.domains import Square, SquareBoundary
from ngrad.integrators import DeterministicIntegrator
from ngrad.inner import model_identity, model_del_i_factory
from ngrad.gram import gram_factory, nat_grad_factory
from ngrad.utility import grid_line_search_factory, del_i

jax.config.update("jax_enable_x64", True)

# random seed
seed = 0

# domains
interior = Square(1.)
initial = SquareBoundary(1., side_number=3)
rboundary = SquareBoundary(1., side_number=0)
lboundary = SquareBoundary(1., side_number=2)

# integrators
interior_integrator = DeterministicIntegrator(interior, 30)
initial_integrator = DeterministicIntegrator(initial, 30)
rboundary_integrator = DeterministicIntegrator(rboundary, 30)
lboundary_integrator = DeterministicIntegrator(lboundary, 30)
eval_integrator = DeterministicIntegrator(interior, 300)

# model
activation = lambda x : jnp.tanh(x)
layer_sizes = [2, 64, 1]
params = init_params(layer_sizes, random.PRNGKey(seed))
model = mlp(activation)
v_model = vmap(model, (None, 0))

# initial condition
def u_0(tx):
    x = tx[1]
    return jnp.sin(jnp.pi * x)
v_u_0 = vmap(u_0, (0))

# solution
def u_star(tx):
    t = tx[0]
    x = tx[1]
    return jnp.exp(-jnp.pi**2 * t * 0.25) * jnp.sin(jnp.pi * x)

v_u_star = vmap(u_star, (0))
v_grad_u_star = vmap(
    lambda x: jnp.dot(grad(u_star)(x), grad(u_star)(x))**0.5, (0)
    )

# defining heat eq inner product
model_del_0 = model_del_i_factory(argnum=0)
model_del_1 = model_del_i_factory(argnum=1)

def model_heat_eq_factory(diffusivity=1.):
    def model_heat_eq(u_theta, g):
        dg_1 = model_del_0(u_theta, g)
        ddg_2 = model_del_1(u_theta, (model_del_1(u_theta, g)))

        def return_heat_eq(x):
            flat_dg_1, unravel = ravel_pytree(dg_1(x))
            flat_ddg_2, unravel = ravel_pytree(ddg_2(x))
            return unravel(flat_dg_1 - diffusivity * flat_ddg_2)
        
        return return_heat_eq

    return model_heat_eq

# assembling gramians
gram_l_boundary = gram_factory(
    model = model,
    trafo = model_identity,
    integrator = lboundary_integrator
)

gram_r_boundary = gram_factory(
    model = model,
    trafo = model_identity,
    integrator = rboundary_integrator
)

gram_initial = gram_factory(
    model = model,
    trafo = model_identity,
    integrator = initial_integrator
)

model_heat_eq = model_heat_eq_factory(0.25)
gram_heat = gram_factory(
    model = model,
    trafo = model_heat_eq,
    integrator = interior_integrator
)

# the full inner product
@jit
def gram(params):
    return (
    gram_l_boundary(params) + 
    gram_r_boundary(params) + 
    gram_initial(params) + 
    gram_heat(params)
    )

# maps: params, tangent_params ---> tangent_params
nat_grad = nat_grad_factory(gram)

# differential operators
dt = lambda g: del_i(g, 0)
ddx = lambda g: del_i(del_i(g, 1), 1)
def heat_operator(u):
    return lambda tx: (dt(u)(tx) - 0.25 * ddx(u)(tx))**2

# loss terms
@jit
def interior_loss(params):
    heat_model = heat_operator(lambda tx: model(params, tx))
    return interior_integrator(vmap(heat_model, (0)))
@jit
def boundary_loss(params):
    return (
        lboundary_integrator(lambda tx: v_model(params, tx)**2) 
            + rboundary_integrator(lambda tx: v_model(params, tx)**2))
@jit
def initial_loss(params):
    return initial_integrator(
        lambda tx: (v_u_0(tx) - v_model(params, tx))**2)
@jit
def loss(params):
    return interior_loss(params) + boundary_loss(params) + initial_loss(params)

# set up grid line search
grid = jnp.linspace(0, 30, 31)
steps = 0.5**grid
ls_update = grid_line_search_factory(loss, steps)    

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

# training loop
iterations = 1000
save_freq = 10

import numpy as np
data = np.empty((iterations // save_freq + 1, 6))

for iteration in range(iterations + 1):
    interior_grads = grad(interior_loss)(params)
    interior_nat_grads = nat_grad(params, interior_grads)

    boundary_grads = grad(boundary_loss)(params)
    boundary_nat_grads = nat_grad(params, boundary_grads)

    initial_grads = grad(initial_loss)(params)
    initial_nat_grads = nat_grad(params, initial_grads)

    updates = jax.tree_util.tree_map(
        lambda x, y, z: (x + y + z) / 3,
        interior_nat_grads,
        boundary_nat_grads,
        initial_nat_grads,
    )
    params, actual_step = ls_update(params, updates)


    if iteration % save_freq == 0:
        l2_error = l2_norm(v_error, eval_integrator)
        h1_error = l2_error + l2_norm(v_error_abs_grad, eval_integrator)

        data[iteration // save_freq, :] = [
            iteration,
            interior_loss,
            boundary_loss,
            initial_loss,
            l2_error,
            h1_error,
        ]

        print(
            f'Seed: {seed} Multi-ENGD Iteration: {iteration}'
            f'\n  with loss: {loss(params)} = {interior_loss(params)} + {boundary_loss(params)} + {initial_loss(params)}'
            f'\n  with relative L2 error: {l2_error/norm_sol_l2}'
            f'\n  with relative H1 error: {h1_error/norm_sol_h1}'
            f'\n  with step: {actual_step}'
        )

np.save("data/multi-engd/heat.npy")
