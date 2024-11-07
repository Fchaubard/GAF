import numpy as np
import jax
import jax.numpy as jnp
import optax
from typing import Any
from jax import Array
from numpy import bool

METHODS = ['sgd', 'adam', 'lamb', 'lars', 'adagrad', 'lbfgs', 'gaf']

def cone(x, params):
    """
    Compute the value of the cone function at x

    Args:
        x:
        params:

    Returns:

    """
    return params[2]*jnp.linalg.norm(x - np.array([params[0], params[1]]), axis=-1)

def random_cone(params, noise_params, key) -> tuple[Any, Array]:

    if len(noise_params) != len(params):
        raise ValueError("Noise params must be the same length as params")

    keys = jax.random.split(key, len(noise_params) + 1)

    perturbed_params = []

    # Perturb each parameter separately with Guassian noise
    for i, param in enumerate(noise_params):
        perturbed_params.append(params[i] + float((jax.random.normal(keys[i], (1,)) * noise_params[i])[0]))

    return lambda x: cone(x, perturbed_params), keys[-1]

def get_batch_gradient(x: jax.Array, true_params: list, noise_params: list, key: jax.random.PRNGKey, batch_size: int = 1) -> \
        tuple[float | Any, Any]:

    grad = None

    for _ in range(batch_size):
        rc, key = random_cone(true_params, noise_params, key)

        if grad is None:
            grad = jax.grad(rc)(x)
        else:
            grad += jax.grad(rc)(x)

    return grad/batch_size, key

def get_batch_value_and_gradient(x: jax.Array, true_params: list, noise_params: list, key: jax.random.PRNGKey, batch_size: int = 1) -> \
tuple[float | Any, float | Any, Any]:

    value = None
    grad = None

    for _ in range(batch_size):
        rc, key = random_cone(x, true_params, noise_params, key)

        if grad is None:
            value = rc(x)
            grad = jax.grad(rc)(x)
        else:
            value += rc(x)
            grad += jax.grad(rc)(x)

    return value/batch_size, grad/batch_size, key

def initial_point(seed: int | None = None, minval = 0., maxval = 1.) -> jax.Array:
    if seed is not None:
        key = jax.random.PRNGKey(seed)
    else:
        key = jax.random.PRNGKey(np.random.randint(0, 2**32))
    return jax.random.uniform(key, (2,), minval=minval, maxval=maxval)

def optimize_random_cone(p0: jax.Array,
                  true_params: list,
                  noise_params: list,
                  steps: int=10,
                  learning_rate: float=0.1,
                  batch_size: int=1,
                  solver_params: dict | None = None,
                  method: str = 'sgd',
                  return_gradients: bool = False,
                  seed: int | None = None) -> list[jax.Array]:

    if method not in METHODS:
        raise ValueError(f"Method {method} not a supported optimization method. Must be one of {METHODS}")

    if len(noise_params) != len(true_params):
        raise ValueError("Noise params must be the same length as true params")

    # If solver_params is None, set to empty dictionary
    if solver_params is None:
        solver_params = {'learning_rate': learning_rate}

    # Create function to optimize
    f_cone = lambda x: cone(x, true_params)

    # Create solver
    if method == 'sgd':
        solver = optax.sgd(**solver_params)
    elif method == 'adam':
        solver = optax.adam(**solver_params)
    elif method == 'lamb':
        solver = optax.lamb(**solver_params)
    elif method == 'lars':
        solver = optax.lars(**solver_params)
    elif method == 'adagrad':
        solver = optax.adagrad(**solver_params)
    elif method == 'lbfgs':
        solver = optax.lbfgs(**solver_params)

    # Initialize random key
    if seed is not None:
        key = jax.random.PRNGKey(seed)
    else:
        key = jax.random.PRNGKey(np.random.randint(0, 2**32))

    # Initialize optimizer and state
    p = p0
    opt_state = solver.init(p)

    # Running variables to save states and gradients
    states = [p]
    opt_states = [opt_state]
    grads  = []

    # Run optimization
    for _ in range(steps):
        # Update optimizer state
        if method == 'lbfgs':
            raise NotImplementedError("L-BFGS not yet implemented")
            value, grad = optax.value_and_grad_from_state(f_cone)(p, state=opt_state)
            # value, grad, key = get_batch_value_and_gradient(p, true_params, noise_params, key, batch_size=batch_size)
            updates, opt_state = solver.update(
                grad, opt_state, p, value=value, grad=grad, value_fn=f_cone
            )
        elif method == 'gaf':
            raise NotImplementedError("GAF not yet implemented")
            # value, grad = optax.value_and_grad_from_state(f_cone)(p, state=opt_state)
            # # value, grad, key = get_batch_value_and_gradient(p, true_params, noise_params, key, batch_size=batch_size)
            # updates, opt_state = solver.update(
            #     grad, opt_state, p, value=value, grad=grad, value_fn=f_cone
            # )
        else:
            # grad = jax.grad(f_cone)(p)
            grad, key = get_batch_gradient(p, true_params, noise_params, key, batch_size=batch_size)
            updates, opt_state = solver.update(grad, opt_state, p)

        p = optax.apply_updates(p, updates)

        # Save step variables
        grads.append(grad)
        states.append(p)
        opt_states.append(opt_state)


    return states