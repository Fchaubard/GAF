from multiprocessing.managers import Value

import numpy as np
import jax
import jax.numpy as jnp
import optax
from typing import Any, Tuple
from jax import Array

METHODS = ['sgd', 'adam', 'lamb', 'lars', 'adagrad', 'rmsprop']

GAF_METHODS = ['pairwise']

def rand_int():
    return np.random.randint(0, 2**32)

def random_fn(fn, params, noise_params, key) -> tuple[Any, Array]:

    if len(noise_params) != len(params):
        raise ValueError("Noise params must be the same length as params")

    keys = jax.random.split(key, len(noise_params) + 1)

    perturbed_params = []

    # Perturb each parameter separately with Guassian noise
    for i, param in enumerate(noise_params):
        perturbed_params.append(params[i] + float((jax.random.normal(keys[i], (1,)) * noise_params[i])[0]))

    return lambda x: fn(x, perturbed_params), keys[-1]

def get_batch_gradient(fn, x: jax.Array, true_params: list, noise_params: list, key: jax.random.PRNGKey, batch_size: int = 1, gaf_params: dict | None = None) -> \
        tuple[float | Any, Any]:

    grads = []

    for _ in range(batch_size):
        rfn, key = random_fn(fn, true_params, noise_params, key)

        grads.append(jax.grad(rfn)(x))

    if gaf_params is not None:
        grad, key = gradient_agreement_filter(grads, gaf_params, key)
    else:
        # If no GAF parameters, just average the gradients
        grad = sum(grads) / batch_size

    return grad, key

def pairwise_agreement_filter(grads: list[jax.Array], threshold: float, key: jax.Array) -> tuple[jax.Array | None, jax.Array]:
    """
    Filter gradients based on pairwise agreement. Iterates through all potential starting graidents, then
    checks for pairwise agreement with other gradients, successively averaging gradients that agree. If no agreement
    is found, another random gradient is selected to start the process again until all potential gradients have been
    exhausted. If no agreement is found then no gradient is returned.

    Args:
        grads: list[jax.Array]
            List of gradients to filter
        threshold: float
            Cosine distance threshold for agreement between gradients. Must be between 0 and 2.
        key: jax.random.PRNGKey
            Random key for generating random

    Returns:
        jax.Array | None:
            Filtered gradient
        jax.Array:
            Random key
    """

    # Check if threshold is valid
    if threshold < 0 or threshold > 2:
        raise ValueError("Threshold must be between 0 and 2")

    grad = None

    starting_grads = set(range(len(grads)))
    valid_found = False

    good_key = key

    while len(starting_grads) > 0:

        use_key, good_key = jax.random.split(good_key)

        grad_idx = int(jax.random.choice(use_key, jnp.array(list(starting_grads))))
        grad_candidate = grads[grad_idx]
        starting_grads.remove(grad_idx)

        # Potential gradients to combine with
        remaining_grads = set(range(len(grads)))
        remaining_grads.remove(grad_idx)

        while len(remaining_grads) > 0:
            # Randomly select another gradient to compare
            use_key, good_key = jax.random.split(good_key)
            grad_2_idx = int(jax.random.choice(use_key, jnp.array(list(remaining_grads))))

            # Compute cosine distance
            cos_sim  = jnp.dot(grad_candidate, grads[grad_2_idx]) / (jnp.linalg.norm(grad_candidate) * jnp.linalg.norm(grads[grad_2_idx]))
            cos_dist = 1 - cos_sim

            if cos_dist > threshold:
                remaining_grads.remove(grad_2_idx)
            else:
                grad_candidate = (grad_candidate + grads[grad_2_idx]) / 2
                remaining_grads.remove(grad_2_idx)
                valid_found = True

        if valid_found:
            grad = grad_candidate
            break

    return grad, good_key


def gradient_agreement_filter(grads: list[jax.Array], gaf_params: dict, key) -> tuple[jax.Array, Any]:
    """
    Filter batch of gradients using a gradient agreement filter (GAF). The GAF method and parameters are specified in
    the gaf_params dictionary.

    The potential parameters are:
    - method: str
        The GAF method to use. Must be one of 'pairwise'
    - threshold: float
        The cosine distance threshold for agreement between gradients. Must be between 0 and 2.
    - key: jax.random.PRNGKey
        Random key for generating random

    Args:
        grads: list[jax.Array]
            List of gradients to filter
        gaf_params: dict
            Dictionary containing GAF method and parameters.

    Returns:
        jax.Array:
            Filtered gradient
    """

    grad = None

    if 'method' not in gaf_params:
        raise ValueError("GAF \"method\" not specified in gaf_params")
    else:
        method = gaf_params['method']

    if method == 'pairwise':
        if 'threshold' not in gaf_params:
            threshold = jnp.cos(85. * jnp.pi / 180.) # Use 85 degree threshold as default
        else:
            threshold = gaf_params['threshold']

        grad, key = pairwise_agreement_filter(grads, threshold, key)

    else:
        raise ValueError(f"Method {method} not a supported GAF method. Must be one of {GAF_METHODS}")

    return grad, key

def initial_point(seed: int | None = None, minval = 0., maxval = 1., dims:int = 2) -> jax.Array:
    if dims < 1:
        raise ValueError("Dimensions must be greater than 0")

    if seed is not None:
        key = jax.random.PRNGKey(seed)
    else:
        key = jax.random.PRNGKey(np.random.randint(0, 2**32))
    return jax.random.uniform(key, (dims,), minval=minval, maxval=maxval)

def optimize_function(fn, p0: jax.Array,
                         true_params: list,
                         noise_params: list,
                         steps: int=10,
                         learning_rate: float=0.1,
                         batch_size: int=1,
                         solver_params: dict | None = None,
                         gaf_params: dict | None = None,
                         method: str = 'sgd',
                         seed: int | None = None) -> list[jax.Array]:

    if len(noise_params) != len(true_params):
        raise ValueError("Noise params must be the same length as true params")

    # If solver_params is None, set to empty dictionary
    if solver_params is None:
        solver_params = {'learning_rate': learning_rate}

    if 'learning_rate' not in solver_params:
        solver_params['learning_rate'] = learning_rate

    # Create function to optimize
    f = lambda x: fn(x, true_params)

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
    elif method == 'rmsprop':
        solver = optax.rmsprop(**solver_params)
    # elif method == 'lbfgs':
    #     solver = optax.lbfgs(**solver_params)
    else:
        raise ValueError(f"Method {method} not a supported optimization method. Must be one of {METHODS}")

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
            # value, grad = optax.value_and_grad_from_state(f)(p, state=opt_state)
            # # value, grad, key = get_batch_value_and_gradient(p, true_params, noise_params, key, batch_size=batch_size)
            # updates, opt_state = solver.update(
            #     grad, opt_state, p, value=value, grad=grad, value_fn=f_cone
            # )
        else:
            # Note we need to pass in the "raw" function here with optional parameter inputs
            # to get new instances of the function with different noise parameters
            grad, key = get_batch_gradient(fn, p, true_params, noise_params, key, batch_size=batch_size, gaf_params=gaf_params)

            # If we didn't get a gradient, don't do any updates
            if grad is not None:
                updates, opt_state = solver.update(grad, opt_state, p)

        # Only apply updates if we have a step
        if grad is not None:
            p = optax.apply_updates(p, updates)

        # Save step variables
        grads.append(grad)
        states.append(p)
        opt_states.append(opt_state)


    return states