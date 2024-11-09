from multiprocessing.managers import Value

import numpy as np
import jax
import jax.numpy as jnp
import optax
from typing import Any, Tuple
from jax import Array

METHODS = ['adadelta', 'adan',  'adagrad', 'adam', 'adamax', 'adafactor', 'sgd', 'yogi'] # 'rmsprop', 'lamb', 'lars', 'lbfgs']

GAF_METHODS = ['pairwise']

def rand_int():
    return np.random.randint(0, 2**32)

def random_fn(fn, params, noise_params, key) -> tuple[Any, Array]:

    if len(noise_params) != len(params):
        raise ValueError("Noise params must be the same length as params")

    keys = jax.random.split(key, len(noise_params))

    perturbed_params = []

    # Perturb each parameter separately with Guassian noise
    for i, param in enumerate(noise_params):
        perturbed_params.append(params[i] + float((jax.random.normal(keys[i], (1,)) * noise_params[i])[0]))

    return lambda x: fn(x, perturbed_params), keys[-1]

def perturb_gradient(grad, meas_noise_params, key) -> tuple[jax.Array, jax.Array]:

    keys = jax.random.split(key, grad.shape[0])

    perturbed_gradient = []
    for i in range(grad.shape[0]):
        # Compute magnitude of gradident component
        grad_i_mag = jnp.linalg.norm(grad[i])

        perturbed_gradient.append(grad[i] + float((jax.random.normal(keys[i], (1,)) * (meas_noise_params[1] / 100.) * grad_i_mag + meas_noise_params[0])[0]))


    return jnp.array(perturbed_gradient), keys[-1]

def get_gradient(fn, x: jax.Array, true_params: list, noise_params: list, meas_noise_params: list, key: jax.random.PRNGKey, batch_size: int = 1, gaf_params: dict | None = None) -> \
        tuple[float | Any, Any]:

    if gaf_params is not None:
        # Do gradient update with GAF

        grads, key = get_gradient_batch(fn, x, true_params, noise_params, meas_noise_params, key, batch_size)
        grad, key = gradient_agreement_filter(grads, gaf_params, key)

        retries = gaf_params.get('max_retries_on_failure', 0)
        if grad is None and retries > 0:
            # If we didn't get a valid gradient, try again
            while retries > 0:
                grads, key = get_gradient_batch(fn, x, true_params, noise_params, meas_noise_params, key, batch_size)
                grad, key = gradient_agreement_filter(grads, gaf_params, key)
                retries -= 1

        return grad, key
    else:
        # Just do regular gradient update based on mini-batch
        grads, key = get_gradient_batch(fn, x, true_params, noise_params, meas_noise_params, key, batch_size)

        return sum(grads) / len(grads), key

def get_gradient_batch(fn, x: jax.Array, true_params: list, noise_params: list, meas_noise_params: list, key: jax.random.PRNGKey, batch_size: int = 1) -> \
        tuple[list[jax.Array], jax.random.PRNGKey]:

    grads = []

    keys = jax.random.split(key, batch_size + 1)

    for i in range(batch_size):
        rfn, key = random_fn(fn, true_params, noise_params, keys[i])

        grad, key = perturb_gradient(jax.grad(rfn)(x), meas_noise_params, key)

        grads.append(grad)

    return grads, keys[-1]
def gradient_agreement_filter(grads: list[jax.Array], gaf_params: dict, key) -> tuple[jax.Array, Any]:
    """
    Filter batch of gradients using a gradient agreement filter (GAF). The GAF method and parameters are specified in
    the gaf_params dictionary.

    The potential parameters are:
    - method: str
        The GAF method to use. Must be one of 'pairwise'
    - cos_dist_max: float
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
        grad, key = pairwise_agreement_filter(
            grads,
            gaf_params.get('cos_dist_max', float(jnp.cos(87.7 * jnp.pi / 180.))),
            key,
            gaf_params.get('pairwise_multipass', False)
        )

    else:
        raise ValueError(f"Method {method} not a supported GAF method. Must be one of {GAF_METHODS}")

    return grad, key


def pairwise_agreement_filter(grads: list[jax.Array], threshold: float, key: jax.Array, multipass: bool = False) -> tuple[jax.Array | None, jax.Array]:
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

        # If we're not doing multiple passes, break after the first pass, even if we didn't find a valid gradient
        if not multipass:
            break

    return grad, good_key

def initial_point(seed: int | None = None, minval = 0., maxval = 1., dims:int = 2) -> jax.Array:
    """
    Generate a random initial point for optimization. The point is generated using a uniform distribution between
    minval and maxval.

    Args:
        seed: int | None
            Random seed for generating the initial point
        minval:

        maxval:
            Maximum value for the uniform distribution
        dims:
            Number of dimensions for the initial point

    Returns:
        jax.Array:
            Random initial point

    """
    if dims < 1:
        raise ValueError("Dimensions must be greater than 0")

    if seed is not None:
        key = jax.random.PRNGKey(seed)
    else:
        key = jax.random.PRNGKey(np.random.randint(0, 2**32))
    return jax.random.uniform(key, (dims,), minval=minval, maxval=maxval)

def initial_point_circle(seed: int | None = None, radius: float = 1., center: jax.Array = jnp.array([0., 0.]), rnoise: float = 0.0) -> jax.Array:
    """
    Initialize a point on a circle at with a given radius and center. The point can be optionally perturbed with
    Gaussian noise.

    Args:
        seed: int | None
            Random seed for generating the initial point
        radius: float
            Radius of the circle
        center: jax.Array
            Center of the circle
        rnoise: float
            Standard deviation of Gaussian noise to add to the radius

    Returns:
        jax.Array:
            Initial point on the circle
    """

    if seed is not None:
        key = jax.random.PRNGKey(seed)
    else:
        key = jax.random.PRNGKey(np.random.randint(0, 2**32))

    keys = jax.random.split(key, 2)

    r = radius + jax.random.normal(keys[0], (1,)) * rnoise
    theta = jax.random.uniform(keys[1], (1,), minval=0., maxval=2*jnp.pi)

    x = float((center[0] + r*jnp.cos(theta))[0])
    y = float((center[1] + r*jnp.sin(theta))[0])

    return jnp.array([x, y])

def optimize_function(fn, p0: jax.Array,
                         true_params: list,
                         perturbation_params: list,
                         meas_noise_params=None,
                         steps: int=10,
                         learning_rate: float=0.1,
                         batch_size: int=1,
                         solver_params: dict | None = None,
                         gaf_params: dict | None = None,
                         method: str = 'sgd',
                         seed: int | None = None) -> list[jax.Array]:

    if len(perturbation_params) != len(true_params):
        raise ValueError("Noise params must be the same length as true params")

    if meas_noise_params is None:
        meas_noise_params = [0., 0.]

    if len(meas_noise_params) != 2:
        raise ValueError("Measurement noise params must be length 2")

    # If solver_params is None, set to empty dictionary
    if solver_params is None:
        solver_params = {'learning_rate': learning_rate}

    if 'learning_rate' not in solver_params:
        solver_params['learning_rate'] = learning_rate

    # Create function to optimize
    f = lambda x: fn(x, true_params)

    # Create solver
    if method in METHODS:
        solver = optax.sgd(**solver_params)
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
            # # value, grad, key = get_batch_value_and_gradient(p, true_params, perturbation_params, key, batch_size=batch_size)
            # updates, opt_state = solver.update(
            #     grad, opt_state, p, value=value, grad=grad, value_fn=f_cone
            # )
        else:
            # Note we need to pass in the "raw" function here with optional parameter inputs
            # to get new instances of the function with different noise parameters
            grad, key = get_gradient(fn, p, true_params, perturbation_params, meas_noise_params, key, batch_size=batch_size, gaf_params=gaf_params)

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