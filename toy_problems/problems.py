"""
Library of toy problems for testing optimization algorithms.
"""
from math import atan2

import jax.numpy as jnp

def cone_function(x, params):
    """
    Compute a cone function with parameterized center and slope.

    Args:
        x: jnp.array
            Input to the cone function
        params:
            Parameters for the cone function. The parameters are:
            - params[0]: float
                Center x-coordinate of the cone
            - params[1]: float
                Center y-coordinate of the cone
            - params[2]: float
                Slope of the cone

    Returns:
        jnp.array:
            Value of the cone function at x
    """

    if x.shape[-1] != 2:
        raise ValueError("Cone function requires 2D input")

    if len(params) != 3:
        raise ValueError("Cone function requires 3 parameters")

    return params[2]*jnp.linalg.norm(x - jnp.array([params[0], params[1]]), axis=-1)

def ackleys_function(x, params):
    """
    Compute the value of the Ackley's function at x

    Args:
        x: jnp.array
            Input to the Ackley's function
        params:
            Parameters for the Ackley's function. The parameters are:
            - params[0]: float
                Value for "a" constant

    Returns:
        jnp.array:
            Value of the Ackley's function at x
    """

    if len(params) != 3:
        raise ValueError("Ackley's function requires 3 parameters")

    d = x.shape[-1] if isinstance(x, jnp.ndarray) else len(x)


    a = params[0]
    b = params[1]
    c = params[2]

    return -a * jnp.exp(-b * jnp.sqrt(jnp.sum(x ** 2, axis=-1) / d)) + -jnp.exp(jnp.sum(jnp.cos(c * x), axis=-1) / d) + a + jnp.exp(1)


def booths_function(x, params: float | None = None):
    """
    Compute the value of the Booth's function at x

    Args:
        x: jnp.array
            Input to the Booth's function
        params:
            Parameters for the Booth's function. The parameters are:
            - params[0]: float (optional)
                value for fixed constant
    Returns:
        jnp.array:
            Value of the Booth's function at x
    """

    if isinstance(params, int) or isinstance(params, float):
        params = [params]

    if len(params) > 1:
        raise ValueError("Booth's function requires eactly 1 parameter")
    elif params is None:
        params = [0]

    x1 = x[..., 0]
    x2 = x[..., 1]

    return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2 + params[0]

def branin_function(x, params):
    """
    Compute the value of the Branin's function at x

    Args:
        x: jnp.array
            Input to the Branin's function
        params:
            Parameters for the Branin's function. The parameters are:
            - params[0]: float
                value for the a constant
            - params[1]: float
                value for the b constant
            - params[2]: float
                value for the c constant
            - params[3]: float
                value for the r constant
            - params[4]: float
                value for the s constant
            - params[5]: float
                value for the t constant

    Returns:
        jnp.array:
            Value of the Branin's function at x
    """

    if x.shape[-1] != 2:
        raise ValueError("Branin's function requires 2D input")

    if len(params) != 6:
        raise ValueError("Branin's function requires 5 parameters")

    a = params[0]
    b = params[1]
    c = params[2]
    r = params[3]
    s = params[4]
    t = params[5]

    x1 = x[..., 0]
    x2 = x[..., 1]

    return a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*jnp.cos(x1) + s

def flower_function(x, params):
    """
    Compute the value of the Flower function at x

    Args:
        x: jnp.array
            Input to the Flower function
        params:
            Parameters for the Flower function. The parameters are:
            - params[0]: float
                value for the a constant
            - params[1]: float
                value for the b constant
            - params[2]: float
                value for the c constant

    Returns:
        jnp.array:
            Value of the Flower function at x
    """

    if x.shape[-1] != 2:
        raise ValueError("Flower function requires 2D input")

    if len(params) != 3:
        raise ValueError("Flower function requires 3 parameters")

    a = params[0]
    b = params[1]
    c = params[2]

    x1 = x[..., 0]
    x2 = x[..., 1]

    return a*jnp.sqrt(x1**2 + x2**2) + b*jnp.sin(c*jnp.atan2(x2, x1))

def michalewicz_function(x, params: list | None = None):
    """
    Compute the value of the Michalewicz's function at x

    Args:
        x: jnp.array
            Input to the Michalewicz's function
        params:
            Parameters for the Michalewicz's function. The parameters are:
            - params[0]: float
                value for the m constant

    Returns:
        jnp.array:
            Value of the Michalewicz's function at x
    """

    if isinstance(params, int) or isinstance(params, float):
        params = [params]

    if len(params) > 1:
        raise ValueError("Michalewicz's function requires eactly 1 parameter")
    elif params is None:
        params = [10]

    m = params[0]

    d = x.shape[-1]
    indices = jnp.arange(1, d + 1)
    return -jnp.sum(jnp.sin(x) * jnp.sin(indices * x**2 / jnp.pi) ** (2 * m), axis=-1)

def rosenbrock_function(x, params):
    """
    Compute the value of the Rosenbrock's function at x

    Args:
        x: jnp.array
            Input to the Rosenbrock's function
        params:
            Parameters for the Rosenbrock's function. The parameters are:
            - params[0]: float
                value for the a constant
            - params[1]: float
                value for the b constant

    Returns:
        jnp.array:
            Value of the Rosenbrock's function at x
    """

    if x.shape[-1] != 2:
        raise ValueError("Rosenbrock's function requires 2D input")

    if len(params) != 2:
        raise ValueError("Rosenbrock's function requires 2 parameters")

    a = params[0]
    b = params[1]

    x1 = x[..., 0]
    x2 = x[..., 1]

    return (a - x1)**2 + b*(x2 - x1**2)**2

def wheelers_ridge_function(x, params):
    """
    Compute the value of the Wheeler's Ridge function at x

    Args:
        x: jnp.array
            Input to the Wheeler's Ridge function
        params:
            Parameters for the Wheeler's Ridge function. The parameters are:
            - params[0]: float
                value for the a constant

    Returns:
        jnp.array:
            Value of the Wheeler's Ridge function at x
    """

    if x.shape[-1] != 2:
        raise ValueError("Wheeler's Ridge function requires 2D input")

    if isinstance(params, int) or isinstance(params, float):
        params = [params]

    if len(params) != 1:
        raise ValueError("Wheeler's Ridge function requires 1 parameter")

    a = params[0]

    x1 = x[..., 0]
    x2 = x[..., 1]

    return -jnp.exp(-(x1*x2 - a)**2 - (x2 - a)**2)

def get_function_by_name(name: str) -> callable:
    """
    Get a function by its name

    Args:
        name: str
            Name of the function to retrieve

    Returns:
        callable:
            Function corresponding to the name
    """

    # Get all functions that follow patterm of *_function
    functions = [f for f in globals() if f.endswith("_function")]

    # Get the function that matches the name
    for function in functions:
        if name == function[:-9]:
            return globals()[function]
    else:
        raise ValueError(f"Unknown function name: {name}. Available functions are: {[f.removesuffix('_function') for f in functions]}")