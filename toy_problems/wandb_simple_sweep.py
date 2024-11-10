import wandb
import jax.numpy as jnp

from core import optimize_function, initial_point
from problems import *

def calf_loss(x, x_opt):
    return jnp.linalg.norm(x - x_opt)

def objective():
    run = wandb.init()

    # Set problem-specific parameters
    domain = [-10, 10]
    true_params = (0, 0, 1)
    noise_params = [0.0*p for p in true_params]
    x_opt = jnp.array([0, 0])

    # Set common parameters
    solver_steps = 10
    solver_params = {
        'learning_rate': wandb.config.learning_rate,
    }
    batch_size = 1

    # Initialize wandb for this trial
    wandb.init(
        project=f"Optimal Hyperparameters - cone - bayes",
        reinit=True,
        name=f"bayes_cone",
    )

    # Tag the run with algorithm and problem for easy filtering
    wandb.run.tags = ['bayes', 'cone', 'noise_free']

    # Save true params and noise params in config:
    wandb.config.update({
        'true_params': true_params,
        'noise_params': noise_params,
        'x_opt': x_opt,
        'solver_steps': solver_steps,
        'solver_params': solver_params,
        'batch_size': batch_size,
    })

    # Run optimization
    p0 = initial_point()
    path = optimize_function(
        p0, true_params,
        noise_params,
        solver_steps,
        solver_params,
        wandb.config.batch_size,
        calf_loss,
        x_opt
    )

    # Log the optimization path
    wandb.log({
        'inital_point': [float(x) for x in p0],
        'final_point': [float(x) for x in path[-1]],
    })

    return x_opt

def run_sweep():
    # Problem Configuration
    problem


    sweep_configuration = {
        "name": "Optimial Base Hyperparameters",
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "validation_loss"},
        "parameters": {
            "learning_rate": {"min": 0.0001, "max": 1},
            "batch_size": {"min": 1, "max": 50},
            "optimizer": {"values": ["adam", "sgd"]},
        },
    }

    sweep_id = wandb.sweep(sweep_configuration, project="Optimal Hyperparameters - cone - bayes")


