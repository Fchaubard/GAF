import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import wandb
import numpy as np


# Load Custom Modules
from core import optimize_function, initial_point
from problems import *

PROJECT_NAME = "Gradient Agreement Filtering"

def calc_loss(x, x_opt):
    return jnp.linalg.norm(x - x_opt)

def objective(trial):
    # Suggest hyperparameters
    algorithm = trial.suggest_categorical('algorithm', ['sgd', 'adam'])
    problem = trial.suggest_categorical('problem', ['cone', 'ackleys'])

    gaf_enabled = False

    # Set problem-specific parameters
    if problem == 'cone':
        domain = [-10, 10]
        true_params = [0, 0, 1]
        x_opt = jnp.array([0, 0])
        noise_params = [0., 0., 0.]
    elif problem == 'ackleys':
        domain = [-5, 5]
        true_params = [20, 0.2, 2*np.pi]
        noise_params = [0., 0., 0.]
        x_opt = jnp.array([0, 0])

    # Set algorithm-specific parameters
    if algorithm == 'sgd':
        pass
    elif algorithm == 'adam':
        pass

    # Set common parameters
    solver_steps = trial.suggest_categorical('steps', [100])
    solver_params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1, log=True),
    }
    batch_size = trial.suggest_int('batch_size', 1, 20)
    meas_noise = [0., 5.]

    # Initialize wandb for this trial
    wandb.init(
        project=PROJECT_NAME,
        config=trial.params,
        reinit=True,
        name=f"{algorithm}_{problem}_{trial.number}",
    )

    # Tag the run with algorithm and problem for easy filtering
    wandb.run.tags = [algorithm, problem]



    # Save true params and noise params in config:
    wandb.config.update({
        'true_params': true_params,
        'noise_params': noise_params,
        'x_opt': [float(x) for x in x_opt],
        'use_gaf': False,
        'meas_noise': meas_noise,
    })

    N_trials = 10  # Number of internal trials per hyperparameter set
    objective_values = []

    # Get Optimization Function
    fn = get_function_by_name(problem)

    for _ in range(N_trials):
        # Get Random point
        pi = initial_point(minval=jnp.array([domain[0], domain[0]]), maxval=jnp.array([domain[1], domain[1]]))

        path = optimize_function(
            fn,
            pi,
            true_params,
            noise_params,
            steps=solver_steps,
            solver_params=solver_params,
            batch_size=batch_size,
            method=algorithm,
        )

        # Compute objective
        objective = calc_loss(path[-1], x_opt)
        objective_values.append(objective)

    # Compute mean and std of objective values
    mean_obj_value = np.mean(objective_values)

    # Log metrics to wandb
    wandb.log({'objective_value': mean_obj_value})

    # Finish the wandb run
    wandb.finish()

    # Return the mean objective value for Optuna to optimize
    return mean_obj_value


wandb_cb = WeightsAndBiasesCallback(wandb_kwargs={"project": PROJECT_NAME}, as_multirun=True)
study = optuna.create_study(study_name="Hyperparameter Search", direction="minimize")
study.optimize(objective, n_trials=1000, callbacks=[wandb_cb], show_progress_bar=True)