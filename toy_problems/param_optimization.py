import optuna
import wandb
import numpy as np
import json
import multiprocessing


# Load Custom Modules
from core import optimize_function, initial_point
from problems import *

PROJECT_NAME = "Optimal Hyperparameters"


storage = "sqlite:///gaf_tfn_hyperparams.db?timeout=10.0"

def calc_loss(x, x_opt):
    return jnp.linalg.norm(x - x_opt)

# Noise-free case
noise = True

params = []

def objective(trial, alg, problem):
    # # Suggest hyperparameters
    # algorithm = trial.suggest_categorical('algorithm', ['sgd', 'adam'])
    # problem = trial.suggest_categorical('problem', ['cone', 'ackleys'])

    # Set problem-specific parameters
    if problem == 'cone':
        domain = [-10, 10]
        true_params = (0, 0, 1)
        noise_params = [0.0*p for p in true_params]
        x_opt = jnp.array([0, 0])
    elif problem == 'ackleys':
        domain = [-10, 10]
        true_params = [20, 0.2, 2*np.pi]
        noise_params = [0.0*p for p in true_params]
        x_opt = jnp.array([0, 0])

    # Set common parameters
    solver_steps = trial.suggest_int('steps', 10, 100, log=True)
    solver_params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 10, log=True),
    }
    batch_size = trial.suggest_int('batch_size', 1, 50)

    # Initialize wandb for this trial
    wandb.init(
        project=f"Optimal Hyperparameters - {problem.upper()} - {alg}",
        config=trial.params,
        reinit=True,
        name=f"{alg}_{problem}_{trial.number}",
    )

    # Tag the run with algorithm and problem for easy filtering
    wandb.run.tags = [alg, problem, 'noise_free']

    # Save true params and noise params in config:
    wandb.config.update({
        'true_params': true_params,
        'noise_params': noise_params,
        'x_opt': [float(x) for x in x_opt],
        'use_gaf': False,
        'noise_free': True,
    })

    N_trials = 25  # Number of internal trials per hyperparameter set
    objective_values = []

    # Get Optimization Function
    fn = get_function_by_name(problem)

    seeds = []
    initial_points = []

    for _ in range(N_trials):
        # Get Random point
        pi = initial_point(minval=jnp.array([domain[0], domain[0]]), maxval=jnp.array([domain[1], domain[1]]))

        seed = np.random.randint(0, 2**32)

        path = optimize_function(
            fn,
            pi,
            true_params,
            noise_params,
            steps=solver_steps,
            solver_params=solver_params,
            batch_size=batch_size,
            method=alg,
            seed=seed
        )

        # Compute objective
        objective = calc_loss(path[-1], x_opt)
        objective_values.append(objective)
        seeds.append(seed)
        initial_points.append(pi)

        wandb.log({'objective_value': objective_values[-1]})

    # Compute mean and std of objective values
    mean_obj_value = np.mean(objective_values)

    # Log metrics to wandb
    wandb.log({'mean_objective_value': mean_obj_value, 'seeds': seeds, 'pi_x': [float(x[0]) for x in initial_points], 'pi_y': [float(x[1]) for x in initial_points]})

    # Finish the wandb run
    wandb.finish()

    # Return the mean objective value for Optuna to optimize
    return mean_obj_value


def run_study(alg, problem):

    study = optuna.create_study(
        study_name=f'TFOpt Params - Function: {problem.upper()}, Algorithm: {alg}',
        direction="minimize",
        storage=f"sqlite:///gaf_tfn_hyperparams.db?timeout=10.0",
        load_if_exists=True
    )
    study.optimize(lambda o: objective(o, alg, problem), n_trials=250)

    print("Study Name: ", study.study_name)
    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == '__main__':

    for problem in ['cone', 'ackleys']:
        for alg in ['sgd', 'adam']:

            n_workers = multiprocessing.cpu_count()
            processes = []

            for _ in range(n_workers):
                p = multiprocessing.Process(target=lambda: run_study(alg, problem))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()