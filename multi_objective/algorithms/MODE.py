# -*- coding: utf-8 -*-

import numpy as np

from multi_objective.mo_utils import (
    best_sum_history_value,
    dominates,
    evaluate_objectives,
    update_archive,
    validate_bounds,
)


def _select_base_indexes(population_size, current_index):
    candidates = [index for index in range(population_size) if index != current_index]
    return np.random.choice(candidates, size=3, replace=False)


def _create_trial_vector(population, current_index, bounds, mutation_factor, crossover_rate):
    population_size, dimension = population.shape
    r1, r2, r3 = _select_base_indexes(population_size, current_index)
    mutant = population[r1] + mutation_factor * (population[r2] - population[r3])

    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]
    mutant = np.clip(mutant, lower_bounds, upper_bounds)

    crossover_mask = np.random.rand(dimension) < crossover_rate
    crossover_mask[np.random.randint(dimension)] = True
    trial = np.where(crossover_mask, mutant, population[current_index])
    return np.clip(trial, lower_bounds, upper_bounds)


def _select_survivor(target, target_objective, trial, trial_objective):
    if dominates(trial_objective, target_objective):
        return trial, trial_objective
    if dominates(target_objective, trial_objective):
        return target, target_objective
    if np.random.rand() < 0.5:
        return trial, trial_objective
    return target, target_objective


def mode(
    objective_function,
    bounds,
    population_size=80,
    max_iter=800,
    mutation_factor=0.5,
    crossover_rate=0.9,
    archive_size=100,
    seed=None,
):
    """Basic multi-objective differential evolution with Pareto greedy selection."""
    if seed is None:
        used_seed = int(np.random.SeedSequence().generate_state(1)[0])
    else:
        used_seed = int(seed)
    np.random.seed(used_seed)

    bounds = validate_bounds(bounds)
    dimension = len(bounds)
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]

    population_size = max(4, int(population_size))
    population = np.random.uniform(lower_bounds, upper_bounds, size=(population_size, dimension))
    objectives = evaluate_objectives(objective_function, population)

    archive_solutions = np.empty((0, dimension), dtype=float)
    archive_objectives = np.empty((0, objectives.shape[1]), dtype=float)
    archive_solutions, archive_objectives = update_archive(
        archive_solutions,
        archive_objectives,
        population,
        objectives,
        archive_size,
    )
    history = [best_sum_history_value(archive_objectives)]

    for _ in range(max_iter):
        next_population = population.copy()
        next_objectives = objectives.copy()

        for index in range(population_size):
            trial = _create_trial_vector(population, index, bounds, mutation_factor, crossover_rate)
            trial_objective = objective_function(trial)
            next_population[index], next_objectives[index] = _select_survivor(
                population[index],
                objectives[index],
                trial,
                trial_objective,
            )

        population = next_population
        objectives = next_objectives
        archive_solutions, archive_objectives = update_archive(
            archive_solutions,
            archive_objectives,
            population,
            objectives,
            archive_size,
        )
        history.append(best_sum_history_value(archive_objectives))

    return archive_solutions, archive_objectives, history, used_seed
