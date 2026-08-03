# -*- coding: utf-8 -*-

import numpy as np

from multi_objective.mo_utils import (
    best_sum_history_value,
    evaluate_objectives,
    update_archive,
    validate_bounds,
)


def _weight_vectors(population_size, objective_count):
    if objective_count == 2:
        values = np.linspace(0.0, 1.0, population_size)
        weights = np.column_stack([values, 1.0 - values])
    else:
        weights = np.random.dirichlet(np.ones(objective_count), size=population_size)
        for index in range(min(objective_count, population_size)):
            weights[index] = 0.0
            weights[index, index] = 1.0
    return np.maximum(weights, 1e-6)


def _neighbor_indexes(weights, neighborhood_size):
    distances = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
    return np.argsort(distances, axis=1)[:, :neighborhood_size]


def _tchebycheff(objective, weight, ideal_point):
    return float(np.max(weight * np.abs(objective - ideal_point)))


def _select_indexes(index_pool, population_size, count):
    if len(index_pool) >= count:
        return np.random.choice(index_pool, size=count, replace=False)
    return np.random.choice(population_size, size=count, replace=False)


def _create_trial(population, current_index, parent_indexes, bounds, mutation_factor, crossover_rate):
    base, left, right = population[parent_indexes]
    mutant = base + mutation_factor * (left - right)

    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]
    mutant = np.clip(mutant, lower_bounds, upper_bounds)

    dimension = population.shape[1]
    crossover_mask = np.random.rand(dimension) < crossover_rate
    crossover_mask[np.random.randint(dimension)] = True
    trial = np.where(crossover_mask, mutant, population[current_index])
    return np.clip(trial, lower_bounds, upper_bounds)


def moead(
    objective_function,
    bounds,
    population_size=80,
    max_iter=800,
    neighborhood_size=20,
    mutation_factor=0.5,
    crossover_rate=0.9,
    archive_size=100,
    seed=None,
):
    """MOEA/D with Tchebycheff decomposition and DE variation."""
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
    neighborhood_size = max(3, min(int(neighborhood_size), population_size))

    population = np.random.uniform(lower_bounds, upper_bounds, size=(population_size, dimension))
    objectives = evaluate_objectives(objective_function, population)
    objective_count = objectives.shape[1]

    weights = _weight_vectors(population_size, objective_count)
    neighbors = _neighbor_indexes(weights, neighborhood_size)
    ideal_point = np.min(objectives, axis=0)

    archive_solutions = np.empty((0, dimension), dtype=float)
    archive_objectives = np.empty((0, objective_count), dtype=float)
    archive_solutions, archive_objectives = update_archive(
        archive_solutions,
        archive_objectives,
        population,
        objectives,
        archive_size,
    )
    history = [best_sum_history_value(archive_objectives)]

    for _ in range(max_iter):
        for index in np.random.permutation(population_size):
            parent_indexes = _select_indexes(neighbors[index], population_size, 3)
            trial = _create_trial(
                population,
                index,
                parent_indexes,
                bounds,
                mutation_factor,
                crossover_rate,
            )
            trial_objective = objective_function(trial)
            ideal_point = np.minimum(ideal_point, trial_objective)

            for neighbor_index in neighbors[index]:
                old_value = _tchebycheff(objectives[neighbor_index], weights[neighbor_index], ideal_point)
                new_value = _tchebycheff(trial_objective, weights[neighbor_index], ideal_point)
                if new_value <= old_value:
                    population[neighbor_index] = trial
                    objectives[neighbor_index] = trial_objective

        archive_solutions, archive_objectives = update_archive(
            archive_solutions,
            archive_objectives,
            population,
            objectives,
            archive_size,
        )
        history.append(best_sum_history_value(archive_objectives))

    return archive_solutions, archive_objectives, history, used_seed
