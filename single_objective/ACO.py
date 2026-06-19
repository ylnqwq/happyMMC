# -*- coding: utf-8 -*-

import numpy as np


def _validate_bounds(bounds):
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must be shaped like [(lower, upper), ...].")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("each lower bound must be smaller than the upper bound.")
    return bounds


def _build_candidate_values(bounds, levels):
    level_positions = np.linspace(0.0, 1.0, levels)
    return bounds[:, 0, None] + level_positions[None, :] * (bounds[:, 1, None] - bounds[:, 0, None])


def _construct_solution(pheromone, candidate_values):
    dimension, levels = pheromone.shape
    level_indexes = np.empty(dimension, dtype=int)
    solution = np.empty(dimension, dtype=float)

    for dimension_index in range(dimension):
        probabilities = pheromone[dimension_index] / np.sum(pheromone[dimension_index])
        level_index = int(np.random.choice(levels, p=probabilities))
        level_indexes[dimension_index] = level_index
        solution[dimension_index] = candidate_values[dimension_index, level_index]

    return solution, level_indexes


def _deposit_amount(value, iteration_best_value):
    baseline = max(abs(iteration_best_value), 1.0)
    shifted = value - iteration_best_value
    return 1.0 / (1.0 + shifted / baseline)


def ant_colony_optimization(
    objective_function,
    bounds,
    ant=75,
    max_iter=750,
    levels=50,
    evaporation_rate=0.2,
    deposit_weight=1.0,
    seed=None,
):
    """Basic ant colony optimization with discrete pheromone tables."""
    if seed is None:
        used_seed = int(np.random.SeedSequence().generate_state(1)[0])
    else:
        used_seed = int(seed)
    np.random.seed(used_seed)

    bounds = _validate_bounds(bounds)
    dimension = len(bounds)
    ant = max(1, int(ant))
    levels = max(2, int(levels))

    candidate_values = _build_candidate_values(bounds, levels)
    pheromone = np.ones((dimension, levels), dtype=float)

    best_solution = None
    best_level_indexes = None
    best_value = np.inf
    history = []

    for _ in range(max_iter + 1):
        solutions = []
        level_indexes_list = []
        values = []

        for _ in range(ant):
            solution, level_indexes = _construct_solution(pheromone, candidate_values)
            value = float(objective_function(solution))
            solutions.append(solution)
            level_indexes_list.append(level_indexes)
            values.append(value)

        values = np.asarray(values, dtype=float)
        iteration_best_index = int(np.argmin(values))
        iteration_best_value = float(values[iteration_best_index])

        if iteration_best_value < best_value:
            best_value = iteration_best_value
            best_solution = solutions[iteration_best_index].copy()
            best_level_indexes = level_indexes_list[iteration_best_index].copy()

        history.append(best_value)

        if len(history) > max_iter:
            break

        pheromone *= 1.0 - evaporation_rate
        for value, level_indexes in zip(values, level_indexes_list):
            amount = deposit_weight * _deposit_amount(value, iteration_best_value)
            for dimension_index, level_index in enumerate(level_indexes):
                pheromone[dimension_index, level_index] += amount

        if best_level_indexes is not None:
            for dimension_index, level_index in enumerate(best_level_indexes):
                pheromone[dimension_index, level_index] += deposit_weight

        pheromone = np.maximum(pheromone, 1.0e-12)

    return best_solution, best_value, history, used_seed
