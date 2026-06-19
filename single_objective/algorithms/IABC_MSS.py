# -*- coding: utf-8 -*-

import numpy as np


def calculate_fitness(values):
    values = np.asarray(values, dtype=float)
    fitness = np.empty_like(values, dtype=float)

    non_negative = values >= 0
    fitness[non_negative] = 1.0 / (1.0 + values[non_negative])
    fitness[~non_negative] = 1.0 + np.abs(values[~non_negative])
    return fitness


def _validate_bounds(bounds):
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must be shaped like [(lower, upper), ...].")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("each lower bound must be smaller than the upper bound.")
    return bounds


def _tent_map(size, iterations=3):
    values = np.random.rand(*size)
    for _ in range(iterations):
        values = np.where(values < 0.5, 2.0 * values, 2.0 * (1.0 - values))
    return values


def initialize_food_sources(food_number, bounds, objective_function):
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]
    chaotic_points = _tent_map((food_number, len(bounds)))
    food_sources = lower_bounds + chaotic_points * (upper_bounds - lower_bounds)
    values = np.array([objective_function(source) for source in food_sources], dtype=float)
    trials = np.zeros(food_number, dtype=int)
    return food_sources, values, trials


def _select_partner(food_number, index):
    partner_index = np.random.randint(food_number)
    while partner_index == index:
        partner_index = np.random.randint(food_number)
    return partner_index


def create_multi_strategy_neighbor(food_sources, values, index, best_solution, bounds, iteration, max_iter):
    food_number, dimension = food_sources.shape
    neighbor = food_sources[index].copy()
    parameter_index = np.random.randint(dimension)
    partner_index = _select_partner(food_number, index)

    phi = np.random.uniform(-1.0, 1.0)
    progress = (iteration + 1) / max(1, max_iter)
    strategy_probability = np.random.rand()

    if strategy_probability < 0.5:
        neighbor[parameter_index] = (
            food_sources[index, parameter_index]
            + phi * (food_sources[index, parameter_index] - food_sources[partner_index, parameter_index])
        )
    elif strategy_probability < 0.85:
        guidance = np.random.rand() * progress
        neighbor[parameter_index] = (
            food_sources[index, parameter_index]
            + phi * (food_sources[index, parameter_index] - food_sources[partner_index, parameter_index])
            + guidance * (best_solution[parameter_index] - food_sources[index, parameter_index])
        )
    else:
        sorted_indexes = np.argsort(values)
        elite_index = sorted_indexes[np.random.randint(max(1, food_number // 5))]
        tangent = np.tan(np.pi * (np.random.rand() - 0.5))
        scale = 0.02 * (1.0 - progress) * (bounds[parameter_index, 1] - bounds[parameter_index, 0])
        neighbor[parameter_index] = food_sources[elite_index, parameter_index] + scale * tangent

    return np.clip(neighbor, bounds[:, 0], bounds[:, 1])


def greedy_select(food_sources, values, trials, index, candidate, candidate_value):
    if candidate_value < values[index]:
        food_sources[index] = candidate
        values[index] = candidate_value
        trials[index] = 0
    else:
        trials[index] += 1


def update_best(food_sources, values, best_solution, best_value):
    best_index = int(np.argmin(values))
    if values[best_index] < best_value:
        return food_sources[best_index].copy(), float(values[best_index])
    return best_solution, best_value


def employed_bee_phase(
    food_sources,
    values,
    trials,
    bounds,
    objective_function,
    best_solution,
    iteration,
    max_iter,
):
    for index in range(len(food_sources)):
        candidate = create_multi_strategy_neighbor(
            food_sources,
            values,
            index,
            best_solution,
            bounds,
            iteration,
            max_iter,
        )
        candidate_value = objective_function(candidate)
        greedy_select(food_sources, values, trials, index, candidate, candidate_value)


def onlooker_bee_phase(
    food_sources,
    values,
    trials,
    bounds,
    objective_function,
    best_solution,
    iteration,
    max_iter,
):
    fitness = calculate_fitness(values)
    probabilities = fitness / fitness.sum()

    food_number = len(food_sources)
    selected_count = 0
    index = 0
    while selected_count < food_number:
        if np.random.rand() < probabilities[index]:
            candidate = create_multi_strategy_neighbor(
                food_sources,
                values,
                index,
                best_solution,
                bounds,
                iteration,
                max_iter,
            )
            candidate_value = objective_function(candidate)
            greedy_select(food_sources, values, trials, index, candidate, candidate_value)
            selected_count += 1

        index = (index + 1) % food_number


def scout_bee_phase(food_sources, values, trials, bounds, objective_function, limit):
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]
    for index in range(len(food_sources)):
        if trials[index] >= limit:
            food_sources[index] = np.random.uniform(lower_bounds, upper_bounds)
            values[index] = objective_function(food_sources[index])
            trials[index] = 0


def iabc_mss(
    objective_function,
    bounds,
    bee=30,
    max_iter=500,
    limit=100,
    seed=None,
):
    """Multi-strategy synthesis improved ABC for continuous minimization."""
    if seed is None:
        used_seed = int(np.random.SeedSequence().generate_state(1)[0])
    else:
        used_seed = int(seed)
    np.random.seed(used_seed)

    bounds = _validate_bounds(bounds)
    food_sources, values, trials = initialize_food_sources(bee, bounds, objective_function)

    best_index = int(np.argmin(values))
    best_solution = food_sources[best_index].copy()
    best_value = float(values[best_index])
    history = [best_value]

    for iteration in range(max_iter):
        employed_bee_phase(
            food_sources,
            values,
            trials,
            bounds,
            objective_function,
            best_solution,
            iteration,
            max_iter,
        )
        best_solution, best_value = update_best(food_sources, values, best_solution, best_value)

        onlooker_bee_phase(
            food_sources,
            values,
            trials,
            bounds,
            objective_function,
            best_solution,
            iteration,
            max_iter,
        )
        best_solution, best_value = update_best(food_sources, values, best_solution, best_value)

        scout_bee_phase(food_sources, values, trials, bounds, objective_function, limit)
        best_solution, best_value = update_best(food_sources, values, best_solution, best_value)
        history.append(best_value)

    return best_solution, best_value, history, used_seed
