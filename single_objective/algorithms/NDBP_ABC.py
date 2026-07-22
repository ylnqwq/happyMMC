# -*- coding: utf-8 -*-

import numpy as np

from single_objective.so_utils import (
    calculate_fitness,
    initialize_random_sources,
    reinitialize_source,
    select_partner,
    validate_bounds,
)


def initialize_food_sources(food_number, bounds, objective_function):
    return initialize_random_sources(food_number, bounds, objective_function)


def create_ndbp_neighbor(food_sources, values, index, best_solution, worst_solution, bounds, iteration, max_iter):
    food_number, dimension = food_sources.shape
    neighbor = food_sources[index].copy()
    parameter_index = np.random.randint(dimension)
    partner_index = select_partner(food_number, index)

    progress = (iteration + 1) / max(1, max_iter)
    phi = np.random.uniform(-1.0, 1.0)
    forward_weight = np.random.rand() * progress
    backward_weight = np.random.rand() * (1.0 - progress)

    if np.random.rand() < 0.5:
        neighbor[parameter_index] = (
            food_sources[index, parameter_index]
            + phi * (food_sources[index, parameter_index] - food_sources[partner_index, parameter_index])
            + forward_weight * (best_solution[parameter_index] - food_sources[index, parameter_index])
        )
    else:
        neighbor[parameter_index] = (
            food_sources[index, parameter_index]
            + phi * (food_sources[index, parameter_index] - food_sources[partner_index, parameter_index])
            + backward_weight * (food_sources[index, parameter_index] - worst_solution[parameter_index])
        )

    if np.random.rand() < 0.2:
        span = bounds[parameter_index, 1] - bounds[parameter_index, 0]
        step = np.random.standard_cauchy() * 0.01 * (1.0 - 0.5 * progress) * span
        neighbor[parameter_index] += step

    return np.clip(neighbor, bounds[:, 0], bounds[:, 1])


def greedy_select(food_sources, values, trials, index, candidate, candidate_value):
    if candidate_value < values[index]:
        food_sources[index] = candidate
        values[index] = candidate_value
        trials[index] = 0
    else:
        trials[index] += 1


def update_best_worst(food_sources, values):
    best_index = int(np.argmin(values))
    worst_index = int(np.argmax(values))
    return food_sources[best_index].copy(), float(values[best_index]), food_sources[worst_index].copy()


def employed_bee_phase(
    food_sources,
    values,
    trials,
    bounds,
    objective_function,
    best_solution,
    worst_solution,
    iteration,
    max_iter,
):
    for index in range(len(food_sources)):
        candidate = create_ndbp_neighbor(
            food_sources,
            values,
            index,
            best_solution,
            worst_solution,
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
    worst_solution,
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
            candidate = create_ndbp_neighbor(
                food_sources,
                values,
                index,
                best_solution,
                worst_solution,
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
            reinitialize_source(food_sources, values, trials, index, lower_bounds, upper_bounds, objective_function)


def ndbp_abc(
    objective_function,
    bounds,
    bee=80,
    max_iter=800,
    limit=160,
    seed=None,
):
    """Nondeterministic bidirectional-planning ABC for continuous minimization."""
    if seed is None:
        used_seed = int(np.random.SeedSequence().generate_state(1)[0])
    else:
        used_seed = int(seed)
    np.random.seed(used_seed)

    bounds = validate_bounds(bounds)
    food_sources, values, trials = initialize_food_sources(bee, bounds, objective_function)

    best_solution, best_value, worst_solution = update_best_worst(food_sources, values)
    history = [best_value]

    for iteration in range(max_iter):
        employed_bee_phase(
            food_sources,
            values,
            trials,
            bounds,
            objective_function,
            best_solution,
            worst_solution,
            iteration,
            max_iter,
        )
        current_best, current_value, worst_solution = update_best_worst(food_sources, values)
        if current_value < best_value:
            best_solution = current_best
            best_value = current_value

        onlooker_bee_phase(
            food_sources,
            values,
            trials,
            bounds,
            objective_function,
            best_solution,
            worst_solution,
            iteration,
            max_iter,
        )
        current_best, current_value, worst_solution = update_best_worst(food_sources, values)
        if current_value < best_value:
            best_solution = current_best
            best_value = current_value

        scout_bee_phase(food_sources, values, trials, bounds, objective_function, limit)
        current_best, current_value, worst_solution = update_best_worst(food_sources, values)
        if current_value < best_value:
            best_solution = current_best
            best_value = current_value

        history.append(best_value)

    return best_solution, best_value, history, used_seed
