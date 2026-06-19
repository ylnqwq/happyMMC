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


def initialize_food_sources(food_number, bounds, objective_function):
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]

    food_sources = np.random.uniform(lower_bounds, upper_bounds, size=(food_number, len(bounds)))
    values = np.array([objective_function(source) for source in food_sources], dtype=float)
    trials = np.zeros(food_number, dtype=int)
    return food_sources, values, trials


def create_gbest_neighbor(food_sources, index, best_solution, bounds, guidance_rate):
    food_number, dimension = food_sources.shape
    neighbor = food_sources[index].copy()

    parameter_index = np.random.randint(dimension)
    partner_index = np.random.randint(food_number)
    while partner_index == index:
        partner_index = np.random.randint(food_number)

    phi = np.random.uniform(-1.0, 1.0)
    psi = np.random.uniform(0.0, guidance_rate)
    neighbor[parameter_index] = (
        food_sources[index, parameter_index]
        + phi * (food_sources[index, parameter_index] - food_sources[partner_index, parameter_index])
        + psi * (best_solution[parameter_index] - food_sources[index, parameter_index])
    )

    return np.clip(neighbor, bounds[:, 0], bounds[:, 1])


def greedy_select(food_sources, values, trials, index, candidate, candidate_value):
    if candidate_value < values[index]:
        food_sources[index] = candidate
        values[index] = candidate_value
        trials[index] = 0
    else:
        trials[index] += 1


def update_best(food_sources, values, best_solution, best_value):
    current_best_index = int(np.argmin(values))
    if values[current_best_index] < best_value:
        return food_sources[current_best_index].copy(), float(values[current_best_index])
    return best_solution, best_value


def employed_bee_phase(food_sources, values, trials, bounds, objective_function, best_solution, guidance_rate):
    for index in range(len(food_sources)):
        candidate = create_gbest_neighbor(food_sources, index, best_solution, bounds, guidance_rate)
        candidate_value = objective_function(candidate)
        greedy_select(food_sources, values, trials, index, candidate, candidate_value)


def onlooker_bee_phase(food_sources, values, trials, bounds, objective_function, best_solution, guidance_rate):
    fitness = calculate_fitness(values)
    probabilities = fitness / fitness.sum()

    food_number = len(food_sources)
    selected_count = 0
    index = 0
    while selected_count < food_number:
        if np.random.rand() < probabilities[index]:
            candidate = create_gbest_neighbor(food_sources, index, best_solution, bounds, guidance_rate)
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


def gbest_guided_abc(
    objective_function,
    bounds,
    bee=30,
    max_iter=500,
    limit=100,
    guidance_rate=1.5,
    seed=None,
):
    """Gbest-guided artificial bee colony algorithm for continuous minimization."""
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

    for _ in range(max_iter):
        employed_bee_phase(
            food_sources,
            values,
            trials,
            bounds,
            objective_function,
            best_solution,
            guidance_rate,
        )
        best_solution, best_value = update_best(food_sources, values, best_solution, best_value)

        onlooker_bee_phase(
            food_sources,
            values,
            trials,
            bounds,
            objective_function,
            best_solution,
            guidance_rate,
        )
        best_solution, best_value = update_best(food_sources, values, best_solution, best_value)

        scout_bee_phase(food_sources, values, trials, bounds, objective_function, limit)
        best_solution, best_value = update_best(food_sources, values, best_solution, best_value)

        history.append(best_value)

    return best_solution, best_value, history, used_seed
