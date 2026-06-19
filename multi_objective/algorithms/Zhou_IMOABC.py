# -*- coding: utf-8 -*-

import numpy as np

from multi_objective.mo_utils import (
    best_sum_history_value,
    crowding_distance,
    evaluate_objectives,
    greedy_select_multi,
    population_scores,
    update_archive,
    validate_bounds,
)


def _logistic_map(size, iterations=5):
    values = np.random.rand(*size)
    for _ in range(iterations):
        values = 4.0 * values * (1.0 - values)
    return values


def initialize_food_sources(food_number, bounds, objective_function):
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]
    points = _logistic_map((food_number, len(bounds)))
    food_sources = lower_bounds + points * (upper_bounds - lower_bounds)
    objectives = evaluate_objectives(objective_function, food_sources)
    trials = np.zeros(food_number, dtype=int)
    return food_sources, objectives, trials


def _select_partner(food_number, index):
    partner_index = np.random.randint(food_number)
    while partner_index == index:
        partner_index = np.random.randint(food_number)
    return partner_index


def _select_archive_guide(archive_solutions, archive_objectives):
    if archive_solutions is None or len(archive_solutions) == 0:
        return None
    distances = crowding_distance(archive_objectives)
    finite = np.where(np.isfinite(distances), distances, np.max(distances[np.isfinite(distances)]) if np.any(np.isfinite(distances)) else 1.0)
    probabilities = finite + 1.0e-12
    probabilities = probabilities / np.sum(probabilities)
    return archive_solutions[int(np.random.choice(len(archive_solutions), p=probabilities))]


def create_neighbor(food_sources, index, bounds, archive_solutions, archive_objectives, iteration, max_iter):
    food_number, dimension = food_sources.shape
    neighbor = food_sources[index].copy()
    parameter_index = np.random.randint(dimension)
    partner_index = _select_partner(food_number, index)

    progress = (iteration + 1) / max(1, max_iter)
    phi = np.random.uniform(-1.0, 1.0) * (1.0 - 0.5 * progress)
    neighbor[parameter_index] = (
        food_sources[index, parameter_index]
        + phi * (food_sources[index, parameter_index] - food_sources[partner_index, parameter_index])
    )

    guide = _select_archive_guide(archive_solutions, archive_objectives)
    if guide is not None:
        beta = np.random.uniform(0.0, 0.6 * progress)
        neighbor[parameter_index] += beta * (guide[parameter_index] - food_sources[index, parameter_index])

    if np.random.rand() < 0.15:
        span = bounds[parameter_index, 1] - bounds[parameter_index, 0]
        neighbor[parameter_index] += np.random.normal(0.0, 0.03 * (1.0 - progress) * span)

    return np.clip(neighbor, bounds[:, 0], bounds[:, 1])


def employed_bee_phase(food_sources, objectives, trials, bounds, objective_function, archive_solutions, archive_objectives, iteration, max_iter):
    for index in range(len(food_sources)):
        candidate = create_neighbor(food_sources, index, bounds, archive_solutions, archive_objectives, iteration, max_iter)
        candidate_objective = objective_function(candidate)
        greedy_select_multi(food_sources, objectives, trials, index, candidate, candidate_objective)


def onlooker_bee_phase(food_sources, objectives, trials, bounds, objective_function, archive_solutions, archive_objectives, iteration, max_iter):
    ranks, distances = population_scores(objectives)
    distance_scores = np.where(np.isfinite(distances), distances, np.max(distances[np.isfinite(distances)]) if np.any(np.isfinite(distances)) else 1.0)
    scores = 1.0 / (1.0 + ranks) + 0.1 * distance_scores
    probabilities = scores / np.sum(scores)

    food_number = len(food_sources)
    selected_count = 0
    index = 0
    while selected_count < food_number:
        if np.random.rand() < probabilities[index]:
            candidate = create_neighbor(food_sources, index, bounds, archive_solutions, archive_objectives, iteration, max_iter)
            candidate_objective = objective_function(candidate)
            greedy_select_multi(food_sources, objectives, trials, index, candidate, candidate_objective)
            selected_count += 1
        index = (index + 1) % food_number


def scout_bee_phase(food_sources, objectives, trials, bounds, objective_function, limit):
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]
    for index in range(len(food_sources)):
        if trials[index] >= limit:
            chaotic = _logistic_map((len(bounds),))
            food_sources[index] = lower_bounds + chaotic * (upper_bounds - lower_bounds)
            objectives[index] = objective_function(food_sources[index])
            trials[index] = 0


def zhou_imoabc(
    objective_function,
    bounds,
    bee=30,
    max_iter=500,
    limit=100,
    archive_size=100,
    seed=None,
):
    if seed is None:
        used_seed = int(np.random.SeedSequence().generate_state(1)[0])
    else:
        used_seed = int(seed)
    np.random.seed(used_seed)

    bounds = validate_bounds(bounds)
    food_sources, objectives, trials = initialize_food_sources(bee, bounds, objective_function)
    archive_solutions = np.empty((0, len(bounds)), dtype=float)
    archive_objectives = np.empty((0, objectives.shape[1]), dtype=float)
    archive_solutions, archive_objectives = update_archive(
        archive_solutions, archive_objectives, food_sources, objectives, archive_size
    )
    history = [best_sum_history_value(archive_objectives)]

    for iteration in range(max_iter):
        employed_bee_phase(food_sources, objectives, trials, bounds, objective_function, archive_solutions, archive_objectives, iteration, max_iter)
        onlooker_bee_phase(food_sources, objectives, trials, bounds, objective_function, archive_solutions, archive_objectives, iteration, max_iter)
        scout_bee_phase(food_sources, objectives, trials, bounds, objective_function, limit)
        archive_solutions, archive_objectives = update_archive(
            archive_solutions, archive_objectives, food_sources, objectives, archive_size
        )
        history.append(best_sum_history_value(archive_objectives))

    return archive_solutions, archive_objectives, history, used_seed

