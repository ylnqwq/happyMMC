# -*- coding: utf-8 -*-

import numpy as np

from multi_objective.mo_utils import (
    best_sum_history_value,
    evaluate_objectives,
    greedy_select_multi,
    population_scores,
    update_archive,
    validate_bounds,
)


def initialize_food_sources(food_number, bounds, objective_function):
    bounds = validate_bounds(bounds)
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]

    food_sources = np.random.uniform(lower_bounds, upper_bounds, size=(food_number, len(bounds)))
    objectives = evaluate_objectives(objective_function, food_sources)
    trials = np.zeros(food_number, dtype=int)
    return food_sources, objectives, trials


def create_neighbor(food_sources, index, bounds):
    food_number, dimension = food_sources.shape
    neighbor = food_sources[index].copy()

    parameter_index = np.random.randint(dimension)
    partner_index = np.random.randint(food_number)
    while partner_index == index:
        partner_index = np.random.randint(food_number)

    phi = np.random.uniform(-1.0, 1.0)
    neighbor[parameter_index] = (
        food_sources[index, parameter_index]
        + phi * (food_sources[index, parameter_index] - food_sources[partner_index, parameter_index])
    )

    bounds = np.asarray(bounds, dtype=float)
    return np.clip(neighbor, bounds[:, 0], bounds[:, 1])


def employed_bee_phase(food_sources, objectives, trials, bounds, objective_function):
    for i in range(len(food_sources)):
        candidate = create_neighbor(food_sources, i, bounds)
        candidate_objective = objective_function(candidate)
        greedy_select_multi(food_sources, objectives, trials, i, candidate, candidate_objective)


def calculate_selection_probabilities(objectives):
    ranks, distances = population_scores(objectives)
    distance_scores = np.where(np.isfinite(distances), distances, np.max(distances[np.isfinite(distances)]) if np.any(np.isfinite(distances)) else 1.0)
    scores = 1.0 / (1.0 + ranks) + 0.05 * distance_scores
    return scores / np.sum(scores)


def onlooker_bee_phase(food_sources, objectives, trials, bounds, objective_function):
    probabilities = calculate_selection_probabilities(objectives)

    food_number = len(food_sources)
    selected_count = 0
    i = 0
    while selected_count < food_number:
        if np.random.rand() < probabilities[i]:
            candidate = create_neighbor(food_sources, i, bounds)
            candidate_objective = objective_function(candidate)
            greedy_select_multi(food_sources, objectives, trials, i, candidate, candidate_objective)
            selected_count += 1

        i = (i + 1) % food_number


def scout_bee_phase(food_sources, objectives, trials, bounds, objective_function, limit):
    bounds = np.asarray(bounds, dtype=float)
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]

    for i in range(len(food_sources)):
        if trials[i] >= limit:
            food_sources[i] = np.random.uniform(lower_bounds, upper_bounds)
            objectives[i] = objective_function(food_sources[i])
            trials[i] = 0


def multi_objective_abc(
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
        archive_solutions,
        archive_objectives,
        food_sources,
        objectives,
        archive_size,
    )

    history = [best_sum_history_value(archive_objectives)]

    for _ in range(max_iter):
        employed_bee_phase(food_sources, objectives, trials, bounds, objective_function)
        onlooker_bee_phase(food_sources, objectives, trials, bounds, objective_function)
        scout_bee_phase(food_sources, objectives, trials, bounds, objective_function, limit)

        archive_solutions, archive_objectives = update_archive(
            archive_solutions,
            archive_objectives,
            food_sources,
            objectives,
            archive_size,
        )
        history.append(best_sum_history_value(archive_objectives))

    return archive_solutions, archive_objectives, history, used_seed

