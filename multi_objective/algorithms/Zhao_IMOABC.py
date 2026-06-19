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


def _good_point_set(food_number, dimension):
    prime = 2 * dimension + 3
    while any(prime % factor == 0 for factor in range(2, int(np.sqrt(prime)) + 1)):
        prime += 1
    r = 2.0 * np.cos(2.0 * np.pi * np.arange(1, dimension + 1) / prime)
    indexes = np.arange(1, food_number + 1).reshape(-1, 1)
    return np.mod(indexes * r, 1.0)


def initialize_food_sources(food_number, bounds, objective_function):
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]
    points = _good_point_set(food_number, len(bounds))
    food_sources = lower_bounds + points * (upper_bounds - lower_bounds)
    objectives = evaluate_objectives(objective_function, food_sources)
    trials = np.zeros(food_number, dtype=int)
    return food_sources, objectives, trials


def _select_partner(food_number, index):
    partner_index = np.random.randint(food_number)
    while partner_index == index:
        partner_index = np.random.randint(food_number)
    return partner_index


def _tournament_select(objectives, tournament_size=3):
    ranks, distances = population_scores(objectives)
    candidates = np.random.choice(len(objectives), size=min(tournament_size, len(objectives)), replace=False)
    best = candidates[0]
    for candidate in candidates[1:]:
        if ranks[candidate] < ranks[best]:
            best = candidate
        elif ranks[candidate] == ranks[best] and distances[candidate] > distances[best]:
            best = candidate
    return best


def create_neighbor(food_sources, objectives, index, bounds, archive_solutions, iteration, max_iter):
    food_number, dimension = food_sources.shape
    neighbor = food_sources[index].copy()
    parameter_index = np.random.randint(dimension)
    partner_index = _select_partner(food_number, index)
    elite_index = _tournament_select(objectives)

    progress = (iteration + 1) / max(1, max_iter)
    phi = np.random.uniform(-1.0, 1.0)
    elite_weight = np.random.uniform(0.0, 0.5)
    neighbor[parameter_index] = (
        food_sources[index, parameter_index]
        + phi * (food_sources[index, parameter_index] - food_sources[partner_index, parameter_index])
        + elite_weight * (food_sources[elite_index, parameter_index] - food_sources[index, parameter_index])
    )

    if archive_solutions is not None and len(archive_solutions) > 0:
        guide = archive_solutions[np.random.randint(len(archive_solutions))]
        archive_weight = np.random.uniform(0.0, 0.4 * progress)
        neighbor[parameter_index] += archive_weight * (guide[parameter_index] - food_sources[index, parameter_index])

    return np.clip(neighbor, bounds[:, 0], bounds[:, 1])


def employed_bee_phase(food_sources, objectives, trials, bounds, objective_function, archive_solutions, iteration, max_iter):
    for index in range(len(food_sources)):
        candidate = create_neighbor(food_sources, objectives, index, bounds, archive_solutions, iteration, max_iter)
        candidate_objective = objective_function(candidate)
        greedy_select_multi(food_sources, objectives, trials, index, candidate, candidate_objective)


def onlooker_bee_phase(food_sources, objectives, trials, bounds, objective_function, archive_solutions, iteration, max_iter):
    for _ in range(len(food_sources)):
        index = _tournament_select(objectives)
        candidate = create_neighbor(food_sources, objectives, index, bounds, archive_solutions, iteration, max_iter)
        candidate_objective = objective_function(candidate)
        greedy_select_multi(food_sources, objectives, trials, index, candidate, candidate_objective)


def scout_and_elimination_phase(food_sources, objectives, trials, bounds, objective_function, limit, elimination_rate):
    ranks, distances = population_scores(objectives)
    finite_distances = np.where(np.isfinite(distances), distances, np.finfo(float).max)
    elimination_number = max(1, int(np.ceil(len(food_sources) * elimination_rate)))
    worst_indexes = np.lexsort((finite_distances, -ranks))[:elimination_number]
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]

    for index in range(len(food_sources)):
        if trials[index] >= limit or index in worst_indexes:
            food_sources[index] = np.random.uniform(lower_bounds, upper_bounds)
            objectives[index] = objective_function(food_sources[index])
            trials[index] = 0


def zhao_imoabc(
    objective_function,
    bounds,
    bee=30,
    max_iter=500,
    limit=100,
    archive_size=100,
    elimination_rate=0.1,
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
        employed_bee_phase(food_sources, objectives, trials, bounds, objective_function, archive_solutions, iteration, max_iter)
        onlooker_bee_phase(food_sources, objectives, trials, bounds, objective_function, archive_solutions, iteration, max_iter)
        scout_and_elimination_phase(food_sources, objectives, trials, bounds, objective_function, limit, elimination_rate)
        archive_solutions, archive_objectives = update_archive(
            archive_solutions, archive_objectives, food_sources, objectives, archive_size
        )
        history.append(best_sum_history_value(archive_objectives))

    return archive_solutions, archive_objectives, history, used_seed

