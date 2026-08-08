# -*- coding: utf-8 -*-

import math

import numpy as np

from multi_objective.mo_utils import (
    best_sum_history_value,
    crowding_distance,
    dominates,
    evaluate_objectives,
    population_scores,
    update_archive,
    validate_bounds,
)


def _circle_map(size, iterations=4, c=0.2, d=0.5):
    values = np.random.rand(*size)
    for _ in range(iterations):
        values = np.mod(values + c - d / (2.0 * np.pi) * np.sin(2.0 * np.pi * values), 1.0)
    return values


def _select_population(solutions, objectives, population_size):
    ranks, distances = population_scores(objectives)
    finite_distances = np.where(np.isfinite(distances), distances, np.finfo(float).max)
    order = np.lexsort((-finite_distances, ranks))
    selected = order[:population_size]
    return solutions[selected], objectives[selected]


def _initialize_population(population_size, bounds, objective_function):
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]
    chaotic = _circle_map((population_size, len(bounds)))
    candidates = lower_bounds + chaotic * (upper_bounds - lower_bounds)
    opposite = lower_bounds + upper_bounds - candidates
    opposite = np.clip(opposite, lower_bounds, upper_bounds)

    all_solutions = np.vstack([candidates, opposite])
    all_objectives = evaluate_objectives(objective_function, all_solutions)
    return _select_population(all_solutions, all_objectives, population_size)


def _scalar_fitness(objectives):
    objectives = np.asarray(objectives, dtype=float)
    ideal = np.min(objectives, axis=0)
    nadir = np.max(objectives, axis=0)
    span = np.where(np.isclose(nadir - ideal, 0.0), 1.0, nadir - ideal)
    normalized = (objectives - ideal) / span
    ranks, _ = population_scores(objectives)
    return ranks.astype(float) + np.sum(normalized, axis=1)


def _select_archive_best(archive_solutions, archive_objectives):
    index = int(np.argmin(np.sum(archive_objectives, axis=1)))
    return archive_solutions[index], archive_objectives[index]


def _select_archive_guide(archive_solutions, archive_objectives):
    distances = crowding_distance(archive_objectives)
    if len(distances) == 0:
        return _select_archive_best(archive_solutions, archive_objectives)[0]
    finite_mask = np.isfinite(distances)
    scores = np.ones(len(distances), dtype=float)
    if np.any(finite_mask):
        finite_max = np.max(distances[finite_mask])
        scores[finite_mask] = distances[finite_mask] + 1.0e-12
        scores[~finite_mask] = finite_max + 1.0
    probabilities = scores / np.sum(scores)
    return archive_solutions[int(np.random.choice(len(archive_solutions), p=probabilities))]


def _levy_step(dimension, beta=1.5):
    numerator = math.gamma(1.0 + beta) * np.sin(np.pi * beta / 2.0)
    denominator = math.gamma((1.0 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0)
    sigma_u = (numerator / denominator) ** (1.0 / beta)
    mu = np.random.normal(0.0, sigma_u, size=dimension)
    nu = np.random.normal(0.0, 1.0, size=dimension)
    return mu / (np.abs(nu) ** (1.0 / beta) + 1.0e-12)


def _golden_sine_coefficients(iteration, max_iter):
    tau = (np.sqrt(5.0) - 1.0) / 2.0
    progress = (iteration + 1.0) / max(1.0, max_iter)
    m = np.pi * (1.0 - progress)
    u = -np.pi * progress
    c1 = m * tau + u * (1.0 - tau)
    c2 = m * (1.0 - tau) + u * tau
    return c1, c2


def _discoverer_update(position, best_position, iteration, max_iter, safety_threshold):
    dimension = len(position)
    random_alarm = np.random.rand()
    if random_alarm < safety_threshold:
        r1 = np.random.uniform(0.0, 2.0 * np.pi, size=dimension)
        r2 = np.random.uniform(0.0, np.pi, size=dimension)
        c1, c2 = _golden_sine_coefficients(iteration, max_iter)
        return np.abs(position * np.sin(r1)) + r2 * np.sin(r1) * np.abs(c1 * best_position - c2 * position)

    return position + np.random.normal(0.0, 1.0, size=dimension)


def _follower_weight(iteration, max_iter):
    progress = (iteration + 1.0) / max(1.0, max_iter)
    value = 1.0 - (np.exp(progress) - 1.0) / (np.e - 1.0)
    return np.sin(np.pi * value / 2.0)


def _follower_update(position, guide_position, index, population_size, iteration, max_iter):
    if index > population_size / 2:
        weight = _follower_weight(iteration, max_iter)
        return position + weight * np.abs(position - guide_position)
    return guide_position + _levy_step(len(position)) * np.abs(position - guide_position)


def _warner_update(position, best_position, worst_position, fitness, best_fitness, worst_fitness):
    if fitness > best_fitness:
        gamma = np.random.normal(0.0, 1.0, size=len(position))
        return best_position + gamma * np.abs(position - best_position)

    theta = np.random.uniform(-1.0, 1.0, size=len(position))
    denominator = fitness - worst_fitness + 1.0e-12
    return position + theta * np.abs((position - worst_position) / denominator)


def _distribution_step(distribution, dimension):
    if distribution == "cauchy":
        return np.random.standard_cauchy(size=dimension)
    if distribution == "student":
        return np.random.standard_t(df=3, size=dimension)
    return np.random.normal(0.0, 1.0, size=dimension)


def _mixed_disturbance(position, best_position, iteration, max_iter):
    progress = (iteration + 1.0) / max(1.0, max_iter)
    omega = math.exp(1.0 - progress) / math.e
    if progress < 1.0 / 3.0:
        distribution = "cauchy"
    elif progress < 2.0 / 3.0:
        distribution = "student"
    else:
        distribution = "gaussian"
    return best_position + omega * _distribution_step(distribution, len(position)) * (best_position - position)


def _select_survivor(current_solution, current_objective, candidate_solution, candidate_objective):
    if dominates(candidate_objective, current_objective):
        return candidate_solution, candidate_objective
    if dominates(current_objective, candidate_objective):
        return current_solution, current_objective
    if np.sum(candidate_objective) <= np.sum(current_objective):
        return candidate_solution, candidate_objective
    return current_solution, current_objective


def issa(
    objective_function,
    bounds,
    bee=80,
    population_size=None,
    max_iter=800,
    limit=None,
    archive_size=100,
    discoverer_rate=0.2,
    warner_rate=0.15,
    safety_threshold=0.8,
    disturbance_rate=0.35,
    seed=None,
):
    if seed is None:
        used_seed = int(np.random.SeedSequence().generate_state(1)[0])
    else:
        used_seed = int(seed)
    np.random.seed(used_seed)

    bounds = validate_bounds(bounds)
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]
    dimension = len(bounds)
    population_size = max(6, int(bee if population_size is None else population_size))

    sparrows, objectives = _initialize_population(population_size, bounds, objective_function)
    archive_solutions = np.empty((0, dimension), dtype=float)
    archive_objectives = np.empty((0, objectives.shape[1]), dtype=float)
    archive_solutions, archive_objectives = update_archive(
        archive_solutions,
        archive_objectives,
        sparrows,
        objectives,
        archive_size,
    )
    history = [best_sum_history_value(archive_objectives)]

    discoverer_count = max(1, int(np.ceil(population_size * discoverer_rate)))
    warner_count = max(1, int(np.ceil(population_size * warner_rate)))

    for iteration in range(max_iter):
        fitness = _scalar_fitness(objectives)
        order = np.argsort(fitness)
        best_index = int(order[0])
        worst_index = int(order[-1])
        best_position, _ = _select_archive_best(archive_solutions, archive_objectives)
        worst_position = sparrows[worst_index]
        guide_position = _select_archive_guide(archive_solutions, archive_objectives)
        best_fitness = float(fitness[best_index])
        worst_fitness = float(fitness[worst_index])
        discoverers = set(int(index) for index in order[:discoverer_count])
        warners = set(int(index) for index in np.random.choice(population_size, size=warner_count, replace=False))

        next_sparrows = sparrows.copy()
        next_objectives = objectives.copy()
        for index in range(population_size):
            if index in discoverers:
                candidate = _discoverer_update(
                    sparrows[index],
                    best_position,
                    iteration,
                    max_iter,
                    safety_threshold,
                )
            else:
                candidate = _follower_update(
                    sparrows[index],
                    guide_position,
                    index,
                    population_size,
                    iteration,
                    max_iter,
                )

            if index in warners:
                candidate = _warner_update(
                    candidate,
                    best_position,
                    worst_position,
                    float(fitness[index]),
                    best_fitness,
                    worst_fitness,
                )

            candidate = np.clip(candidate, lower_bounds, upper_bounds)
            candidate_objective = objective_function(candidate)
            selected, selected_objective = _select_survivor(
                sparrows[index],
                objectives[index],
                candidate,
                candidate_objective,
            )

            if np.random.rand() < disturbance_rate:
                disturbed = _mixed_disturbance(selected, best_position, iteration, max_iter)
                disturbed = np.clip(disturbed, lower_bounds, upper_bounds)
                disturbed_objective = objective_function(disturbed)
                selected, selected_objective = _select_survivor(
                    selected,
                    selected_objective,
                    disturbed,
                    disturbed_objective,
                )

            next_sparrows[index] = selected
            next_objectives[index] = selected_objective

        sparrows = next_sparrows
        objectives = next_objectives
        archive_solutions, archive_objectives = update_archive(
            archive_solutions,
            archive_objectives,
            sparrows,
            objectives,
            archive_size,
        )
        history.append(best_sum_history_value(archive_objectives))

    return archive_solutions, archive_objectives, history, used_seed
