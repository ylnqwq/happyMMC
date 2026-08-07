# -*- coding: utf-8 -*-

import numpy as np

from multi_objective.mo_utils import (
    best_sum_history_value,
    crowding_distance,
    evaluate_objectives,
    update_archive,
    validate_bounds,
)


def _initialize_wolves(population_size, bounds, objective_function):
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]
    wolves = np.random.uniform(lower_bounds, upper_bounds, size=(population_size, len(bounds)))
    objectives = evaluate_objectives(objective_function, wolves)
    return wolves, objectives


def _leader_probabilities(archive_objectives):
    distances = crowding_distance(archive_objectives)
    if len(distances) == 0:
        return np.array([], dtype=float)

    finite_mask = np.isfinite(distances)
    scores = np.ones(len(distances), dtype=float)
    if np.any(finite_mask):
        finite_max = np.max(distances[finite_mask])
        scores[finite_mask] = distances[finite_mask] + 1.0e-12
        scores[~finite_mask] = finite_max + 1.0
    return scores / np.sum(scores)


def _select_leaders(archive_solutions, archive_objectives):
    if len(archive_solutions) == 0:
        raise ValueError("archive_solutions must not be empty when selecting leaders.")

    probabilities = _leader_probabilities(archive_objectives)
    replace = len(archive_solutions) < 3
    indexes = np.random.choice(len(archive_solutions), size=3, replace=replace, p=probabilities)
    return archive_solutions[indexes[0]], archive_solutions[indexes[1]], archive_solutions[indexes[2]]


def _cosine_convergence_factor(iteration, max_iter, initial_value=2.0, final_value=0.0, exponent=1.0):
    if max_iter <= 1:
        return final_value

    t = iteration + 1.0
    cosine_value = np.cos((t - 1.0) * np.pi / (max_iter - 1.0))
    cosine_power = np.abs(cosine_value) ** exponent
    if t < 0.5 * max_iter:
        ratio = (1.0 + cosine_power) / 2.0
    else:
        ratio = (1.0 - cosine_power) / 2.0
    return final_value + (initial_value - final_value) * ratio


def _adaptive_inertia_weight(iteration, max_iter):
    t = float(iteration)
    return np.sin(np.pi * t / (2.0 * max(1, max_iter)) + np.pi) + 1.0


def _update_position(wolf, alpha, beta, delta, convergence_factor, inertia_weight, bounds):
    leaders = (alpha, beta, delta)
    candidates = []
    for leader in leaders:
        r1 = np.random.rand(len(wolf))
        r2 = np.random.rand(len(wolf))
        coefficient_a = 2.0 * convergence_factor * r1 - convergence_factor
        coefficient_c = 2.0 * r2
        distance = np.abs(coefficient_c * leader - wolf)
        candidates.append(inertia_weight * leader - coefficient_a * distance)

    new_position = np.mean(np.vstack(candidates), axis=0)
    return np.clip(new_position, bounds[:, 0], bounds[:, 1])


def _best_sum_solution(archive_solutions, archive_objectives):
    best_index = int(np.argmin(np.sum(archive_objectives, axis=1)))
    return archive_solutions[best_index]


def _cauchy_variation(global_best, bounds):
    random_values = np.random.rand(len(global_best))
    cauchy_step = np.tan(np.pi * (random_values - 0.5))
    mutated = global_best + global_best * cauchy_step
    return np.clip(mutated, bounds[:, 0], bounds[:, 1])


def yang_igwo(
    objective_function,
    bounds,
    bee=80,
    population_size=None,
    max_iter=800,
    limit=None,
    archive_size=100,
    convergence_exponent=1.0,
    seed=None,
):
    if seed is None:
        used_seed = int(np.random.SeedSequence().generate_state(1)[0])
    else:
        used_seed = int(seed)
    np.random.seed(used_seed)

    bounds = validate_bounds(bounds)
    population_size = int(bee if population_size is None else population_size)
    wolves, objectives = _initialize_wolves(population_size, bounds, objective_function)

    archive_solutions = np.empty((0, len(bounds)), dtype=float)
    archive_objectives = np.empty((0, objectives.shape[1]), dtype=float)
    archive_solutions, archive_objectives = update_archive(
        archive_solutions,
        archive_objectives,
        wolves,
        objectives,
        archive_size,
    )
    history = [best_sum_history_value(archive_objectives)]

    for iteration in range(max_iter):
        convergence_factor = _cosine_convergence_factor(
            iteration,
            max_iter,
            exponent=convergence_exponent,
        )
        inertia_weight = _adaptive_inertia_weight(iteration, max_iter)

        next_wolves = np.empty_like(wolves)
        for index, wolf in enumerate(wolves):
            alpha, beta, delta = _select_leaders(archive_solutions, archive_objectives)
            next_wolves[index] = _update_position(
                wolf,
                alpha,
                beta,
                delta,
                convergence_factor,
                inertia_weight,
                bounds,
            )

        wolves = next_wolves
        objectives = evaluate_objectives(objective_function, wolves)
        cauchy_candidate = _cauchy_variation(_best_sum_solution(archive_solutions, archive_objectives), bounds)
        cauchy_objective = objective_function(cauchy_candidate)
        worst_index = int(np.argmax(np.sum(objectives, axis=1)))
        wolves[worst_index] = cauchy_candidate
        objectives[worst_index] = cauchy_objective
        archive_solutions, archive_objectives = update_archive(
            archive_solutions,
            archive_objectives,
            wolves,
            objectives,
            archive_size,
        )
        history.append(best_sum_history_value(archive_objectives))

    return archive_solutions, archive_objectives, history, used_seed
