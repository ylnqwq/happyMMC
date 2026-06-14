# -*- coding: utf-8 -*-

import numpy as np


def validate_bounds(bounds):
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds 必须是形如 [(lower, upper), ...] 的二维数组或列表")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("每个变量的下界必须小于上界")
    return bounds


def evaluate_objectives(objective_function, solutions):
    return np.array([objective_function(solution) for solution in solutions], dtype=float)


def dominates(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return np.all(left <= right) and np.any(left < right)


def non_dominated_mask(objectives):
    objectives = np.asarray(objectives, dtype=float)
    mask = np.ones(len(objectives), dtype=bool)

    for i in range(len(objectives)):
        if not mask[i]:
            continue
        for j in range(len(objectives)):
            if i != j and dominates(objectives[j], objectives[i]):
                mask[i] = False
                break

    return mask


def pareto_rank(objectives):
    objectives = np.asarray(objectives, dtype=float)
    remaining = np.arange(len(objectives))
    ranks = np.zeros(len(objectives), dtype=int)
    rank = 0

    while len(remaining) > 0:
        front_mask = non_dominated_mask(objectives[remaining])
        front = remaining[front_mask]
        ranks[front] = rank
        remaining = remaining[~front_mask]
        rank += 1

    return ranks


def crowding_distance(objectives):
    objectives = np.asarray(objectives, dtype=float)
    count = len(objectives)
    if count == 0:
        return np.array([], dtype=float)
    if count <= 2:
        return np.full(count, np.inf)

    distance = np.zeros(count, dtype=float)
    objective_count = objectives.shape[1]

    for objective_index in range(objective_count):
        order = np.argsort(objectives[:, objective_index])
        ordered_values = objectives[order, objective_index]
        span = ordered_values[-1] - ordered_values[0]

        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        if np.isclose(span, 0.0):
            continue

        for position in range(1, count - 1):
            previous_value = ordered_values[position - 1]
            next_value = ordered_values[position + 1]
            distance[order[position]] += (next_value - previous_value) / span

    return distance


def truncate_by_crowding(solutions, objectives, max_size):
    if len(solutions) <= max_size:
        return solutions, objectives

    distance = crowding_distance(objectives)
    finite_distance = np.where(np.isfinite(distance), distance, np.finfo(float).max)
    selected = np.argsort(-finite_distance)[:max_size]
    return solutions[selected], objectives[selected]


def update_archive(archive_solutions, archive_objectives, candidate_solutions, candidate_objectives, max_size):
    candidate_solutions = np.asarray(candidate_solutions, dtype=float)
    candidate_objectives = np.asarray(candidate_objectives, dtype=float)

    if len(archive_solutions) == 0:
        all_solutions = candidate_solutions
        all_objectives = candidate_objectives
    else:
        all_solutions = np.vstack([archive_solutions, candidate_solutions])
        all_objectives = np.vstack([archive_objectives, candidate_objectives])

    _, unique_indexes = np.unique(all_objectives, axis=0, return_index=True)
    unique_indexes = np.sort(unique_indexes)
    all_solutions = all_solutions[unique_indexes]
    all_objectives = all_objectives[unique_indexes]

    mask = non_dominated_mask(all_objectives)
    archive_solutions = all_solutions[mask]
    archive_objectives = all_objectives[mask]
    return truncate_by_crowding(archive_solutions, archive_objectives, max_size)


def population_scores(objectives):
    ranks = pareto_rank(objectives)
    distances = np.zeros(len(objectives), dtype=float)
    for rank in np.unique(ranks):
        indexes = np.where(ranks == rank)[0]
        distances[indexes] = crowding_distance(objectives[indexes])
    return ranks, distances


def greedy_select_multi(solutions, objectives, trials, index, candidate_solution, candidate_objective):
    current_objective = objectives[index]

    if dominates(candidate_objective, current_objective):
        solutions[index] = candidate_solution
        objectives[index] = candidate_objective
        trials[index] = 0
        return

    if dominates(current_objective, candidate_objective):
        trials[index] += 1
        return

    if np.random.rand() < 0.5:
        solutions[index] = candidate_solution
        objectives[index] = candidate_objective
        trials[index] = 0
    else:
        trials[index] += 1


def spacing_metric(objectives):
    objectives = np.asarray(objectives, dtype=float)
    if len(objectives) <= 2:
        return 0.0

    distances = []
    for i in range(len(objectives)):
        diff = objectives - objectives[i]
        norm = np.linalg.norm(diff, axis=1)
        norm[i] = np.inf
        distances.append(np.min(norm))

    distances = np.asarray(distances, dtype=float)
    return float(np.std(distances, ddof=1))


def best_sum_history_value(archive_objectives):
    if len(archive_objectives) == 0:
        return np.inf
    return float(np.min(np.sum(archive_objectives, axis=1)))
