# -*- coding: utf-8 -*-

import numpy as np

from multi_objective.mo_utils import (
    best_sum_history_value,
    crowding_distance,
    evaluate_objectives,
    non_dominated_mask,
    pareto_rank,
    update_archive,
    validate_bounds,
)


def _binary_tournament(population, objectives):
    ranks = pareto_rank(objectives)
    distances = np.zeros(len(population), dtype=float)
    for rank in np.unique(ranks):
        indexes = np.where(ranks == rank)[0]
        distances[indexes] = crowding_distance(objectives[indexes])

    left, right = np.random.choice(len(population), size=2, replace=False)
    if ranks[left] < ranks[right]:
        return population[left].copy()
    if ranks[right] < ranks[left]:
        return population[right].copy()
    if distances[left] >= distances[right]:
        return population[left].copy()
    return population[right].copy()


def _sbx_crossover(parent_a, parent_b, bounds, crossover_rate, eta=20.0):
    if np.random.rand() >= crossover_rate:
        return parent_a.copy(), parent_b.copy()

    child_a = parent_a.copy()
    child_b = parent_b.copy()
    for index in range(len(parent_a)):
        if np.random.rand() > 0.5 or np.isclose(parent_a[index], parent_b[index]):
            continue

        lower, upper = bounds[index]
        x1 = min(parent_a[index], parent_b[index])
        x2 = max(parent_a[index], parent_b[index])
        rand = np.random.rand()

        beta = 1.0 + 2.0 * (x1 - lower) / (x2 - x1)
        alpha = 2.0 - beta ** -(eta + 1.0)
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
        c1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

        beta = 1.0 + 2.0 * (upper - x2) / (x2 - x1)
        alpha = 2.0 - beta ** -(eta + 1.0)
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
        c2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

        child_a[index] = np.clip(c1, lower, upper)
        child_b[index] = np.clip(c2, lower, upper)

    return child_a, child_b


def _polynomial_mutation(individual, bounds, mutation_rate, eta=20.0):
    child = individual.copy()
    for index in range(len(child)):
        if np.random.rand() >= mutation_rate:
            continue

        lower, upper = bounds[index]
        span = upper - lower
        if span <= 0:
            continue

        delta1 = (child[index] - lower) / span
        delta2 = (upper - child[index]) / span
        rand = np.random.rand()
        power = 1.0 / (eta + 1.0)

        if rand < 0.5:
            xy = 1.0 - delta1
            value = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1.0)
            deltaq = value**power - 1.0
        else:
            xy = 1.0 - delta2
            value = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1.0)
            deltaq = 1.0 - value**power

        child[index] = np.clip(child[index] + deltaq * span, lower, upper)
    return child


def _environmental_select(population, objectives, population_size):
    ranks = pareto_rank(objectives)
    selected = []

    for rank in np.unique(ranks):
        front = np.where(ranks == rank)[0]
        if len(selected) + len(front) <= population_size:
            selected.extend(front.tolist())
            continue

        distances = crowding_distance(objectives[front])
        order = np.argsort(-distances)
        needed = population_size - len(selected)
        selected.extend(front[order[:needed]].tolist())
        break

    selected = np.asarray(selected, dtype=int)
    return population[selected], objectives[selected]


def nsga2(
    objective_function,
    bounds,
    population_size=75,
    max_iter=750,
    crossover_rate=0.9,
    mutation_rate=None,
    archive_size=100,
    seed=None,
):
    if seed is None:
        used_seed = int(np.random.SeedSequence().generate_state(1)[0])
    else:
        used_seed = int(seed)
    np.random.seed(used_seed)

    bounds = validate_bounds(bounds)
    dimension = len(bounds)
    if mutation_rate is None:
        mutation_rate = 1.0 / dimension

    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]
    population = np.random.uniform(lower_bounds, upper_bounds, size=(population_size, dimension))
    objectives = evaluate_objectives(objective_function, population)

    archive_solutions = np.empty((0, dimension), dtype=float)
    archive_objectives = np.empty((0, objectives.shape[1]), dtype=float)
    archive_solutions, archive_objectives = update_archive(
        archive_solutions, archive_objectives, population, objectives, archive_size
    )
    history = [best_sum_history_value(archive_objectives)]

    for _ in range(max_iter):
        offspring = []
        while len(offspring) < population_size:
            parent_a = _binary_tournament(population, objectives)
            parent_b = _binary_tournament(population, objectives)
            child_a, child_b = _sbx_crossover(parent_a, parent_b, bounds, crossover_rate)
            child_a = _polynomial_mutation(child_a, bounds, mutation_rate)
            child_b = _polynomial_mutation(child_b, bounds, mutation_rate)
            offspring.append(child_a)
            if len(offspring) < population_size:
                offspring.append(child_b)

        offspring = np.asarray(offspring, dtype=float)
        offspring_objectives = evaluate_objectives(objective_function, offspring)
        combined = np.vstack([population, offspring])
        combined_objectives = np.vstack([objectives, offspring_objectives])
        population, objectives = _environmental_select(combined, combined_objectives, population_size)
        archive_solutions, archive_objectives = update_archive(
            archive_solutions, archive_objectives, population, objectives, archive_size
        )
        history.append(best_sum_history_value(archive_objectives))

    return archive_solutions, archive_objectives, history, used_seed

