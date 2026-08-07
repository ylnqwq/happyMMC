# -*- coding: utf-8 -*-

import numpy as np

from multi_objective.mo_utils import (
    best_sum_history_value,
    dominates,
    evaluate_objectives,
    update_archive,
    validate_bounds,
)


def _random_indexes(population_size, excluded, count):
    excluded = set(excluded)
    candidates = [index for index in range(population_size) if index not in excluded]
    replace = len(candidates) < count
    return np.random.choice(candidates, size=count, replace=replace)


def _crossover(target, mutant, crossover_rate):
    dimension = len(target)
    mask = np.random.rand(dimension) < crossover_rate
    mask[np.random.randint(dimension)] = True
    return np.where(mask, mutant, target)


def _select_survivor(target, target_objective, trial, trial_objective):
    if dominates(trial_objective, target_objective):
        return trial, trial_objective
    if dominates(target_objective, trial_objective):
        return target, target_objective
    if np.sum(trial_objective) <= np.sum(target_objective):
        return trial, trial_objective
    return target, target_objective


def _best_sum_solution(solutions, objectives):
    return solutions[int(np.argmin(np.sum(objectives, axis=1)))]


def _build_competition_groups(objectives, previous_groups=None):
    population_size = len(objectives)
    indexes = np.random.permutation(population_size)
    winners = []
    losers = []
    nondominated = []

    pair_count = population_size // 2
    for pair_index in range(pair_count):
        left = int(indexes[2 * pair_index])
        right = int(indexes[2 * pair_index + 1])
        if dominates(objectives[left], objectives[right]):
            winners.append(left)
            losers.append(right)
        elif dominates(objectives[right], objectives[left]):
            winners.append(right)
            losers.append(left)
        else:
            nondominated.extend([left, right])

    if population_size % 2 == 1:
        nondominated.append(int(indexes[-1]))

    if not winners and not losers and previous_groups is not None:
        return previous_groups

    if not winners or not losers:
        order = np.argsort(np.sum(objectives, axis=1))
        split = max(1, population_size // 3)
        winners = list(order[:split])
        losers = list(order[-split:])
        nondominated = [index for index in range(population_size) if index not in set(winners + losers)]

    return {
        "winners": np.array(winners, dtype=int),
        "losers": np.array(losers, dtype=int),
        "nondominated": np.array(nondominated, dtype=int),
    }


def _sample_group(groups, group_name, population_size, fallback_excluded=()):
    group = groups[group_name]
    if len(group) > 0:
        return int(np.random.choice(group))
    return int(_random_indexes(population_size, fallback_excluded, 1)[0])


def _mutate_winner(population, index, groups, mutation_factor):
    population_size = len(population)
    winner_index = _sample_group(groups, "winners", population_size, fallback_excluded=(index,))
    loser_index = _sample_group(groups, "losers", population_size, fallback_excluded=(index, winner_index))
    random_index = int(_random_indexes(population_size, (index, winner_index, loser_index), 1)[0])
    return population[winner_index] + mutation_factor * (population[random_index] - population[loser_index])


def _mutate_loser(population, index, groups, mutation_factor):
    population_size = len(population)
    winner_index = _sample_group(groups, "winners", population_size, fallback_excluded=(index,))
    loser_index = _sample_group(groups, "losers", population_size, fallback_excluded=(index, winner_index))
    return (
        population[index]
        + mutation_factor * (population[winner_index] - population[index])
        + mutation_factor * (population[winner_index] - population[loser_index])
    )


def _mutate_nondominated(population, index, global_best, mutation_factor):
    population_size = len(population)
    r1, r2 = _random_indexes(population_size, (index,), 2)
    if np.random.rand() < 0.5:
        return population[index] + mutation_factor * (global_best - population[r2])
    return population[index] + mutation_factor * (global_best - population[index]) + mutation_factor * (
        population[r1] - population[r2]
    )


def _elite_self_search(solution, bounds, iteration, max_iter):
    ratio = max(0.0, (max_iter - iteration) / max(1, max_iter))
    gaussian = np.random.normal(0.0, 1.0, size=len(solution))
    explored = solution + ratio * solution * gaussian
    return np.clip(explored, bounds[:, 0], bounds[:, 1])


def cmmode(
    objective_function,
    bounds,
    bee=80,
    population_size=None,
    max_iter=800,
    limit=None,
    archive_size=100,
    mutation_factor=0.5,
    crossover_rate=0.9,
    elite_search_rate=0.5,
    seed=None,
):
    if seed is None:
        used_seed = int(np.random.SeedSequence().generate_state(1)[0])
    else:
        used_seed = int(seed)
    np.random.seed(used_seed)

    bounds = validate_bounds(bounds)
    dimension = len(bounds)
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]
    population_size = max(6, int(bee if population_size is None else population_size))

    population = np.random.uniform(lower_bounds, upper_bounds, size=(population_size, dimension))
    objectives = evaluate_objectives(objective_function, population)
    archive_solutions = np.empty((0, dimension), dtype=float)
    archive_objectives = np.empty((0, objectives.shape[1]), dtype=float)
    archive_solutions, archive_objectives = update_archive(
        archive_solutions,
        archive_objectives,
        population,
        objectives,
        archive_size,
    )
    history = [best_sum_history_value(archive_objectives)]
    previous_groups = None

    for iteration in range(max_iter):
        groups = _build_competition_groups(objectives, previous_groups)
        previous_groups = groups
        winner_set = set(groups["winners"])
        loser_set = set(groups["losers"])
        global_best = _best_sum_solution(archive_solutions, archive_objectives)

        next_population = population.copy()
        next_objectives = objectives.copy()
        for index in range(population_size):
            if index in winner_set:
                mutant = _mutate_winner(population, index, groups, mutation_factor)
            elif index in loser_set:
                mutant = _mutate_loser(population, index, groups, mutation_factor)
            else:
                mutant = _mutate_nondominated(population, index, global_best, mutation_factor)

            mutant = np.clip(mutant, lower_bounds, upper_bounds)
            trial = _crossover(population[index], mutant, crossover_rate)
            trial = np.clip(trial, lower_bounds, upper_bounds)
            trial_objective = objective_function(trial)
            selected, selected_objective = _select_survivor(
                population[index],
                objectives[index],
                trial,
                trial_objective,
            )

            if index not in winner_set and np.random.rand() < elite_search_rate:
                elite_trial = _elite_self_search(selected, bounds, iteration, max_iter)
                elite_objective = objective_function(elite_trial)
                selected, selected_objective = _select_survivor(
                    selected,
                    selected_objective,
                    elite_trial,
                    elite_objective,
                )

            next_population[index] = selected
            next_objectives[index] = selected_objective

        population = next_population
        objectives = next_objectives
        archive_solutions, archive_objectives = update_archive(
            archive_solutions,
            archive_objectives,
            population,
            objectives,
            archive_size,
        )
        history.append(best_sum_history_value(archive_objectives))

    return archive_solutions, archive_objectives, history, used_seed
