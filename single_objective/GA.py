# -*- coding: utf-8 -*-

import numpy as np


def _validate_bounds(bounds):
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must be shaped like [(lower, upper), ...].")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("each lower bound must be smaller than the upper bound.")
    return bounds


def _evaluate_population(population, objective_function):
    return np.array([objective_function(individual) for individual in population], dtype=float)


def _fitness_from_values(values):
    values = np.asarray(values, dtype=float)
    shifted = values - np.min(values)
    return 1.0 / (1.0 + shifted)


def _roulette_select(population, values):
    fitness = _fitness_from_values(values)
    probabilities = fitness / np.sum(fitness)
    index = int(np.random.choice(len(population), p=probabilities))
    return population[index].copy()


def _single_point_crossover(parent_a, parent_b):
    dimension = len(parent_a)
    if dimension <= 1:
        return parent_a.copy(), parent_b.copy()

    point = np.random.randint(1, dimension)
    child_a = np.concatenate((parent_a[:point], parent_b[point:]))
    child_b = np.concatenate((parent_b[:point], parent_a[point:]))
    return child_a, child_b


def _mutate(individual, bounds, mutation_rate):
    mask = np.random.rand(len(individual)) < mutation_rate
    if np.any(mask):
        individual = individual.copy()
        individual[mask] = np.random.uniform(bounds[mask, 0], bounds[mask, 1])
    return individual


def genetic_algorithm(
    objective_function,
    bounds,
    population_size=75,
    max_iter=750,
    crossover_rate=0.8,
    mutation_rate=None,
    seed=None,
):
    """Basic real-coded genetic algorithm for continuous minimization."""
    if seed is None:
        used_seed = int(np.random.SeedSequence().generate_state(1)[0])
    else:
        used_seed = int(seed)
    np.random.seed(used_seed)

    bounds = _validate_bounds(bounds)
    dimension = len(bounds)
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]

    population_size = max(2, int(population_size))
    if mutation_rate is None:
        mutation_rate = 1.0 / dimension

    population = np.random.uniform(lower_bounds, upper_bounds, size=(population_size, dimension))
    values = _evaluate_population(population, objective_function)

    best_index = int(np.argmin(values))
    best_solution = population[best_index].copy()
    best_value = float(values[best_index])
    history = [best_value]

    for _ in range(max_iter):
        next_population = []
        while len(next_population) < population_size:
            parent_a = _roulette_select(population, values)
            parent_b = _roulette_select(population, values)

            if np.random.rand() < crossover_rate:
                child_a, child_b = _single_point_crossover(parent_a, parent_b)
            else:
                child_a, child_b = parent_a.copy(), parent_b.copy()

            child_a = _mutate(child_a, bounds, mutation_rate)
            child_b = _mutate(child_b, bounds, mutation_rate)
            next_population.append(child_a)
            if len(next_population) < population_size:
                next_population.append(child_b)

        population = np.asarray(next_population, dtype=float)
        values = _evaluate_population(population, objective_function)

        current_best_index = int(np.argmin(values))
        if values[current_best_index] < best_value:
            best_value = float(values[current_best_index])
            best_solution = population[current_best_index].copy()

        history.append(best_value)

    return best_solution, best_value, history, used_seed
