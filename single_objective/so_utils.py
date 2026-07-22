# -*- coding: utf-8 -*-

import numpy as np


def validate_bounds(bounds):
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must be shaped like [(lower, upper), ...].")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("each lower bound must be smaller than the upper bound.")
    return bounds


def evaluate_values(objective_function, sources):
    return np.array([objective_function(source) for source in sources], dtype=float)


def calculate_fitness(values):
    values = np.asarray(values, dtype=float)
    fitness = np.empty_like(values, dtype=float)
    non_negative = values >= 0
    fitness[non_negative] = 1.0 / (1.0 + values[non_negative])
    fitness[~non_negative] = 1.0 + np.abs(values[~non_negative])
    return fitness


def select_partner(item_count, current_index):
    partner_index = np.random.randint(item_count)
    while partner_index == current_index:
        partner_index = np.random.randint(item_count)
    return partner_index


def initialize_random_sources(food_number, bounds, objective_function):
    bounds = validate_bounds(bounds)
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]
    food_sources = np.random.uniform(lower_bounds, upper_bounds, size=(food_number, len(bounds)))
    values = evaluate_values(objective_function, food_sources)
    trials = np.zeros(food_number, dtype=int)
    return food_sources, values, trials


def reinitialize_source(sources, values, trials, index, lower_bounds, upper_bounds, objective_function):
    sources[index] = np.random.uniform(lower_bounds, upper_bounds)
    values[index] = objective_function(sources[index])
    trials[index] = 0
