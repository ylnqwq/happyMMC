# -*- coding: utf-8 -*-

import numpy as np


def calculate_fitness(values):
    """将最小化目标函数值转换为适应度，目标值越小适应度越大"""
    values = np.asarray(values, dtype=float)
    fitness = np.empty_like(values, dtype=float)

    non_negative = values >= 0
    fitness[non_negative] = 1.0 / (1.0 + values[non_negative])
    fitness[~non_negative] = 1.0 + np.abs(values[~non_negative])
    return fitness


def initialize_food_sources(food_number, bounds, objective_function):
    """随机初始化蜜源位置、目标函数值和试探次数"""
    bounds = np.asarray(bounds, dtype=float)
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]

    food_sources = np.random.uniform(lower_bounds, upper_bounds, size=(food_number, len(bounds)))
    values = np.array([objective_function(source) for source in food_sources], dtype=float)
    trials = np.zeros(food_number, dtype=int)
    return food_sources, values, trials


def create_neighbor(food_sources, index, bounds):
    """根据当前蜜源随机生成一个邻域蜜源。"""
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


def greedy_select(food_sources, values, trials, index, candidate, candidate_value):
    """贪婪选择，候选蜜源更优则替换，否则试探次数加一。"""
    if candidate_value < values[index]:
        food_sources[index] = candidate
        values[index] = candidate_value
        trials[index] = 0
    else:
        trials[index] += 1


def employed_bee_phase(food_sources, values, trials, bounds, objective_function):
    """雇佣蜂阶段。"""
    for i in range(len(food_sources)):
        candidate = create_neighbor(food_sources, i, bounds)
        candidate_value = objective_function(candidate)
        greedy_select(food_sources, values, trials, i, candidate, candidate_value)


def onlooker_bee_phase(food_sources, values, trials, bounds, objective_function):
    """观察蜂阶段，按适应度轮盘赌选择蜜源。"""
    fitness = calculate_fitness(values)
    probabilities = fitness / fitness.sum()

    food_number = len(food_sources)
    selected_count = 0
    i = 0
    while selected_count < food_number:
        if np.random.rand() < probabilities[i]:
            candidate = create_neighbor(food_sources, i, bounds)
            candidate_value = objective_function(candidate)
            greedy_select(food_sources, values, trials, i, candidate, candidate_value)
            selected_count += 1

        i = (i + 1) % food_number


def scout_bee_phase(food_sources, values, trials, bounds, objective_function, limit):
    """侦察蜂阶段，超过试探上限的蜜源会被随机重新初始化。"""
    bounds = np.asarray(bounds, dtype=float)
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]

    for i in range(len(food_sources)):
        if trials[i] >= limit:
            food_sources[i] = np.random.uniform(lower_bounds, upper_bounds)
            values[i] = objective_function(food_sources[i])
            trials[i] = 0


def artificial_bee_colony(
    objective_function,
    bounds,
    bee=30,
    max_iter=500,
    limit=100,
    seed=None,
):
    """经典人工蜂群算法，默认求解最小化问题。"""
    if seed is None:
        used_seed = int(np.random.SeedSequence().generate_state(1)[0])
    else:
        used_seed = int(seed)
    np.random.seed(used_seed)

    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds 必须是形如 [(lower, upper), ...] 的二维数组或列表")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("每个变量的下界必须小于上界")

    food_sources, values, trials = initialize_food_sources(bee, bounds, objective_function)

    best_index = np.argmin(values)
    best_solution = food_sources[best_index].copy()
    best_value = values[best_index]
    history = [best_value]

    for _ in range(max_iter):
        employed_bee_phase(food_sources, values, trials, bounds, objective_function)
        onlooker_bee_phase(food_sources, values, trials, bounds, objective_function)
        scout_bee_phase(food_sources, values, trials, bounds, objective_function, limit)

        current_best_index = np.argmin(values)
        if values[current_best_index] < best_value:
            best_value = values[current_best_index]
            best_solution = food_sources[current_best_index].copy()

        history.append(best_value)

    return best_solution, best_value, history, used_seed
