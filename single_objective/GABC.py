# -*- coding: utf-8 -*-

import numpy as np


def calculate_fitness(values):
    """将最小化目标函数值转换为适应度，目标值越小适应度越大。"""
    values = np.asarray(values, dtype=float)
    fitness = np.empty_like(values, dtype=float)

    non_negative = values >= 0
    fitness[non_negative] = 1.0 / (1.0 + values[non_negative])
    fitness[~non_negative] = 1.0 + np.abs(values[~non_negative])
    return fitness


def calculate_relative_fitness(values):
    """计算相对适应度，降低极端个体对选择过程的影响。"""
    fitness = calculate_fitness(values)
    min_fitness = np.min(fitness)
    max_fitness = np.max(fitness)

    if np.isclose(max_fitness, min_fitness):
        return np.ones_like(fitness)

    return (fitness - min_fitness) / (max_fitness - min_fitness) + 1e-12


def good_point_set(food_number, dimension):
    """佳点集初始化序列，用确定性低差异序列提升初始种群多样性。"""
    prime = 2 * dimension + 3
    while any(prime % factor == 0 for factor in range(2, int(np.sqrt(prime)) + 1)):
        prime += 1

    r = 2 * np.cos(2 * np.pi * np.arange(1, dimension + 1) / prime)
    indexes = np.arange(1, food_number + 1).reshape(-1, 1)
    return np.mod(indexes * r, 1.0)


def initialize_food_sources(food_number, bounds, objective_function):
    """使用佳点集初始化蜜源位置、目标函数值和试探次数。"""
    bounds = np.asarray(bounds, dtype=float)
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]
    dimension = len(bounds)

    good_points = good_point_set(food_number, dimension)
    food_sources = lower_bounds + good_points * (upper_bounds - lower_bounds)
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


def greedy_select(food_sources, values, trials, index, candidate, candidate_value, count_failure=True):
    """候选蜜源更优则替换，否则试探次数加一。"""
    if candidate_value < values[index]:
        food_sources[index] = candidate
        values[index] = candidate_value
        trials[index] = 0
    elif count_failure:
        trials[index] += 1


def employed_bee_phase(food_sources, values, trials, bounds, objective_function):
    """雇佣蜂阶段。"""
    for i in range(len(food_sources)):
        candidate = create_neighbor(food_sources, i, bounds)
        candidate_value = objective_function(candidate)
        greedy_select(food_sources, values, trials, i, candidate, candidate_value)


def tournament_select(relative_fitness, tournament_size=3):
    """锦标赛选择，从随机候选中选择相对适应度最高的个体。"""
    food_number = len(relative_fitness)
    tournament_size = min(tournament_size, food_number)
    candidates = np.random.choice(food_number, size=tournament_size, replace=False)
    winner_position = np.argmax(relative_fitness[candidates])
    return candidates[winner_position]


def onlooker_bee_phase(
    food_sources,
    values,
    trials,
    bounds,
    objective_function,
    tournament_size=3,
):
    """观察蜂阶段，使用相对适应度进行锦标赛选择。"""
    relative_fitness = calculate_relative_fitness(values)

    for _ in range(len(food_sources)):
        selected_index = tournament_select(relative_fitness, tournament_size)
        candidate = create_neighbor(food_sources, selected_index, bounds)
        candidate_value = objective_function(candidate)
        greedy_select(food_sources, values, trials, selected_index, candidate, candidate_value)


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


def elite_enhancement_phase(
    food_sources,
    values,
    trials,
    bounds,
    objective_function,
    elite_rate=0.1,
):
    """对排名靠前的精英蜜源额外搜索一轮。"""
    food_number = len(food_sources)
    elite_number = max(1, int(np.ceil(food_number * elite_rate)))
    elite_indexes = np.argsort(values)[:elite_number]

    for index in elite_indexes:
        candidate = create_neighbor(food_sources, index, bounds)
        candidate_value = objective_function(candidate)
        greedy_select(
            food_sources,
            values,
            trials,
            index,
            candidate,
            candidate_value,
            count_failure=False,
        )


def worst_elimination_phase(
    food_sources,
    values,
    trials,
    bounds,
    objective_function,
    elimination_rate=0.1,
):
    """按比例淘汰较差蜜源并重新随机赋值。"""
    food_number = len(food_sources)
    elimination_number = int(np.ceil(food_number * elimination_rate))
    if elimination_number <= 0:
        return

    worst_indexes = np.argsort(values)[-elimination_number:]

    bounds = np.asarray(bounds, dtype=float)
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]

    for index in worst_indexes:
        food_sources[index] = np.random.uniform(lower_bounds, upper_bounds)
        values[index] = objective_function(food_sources[index])
        trials[index] = 0


def get_current_elimination_rate(initial_rate, iteration, max_iter):
    """按迭代进度逐步增加末位淘汰比例，前期为 0，最后升至 initial_rate。"""
    if initial_rate <= 0:
        return 0.0

    stage = int((iteration + 1) / max(1, max_iter) * 10)
    stage = min(stage, 10)
    return initial_rate * stage / 10


def gabc(
    objective_function,
    bounds,
    bee=30,
    max_iter=500,
    limit=100,
    tournament_size=3,
    elite_rate=0.1,
    elimination_rate=0.1,
    seed=None,
):
    """改进人工蜂群算法 GABC，默认求解最小化问题。"""
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

    for iteration in range(max_iter):
        employed_bee_phase(food_sources, values, trials, bounds, objective_function)
        onlooker_bee_phase(
            food_sources,
            values,
            trials,
            bounds,
            objective_function,
            tournament_size=tournament_size,
        )
        scout_bee_phase(food_sources, values, trials, bounds, objective_function, limit)
        elite_enhancement_phase(
            food_sources,
            values,
            trials,
            bounds,
            objective_function,
            elite_rate=elite_rate,
        )
        current_elimination_rate = get_current_elimination_rate(
            elimination_rate,
            iteration,
            max_iter,
        )
        worst_elimination_phase(
            food_sources,
            values,
            trials,
            bounds,
            objective_function,
            elimination_rate=current_elimination_rate,
        )

        current_best_index = np.argmin(values)
        if values[current_best_index] < best_value:
            best_value = values[current_best_index]
            best_solution = food_sources[current_best_index].copy()

        history.append(best_value)

    return best_solution, best_value, history, used_seed
