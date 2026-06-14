# -*- coding: utf-8 -*-

import numpy as np


def zdt1(x):
    x = np.asarray(x, dtype=float)
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
    h = 1.0 - np.sqrt(f1 / g)
    return np.array([f1, g * h], dtype=float)


def zdt2(x):
    x = np.asarray(x, dtype=float)
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
    h = 1.0 - (f1 / g) ** 2
    return np.array([f1, g * h], dtype=float)


def zdt3(x):
    x = np.asarray(x, dtype=float)
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
    h = 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1)
    return np.array([f1, g * h], dtype=float)


def zdt4(x):
    x = np.asarray(x, dtype=float)
    f1 = x[0]
    tail = x[1:]
    g = 1.0 + 10.0 * (len(x) - 1) + np.sum(tail**2 - 10.0 * np.cos(4.0 * np.pi * tail))
    h = 1.0 - np.sqrt(f1 / g)
    return np.array([f1, g * h], dtype=float)


def zdt6(x):
    x = np.asarray(x, dtype=float)
    f1 = 1.0 - np.exp(-4.0 * x[0]) * np.sin(6.0 * np.pi * x[0]) ** 6
    g = 1.0 + 9.0 * (np.sum(x[1:]) / (len(x) - 1)) ** 0.25
    h = 1.0 - (f1 / g) ** 2
    return np.array([f1, g * h], dtype=float)


def _benchmark(benchmark_id, name, objective_function, bounds, reference_point):
    return {
        "id": benchmark_id,
        "name": name,
        "function": objective_function,
        "bounds": bounds,
        "objective_count": 2,
        "reference_point": np.asarray(reference_point, dtype=float),
    }


ZDT_BENCHMARKS = [
    _benchmark("ZDT1", "ZDT1 双目标函数", zdt1, [(0.0, 1.0)] * 30, [1.1, 1.1]),
    _benchmark("ZDT2", "ZDT2 双目标函数", zdt2, [(0.0, 1.0)] * 30, [1.1, 1.1]),
    _benchmark("ZDT3", "ZDT3 双目标函数", zdt3, [(0.0, 1.0)] * 30, [1.1, 1.1]),
    _benchmark("ZDT4", "ZDT4 双目标函数", zdt4, [(0.0, 1.0)] + [(-5.0, 5.0)] * 9, [1.1, 120.0]),
    _benchmark("ZDT6", "ZDT6 双目标函数", zdt6, [(0.0, 1.0)] * 10, [1.1, 1.1]),
]
