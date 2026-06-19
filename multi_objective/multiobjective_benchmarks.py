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


def mmf1(x):
    x = np.asarray(x, dtype=float).copy()
    x1 = abs(x[0] - 2.0)
    f1 = x1
    f2 = 1.0 - np.sqrt(x1) + 2.0 * (x[1] - np.sin(6.0 * np.pi * x1 + np.pi)) ** 2
    return np.array([f1, f2], dtype=float)


def mmf2(x):
    x = np.asarray(x, dtype=float).copy()
    if x[1] > 1.0:
        x[1] -= 1.0
    f1 = x[0]
    y2 = x[1] - np.sqrt(x[0])
    f2 = 1.0 - np.sqrt(x[0]) + 2.0 * (4.0 * y2**2 - 2.0 * np.cos(20.0 * np.pi * y2 / np.sqrt(2.0)) + 2.0)
    return np.array([f1, f2], dtype=float)


def mmf4(x):
    x = np.asarray(x, dtype=float).copy()
    if x[1] > 1.0:
        x[1] -= 1.0
    f1 = abs(x[0])
    f2 = 1.0 - x[0] ** 2 + 2.0 * (x[1] - np.sin(np.pi * abs(x[0]))) ** 2
    return np.array([f1, f2], dtype=float)


def mmf5(x):
    x = np.asarray(x, dtype=float).copy()
    if x[1] > 1.0:
        x[1] -= 2.0
    x1 = abs(x[0] - 2.0)
    f1 = x1
    f2 = 1.0 - np.sqrt(x1) + 2.0 * (x[1] - np.sin(6.0 * np.pi * x1 + np.pi)) ** 2
    return np.array([f1, f2], dtype=float)


def mmf7(x):
    x = np.asarray(x, dtype=float).copy()
    x1 = abs(x[0] - 2.0)
    target = (0.3 * x1**2 * np.cos(24.0 * np.pi * x1 + 4.0 * np.pi) + 0.6 * x1) * np.sin(
        6.0 * np.pi * x1 + np.pi
    )
    f1 = x1
    f2 = 1.0 - np.sqrt(x1) + (x[1] - target) ** 2
    return np.array([f1, f2], dtype=float)


def mmf8(x):
    x = np.asarray(x, dtype=float).copy()
    if x[1] > 4.0:
        x[1] -= 4.0
    sin_abs = np.sin(abs(x[0]))
    f1 = sin_abs
    f2 = np.sqrt(max(0.0, 1.0 - sin_abs**2)) + 2.0 * (x[1] - (sin_abs + abs(x[0]))) ** 2
    return np.array([f1, f2], dtype=float)


def mmf10(x):
    x = np.asarray(x, dtype=float)
    g = 2.0 - np.exp(-((x[1] - 0.2) / 0.004) ** 2) - 0.8 * np.exp(-((x[1] - 0.6) / 0.4) ** 2)
    return np.array([x[0], g / x[0]], dtype=float)


def mmf11(x):
    x = np.asarray(x, dtype=float)
    number_of_peaks = 2.0
    temp1 = np.sin(number_of_peaks * np.pi * x[1]) ** 6
    temp2 = np.exp(-2.0 * np.log10(2.0) * ((x[1] - 0.1) / 0.8) ** 2)
    g = 2.0 - temp2 * temp1
    return np.array([x[0], g / x[0]], dtype=float)


def mmf12(x):
    x = np.asarray(x, dtype=float)
    q = 4.0
    alpha = 2.0
    number_of_peaks = 2.0
    f1 = x[0]
    g = 2.0 - (np.sin(number_of_peaks * np.pi * x[1]) ** 6) * np.exp(
        -2.0 * np.log10(2.0) * ((x[1] - 0.1) / 0.8) ** 2
    )
    h = 1.0 - (f1 / g) ** alpha - (f1 / g) * np.sin(2.0 * np.pi * q * f1)
    return np.array([f1, g * h], dtype=float)


def mmf13(x):
    x = np.asarray(x, dtype=float)
    nonlinear_variable = x[1] + np.sqrt(x[2])
    g = 2.0 - np.exp(-2.0 * np.log10(2.0) * ((nonlinear_variable - 0.1) / 0.8) ** 2) * np.sin(
        2.0 * np.pi * nonlinear_variable
    ) ** 6
    return np.array([x[0], g / x[0]], dtype=float)


def mmf1_e(x):
    x = np.asarray(x, dtype=float)
    if x[0] < 2.0:
        x1 = 2.0 - x[0]
        f2 = 1.0 - np.sqrt(x1) + 2.0 * (x[1] - np.sin(6.0 * np.pi * x1 + np.pi)) ** 2
    else:
        x1 = x[0] - 2.0
        f2 = 1.0 - np.sqrt(x1) + 2.0 * (x[1] - np.exp(x[0]) * np.sin(6.0 * np.pi * x1 + np.pi)) ** 2
    return np.array([x1, f2], dtype=float)


def _dtlz2_shape_3d(x, g):
    c1 = np.cos(x[0] * np.pi / 2.0)
    c2 = np.cos(x[1] * np.pi / 2.0)
    s1 = np.sin(x[0] * np.pi / 2.0)
    s2 = np.sin(x[1] * np.pi / 2.0)
    radius = 1.0 + g
    return radius * np.array([c1 * c2, c1 * s2, s1], dtype=float)


def mmf14(x):
    x = np.asarray(x, dtype=float)
    number_of_peaks = 2.0
    g = 2.0 - np.sin(number_of_peaks * np.pi * x[-1]) ** 2
    return _dtlz2_shape_3d(x, g)


def mmf14_a(x):
    x = np.asarray(x, dtype=float)
    number_of_peaks = 2.0
    shifted = x[-1] - 0.5 * np.sin(np.pi * x[-2])
    g = 2.0 - np.sin(number_of_peaks * np.pi * (shifted + 1.0 / (2.0 * number_of_peaks))) ** 2
    return _dtlz2_shape_3d(x, g)


def mmf15(x):
    x = np.asarray(x, dtype=float)
    number_of_peaks = 2.0
    g = 2.0 - np.exp(-2.0 * np.log10(2.0) * ((x[-1] - 0.1) / 0.8) ** 2) * np.sin(
        number_of_peaks * np.pi * x[-1]
    ) ** 2
    return _dtlz2_shape_3d(x, g)


def mmf15_a(x):
    x = np.asarray(x, dtype=float)
    number_of_peaks = 2.0
    shifted = x[-1] - 0.5 * np.sin(np.pi * x[-2])
    shifted += 1.0 / (2.0 * number_of_peaks)
    g = 2.0 - np.exp(-2.0 * np.log10(2.0) * ((shifted - 0.1) / 0.8) ** 2) * np.sin(
        number_of_peaks * np.pi * shifted
    ) ** 2
    return _dtlz2_shape_3d(x, g)


def _mmf16_l(x, global_peaks, local_peaks):
    x = np.asarray(x, dtype=float)
    if x[-1] < 0.5:
        g = 2.0 - np.sin(2.0 * global_peaks * np.pi * x[-1]) ** 2
    else:
        g = 2.0 - np.exp(-2.0 * np.log10(2.0) * ((x[-1] - 0.1) / 0.8) ** 2) * np.sin(
            2.0 * local_peaks * np.pi * x[-1]
        ) ** 2
    return _dtlz2_shape_3d(x, g)


def mmf16_l1(x):
    return _mmf16_l(x, global_peaks=2.0, local_peaks=1.0)


def mmf16_l2(x):
    return _mmf16_l(x, global_peaks=1.0, local_peaks=2.0)


def mmf16_l3(x):
    return _mmf16_l(x, global_peaks=2.0, local_peaks=2.0)


def _benchmark(benchmark_id, name, objective_function, bounds, reference_point):
    reference_point = np.asarray(reference_point, dtype=float)
    return {
        "id": benchmark_id,
        "name": name,
        "function": objective_function,
        "bounds": bounds,
        "objective_count": len(reference_point),
        "reference_point": reference_point,
    }


ZDT_BENCHMARKS = [
    _benchmark("ZDT1", "ZDT1 双目标函数", zdt1, [(0.0, 1.0)] * 30, [1.1, 1.1]),
    _benchmark("ZDT2", "ZDT2 双目标函数", zdt2, [(0.0, 1.0)] * 30, [1.1, 1.1]),
    _benchmark("ZDT3", "ZDT3 双目标函数", zdt3, [(0.0, 1.0)] * 30, [1.1, 1.1]),
    _benchmark("ZDT4", "ZDT4 双目标函数", zdt4, [(0.0, 1.0)] + [(-5.0, 5.0)] * 9, [1.1, 120.0]),
    _benchmark("ZDT6", "ZDT6 双目标函数", zdt6, [(0.0, 1.0)] * 10, [1.1, 1.1]),
]


CEC2020_MMO_BENCHMARKS = [
    _benchmark("MMF1", "CEC2020 MMO MMF1 双目标函数", mmf1, [(1.0, 3.0), (-1.0, 1.0)], [1.1, 1.1]),
    _benchmark("MMF2", "CEC2020 MMO MMF2 双目标函数", mmf2, [(0.0, 1.0), (0.0, 2.0)], [1.1, 1.1]),
    _benchmark("MMF4", "CEC2020 MMO MMF4 双目标函数", mmf4, [(-1.0, 1.0), (0.0, 2.0)], [1.1, 1.1]),
    _benchmark("MMF5", "CEC2020 MMO MMF5 双目标函数", mmf5, [(1.0, 3.0), (-1.0, 3.0)], [1.1, 1.1]),
    _benchmark("MMF7", "CEC2020 MMO MMF7 双目标函数", mmf7, [(1.0, 3.0), (-1.0, 1.0)], [1.1, 1.1]),
    _benchmark("MMF8", "CEC2020 MMO MMF8 双目标函数", mmf8, [(-np.pi, np.pi), (0.0, 9.0)], [1.1, 1.1]),
    _benchmark("MMF10", "CEC2020 MMO MMF10 双目标函数", mmf10, [(0.1, 1.1), (0.1, 1.1)], [1.21, 13.2]),
    _benchmark("MMF11", "CEC2020 MMO MMF11 双目标函数", mmf11, [(0.1, 1.1), (0.1, 1.1)], [1.21, 15.4]),
    _benchmark("MMF12", "CEC2020 MMO MMF12 双目标函数", mmf12, [(0.0, 1.0), (0.0, 1.0)], [1.54, 1.1]),
    _benchmark(
        "MMF13",
        "CEC2020 MMO MMF13 双目标函数",
        mmf13,
        [(0.1, 1.1), (0.1, 1.1), (0.1, 1.1)],
        [1.54, 15.4],
    ),
    _benchmark("MMF14", "CEC2020 MMO MMF14 三目标函数", mmf14, [(0.0, 1.0)] * 3, [2.2, 2.2, 2.2]),
    _benchmark("MMF15", "CEC2020 MMO MMF15 三目标函数", mmf15, [(0.0, 1.0)] * 3, [2.5, 2.5, 2.5]),
    _benchmark("MMF1_E", "CEC2020 MMO MMF1_e 双目标函数", mmf1_e, [(1.0, 3.0), (-20.0, 20.0)], [1.1, 1.1]),
    _benchmark("MMF14_A", "CEC2020 MMO MMF14_a 三目标函数", mmf14_a, [(0.0, 1.0)] * 3, [2.2, 2.2, 2.2]),
    _benchmark("MMF15_A", "CEC2020 MMO MMF15_a 三目标函数", mmf15_a, [(0.0, 1.0)] * 3, [2.5, 2.5, 2.5]),
    _benchmark("MMF10_L", "CEC2020 MMO MMF10_l 双目标函数", mmf10, [(0.1, 1.1), (0.1, 1.1)], [1.21, 13.2]),
    _benchmark("MMF11_L", "CEC2020 MMO MMF11_l 双目标函数", mmf11, [(0.1, 1.1), (0.1, 1.1)], [1.21, 15.4]),
    _benchmark("MMF12_L", "CEC2020 MMO MMF12_l 双目标函数", mmf12, [(0.0, 1.0), (0.0, 1.0)], [1.54, 1.1]),
    _benchmark(
        "MMF13_L",
        "CEC2020 MMO MMF13_l 双目标函数",
        mmf13,
        [(0.1, 1.1), (0.1, 1.1), (0.1, 1.1)],
        [1.54, 15.4],
    ),
    _benchmark("MMF15_L", "CEC2020 MMO MMF15_l 三目标函数", mmf15, [(0.0, 1.0)] * 3, [2.5, 2.5, 2.5]),
    _benchmark("MMF15_A_L", "CEC2020 MMO MMF15_a_l 三目标函数", mmf15_a, [(0.0, 1.0)] * 3, [2.5, 2.5, 2.5]),
    _benchmark("MMF16_L1", "CEC2020 MMO MMF16_l1 三目标函数", mmf16_l1, [(0.0, 1.0)] * 3, [2.5, 2.5, 2.5]),
    _benchmark("MMF16_L2", "CEC2020 MMO MMF16_l2 三目标函数", mmf16_l2, [(0.0, 1.0)] * 3, [2.5, 2.5, 2.5]),
    _benchmark("MMF16_L3", "CEC2020 MMO MMF16_l3 三目标函数", mmf16_l3, [(0.0, 1.0)] * 3, [2.5, 2.5, 2.5]),
]
