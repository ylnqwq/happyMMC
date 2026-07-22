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


def _uf_odd_even_y(x, shape_function):
    x = np.asarray(x, dtype=float)
    n = len(x)
    j = np.arange(2, n + 1, dtype=float)
    y = x[1:] - shape_function(j, n)
    odd_mask = (j.astype(int) % 2) == 1
    even_mask = ~odd_mask
    return y[odd_mask], y[even_mask], j[odd_mask], j[even_mask]


def _uf_three_way_y(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    j = np.arange(3, n + 1, dtype=float)
    y = x[2:] - 2.0 * x[1] * np.sin(2.0 * np.pi * x[0] + j * np.pi / n)
    int_j = j.astype(int)
    return (
        y[(int_j - 1) % 3 == 0],
        y[(int_j - 2) % 3 == 0],
        y[int_j % 3 == 0],
    )


def uf1(x):
    x = np.asarray(x, dtype=float)
    y_odd, y_even, _, _ = _uf_odd_even_y(
        x,
        lambda j, n: np.sin(6.0 * np.pi * x[0] + j * np.pi / n),
    )
    return np.array(
        [
            x[0] + 2.0 * np.mean(y_odd**2),
            1.0 - np.sqrt(x[0]) + 2.0 * np.mean(y_even**2),
        ],
        dtype=float,
    )


def uf2(x):
    x = np.asarray(x, dtype=float)

    def shape(j, n):
        base = 0.3 * x[0] * (x[0] * np.cos(24.0 * np.pi * x[0] + 4.0 * j * np.pi / n) + 2.0)
        angle = 6.0 * np.pi * x[0] + j * np.pi / n
        return np.where((j.astype(int) % 2) == 1, base * np.cos(angle), base * np.sin(angle))

    y_odd, y_even, _, _ = _uf_odd_even_y(x, shape)
    return np.array(
        [
            x[0] + 2.0 * np.mean(y_odd**2),
            1.0 - np.sqrt(x[0]) + 2.0 * np.mean(y_even**2),
        ],
        dtype=float,
    )


def uf3(x):
    x = np.asarray(x, dtype=float)
    n = len(x)

    def shape(j, _):
        return x[0] ** (0.5 * (1.0 + 3.0 * (j - 2.0) / (n - 2.0)))

    y_odd, y_even, j_odd, j_even = _uf_odd_even_y(x, shape)
    p_odd = np.cos(20.0 * y_odd * np.pi / np.sqrt(j_odd))
    p_even = np.cos(20.0 * y_even * np.pi / np.sqrt(j_even))
    return np.array(
        [
            x[0] + 2.0 * (4.0 * np.sum(y_odd**2) - 2.0 * np.prod(p_odd) + 2.0) / len(y_odd),
            1.0
            - np.sqrt(x[0])
            + 2.0 * (4.0 * np.sum(y_even**2) - 2.0 * np.prod(p_even) + 2.0) / len(y_even),
        ],
        dtype=float,
    )


def uf4(x):
    x = np.asarray(x, dtype=float)
    y_odd, y_even, _, _ = _uf_odd_even_y(
        x,
        lambda j, n: np.sin(6.0 * np.pi * x[0] + j * np.pi / n),
    )
    h_odd = np.abs(y_odd) / (1.0 + np.exp(2.0 * np.abs(y_odd)))
    h_even = np.abs(y_even) / (1.0 + np.exp(2.0 * np.abs(y_even)))
    return np.array(
        [
            x[0] + 2.0 * np.mean(h_odd),
            1.0 - x[0] ** 2 + 2.0 * np.mean(h_even),
        ],
        dtype=float,
    )


def uf5(x):
    x = np.asarray(x, dtype=float)
    y_odd, y_even, _, _ = _uf_odd_even_y(
        x,
        lambda j, n: np.sin(6.0 * np.pi * x[0] + j * np.pi / n),
    )
    n_peaks = 10.0
    epsilon = 0.1
    h = (0.5 / n_peaks + epsilon) * abs(np.sin(2.0 * n_peaks * np.pi * x[0]))
    h_odd = 2.0 * y_odd**2 - np.cos(4.0 * np.pi * y_odd) + 1.0
    h_even = 2.0 * y_even**2 - np.cos(4.0 * np.pi * y_even) + 1.0
    return np.array(
        [
            x[0] + h + 2.0 * np.mean(h_odd),
            1.0 - x[0] + h + 2.0 * np.mean(h_even),
        ],
        dtype=float,
    )


def uf6(x):
    x = np.asarray(x, dtype=float)
    y_odd, y_even, j_odd, j_even = _uf_odd_even_y(
        x,
        lambda j, n: np.sin(6.0 * np.pi * x[0] + j * np.pi / n),
    )
    p_odd = np.cos(20.0 * y_odd * np.pi / np.sqrt(j_odd))
    p_even = np.cos(20.0 * y_even * np.pi / np.sqrt(j_even))
    n_peaks = 2.0
    epsilon = 0.1
    h = 2.0 * (0.5 / n_peaks + epsilon) * np.sin(2.0 * n_peaks * np.pi * x[0])
    h = max(0.0, h)
    return np.array(
        [
            x[0] + h + 2.0 * (4.0 * np.sum(y_odd**2) - 2.0 * np.prod(p_odd) + 2.0) / len(y_odd),
            1.0
            - x[0]
            + h
            + 2.0 * (4.0 * np.sum(y_even**2) - 2.0 * np.prod(p_even) + 2.0) / len(y_even),
        ],
        dtype=float,
    )


def uf7(x):
    x = np.asarray(x, dtype=float)
    y_odd, y_even, _, _ = _uf_odd_even_y(
        x,
        lambda j, n: np.sin(6.0 * np.pi * x[0] + j * np.pi / n),
    )
    y = x[0] ** 0.2
    return np.array(
        [
            y + 2.0 * np.mean(y_odd**2),
            1.0 - y + 2.0 * np.mean(y_even**2),
        ],
        dtype=float,
    )


def uf8(x):
    x = np.asarray(x, dtype=float)
    y1, y2, y3 = _uf_three_way_y(x)
    return np.array(
        [
            np.cos(0.5 * np.pi * x[0]) * np.cos(0.5 * np.pi * x[1]) + 2.0 * np.mean(y1**2),
            np.cos(0.5 * np.pi * x[0]) * np.sin(0.5 * np.pi * x[1]) + 2.0 * np.mean(y2**2),
            np.sin(0.5 * np.pi * x[0]) + 2.0 * np.mean(y3**2),
        ],
        dtype=float,
    )


def uf9(x):
    x = np.asarray(x, dtype=float)
    y1, y2, y3 = _uf_three_way_y(x)
    epsilon = 0.1
    h = max(0.0, (1.0 + epsilon) * (1.0 - 4.0 * (2.0 * x[0] - 1.0) ** 2))
    return np.array(
        [
            0.5 * (h + 2.0 * x[0]) * x[1] + 2.0 * np.mean(y1**2),
            0.5 * (h - 2.0 * x[0] + 2.0) * x[1] + 2.0 * np.mean(y2**2),
            1.0 - x[1] + 2.0 * np.mean(y3**2),
        ],
        dtype=float,
    )


def uf10(x):
    x = np.asarray(x, dtype=float)
    y1, y2, y3 = _uf_three_way_y(x)
    h1 = 4.0 * y1**2 - np.cos(8.0 * np.pi * y1) + 1.0
    h2 = 4.0 * y2**2 - np.cos(8.0 * np.pi * y2) + 1.0
    h3 = 4.0 * y3**2 - np.cos(8.0 * np.pi * y3) + 1.0
    return np.array(
        [
            np.cos(0.5 * np.pi * x[0]) * np.cos(0.5 * np.pi * x[1]) + 2.0 * np.mean(h1),
            np.cos(0.5 * np.pi * x[0]) * np.sin(0.5 * np.pi * x[1]) + 2.0 * np.mean(h2),
            np.sin(0.5 * np.pi * x[0]) + 2.0 * np.mean(h3),
        ],
        dtype=float,
    )


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
    _benchmark("ZDT1", "ZDT1 bi-objective function", zdt1, [(0.0, 1.0)] * 30, [1.1, 1.1]),
    _benchmark("ZDT2", "ZDT2 bi-objective function", zdt2, [(0.0, 1.0)] * 30, [1.1, 1.1]),
    _benchmark("ZDT3", "ZDT3 bi-objective function", zdt3, [(0.0, 1.0)] * 30, [1.1, 1.1]),
    _benchmark("ZDT4", "ZDT4 bi-objective function", zdt4, [(0.0, 1.0)] + [(-5.0, 5.0)] * 9, [1.1, 120.0]),
    _benchmark("ZDT6", "ZDT6 bi-objective function", zdt6, [(0.0, 1.0)] * 10, [1.1, 1.1]),
]


CEC2009_UF_BENCHMARKS = [
    _benchmark("UF1", "CEC2009 UF1 bi-objective function", uf1, [(0.0, 1.0)] + [(-1.0, 1.0)] * 29, [1.1, 1.1]),
    _benchmark("UF2", "CEC2009 UF2 bi-objective function", uf2, [(0.0, 1.0)] + [(-1.0, 1.0)] * 29, [1.1, 1.1]),
    _benchmark("UF3", "CEC2009 UF3 bi-objective function", uf3, [(0.0, 1.0)] * 30, [1.1, 1.1]),
    _benchmark("UF4", "CEC2009 UF4 bi-objective function", uf4, [(0.0, 1.0)] + [(-2.0, 2.0)] * 29, [1.1, 1.1]),
    _benchmark("UF5", "CEC2009 UF5 bi-objective function", uf5, [(0.0, 1.0)] + [(-1.0, 1.0)] * 29, [1.1, 1.1]),
    _benchmark("UF6", "CEC2009 UF6 bi-objective function", uf6, [(0.0, 1.0)] + [(-1.0, 1.0)] * 29, [1.1, 1.1]),
    _benchmark("UF7", "CEC2009 UF7 bi-objective function", uf7, [(0.0, 1.0)] + [(-1.0, 1.0)] * 29, [1.1, 1.1]),
    _benchmark(
        "UF8",
        "CEC2009 UF8 tri-objective function",
        uf8,
        [(0.0, 1.0), (0.0, 1.0)] + [(-2.0, 2.0)] * 28,
        [1.1, 1.1, 1.1],
    ),
    _benchmark(
        "UF9",
        "CEC2009 UF9 tri-objective function",
        uf9,
        [(0.0, 1.0), (0.0, 1.0)] + [(-2.0, 2.0)] * 28,
        [1.1, 1.1, 1.1],
    ),
    _benchmark(
        "UF10",
        "CEC2009 UF10 tri-objective function",
        uf10,
        [(0.0, 1.0), (0.0, 1.0)] + [(-2.0, 2.0)] * 28,
        [1.1, 1.1, 1.1],
    ),
]


CEC2020_MMO_BENCHMARKS = [
    _benchmark("MMF1", "CEC2020 MMO MMF1 bi-objective function", mmf1, [(1.0, 3.0), (-1.0, 1.0)], [1.1, 1.1]),
    _benchmark("MMF2", "CEC2020 MMO MMF2 bi-objective function", mmf2, [(0.0, 1.0), (0.0, 2.0)], [1.1, 1.1]),
    _benchmark("MMF4", "CEC2020 MMO MMF4 bi-objective function", mmf4, [(-1.0, 1.0), (0.0, 2.0)], [1.1, 1.1]),
    _benchmark("MMF5", "CEC2020 MMO MMF5 bi-objective function", mmf5, [(1.0, 3.0), (-1.0, 3.0)], [1.1, 1.1]),
    _benchmark("MMF7", "CEC2020 MMO MMF7 bi-objective function", mmf7, [(1.0, 3.0), (-1.0, 1.0)], [1.1, 1.1]),
    _benchmark("MMF8", "CEC2020 MMO MMF8 bi-objective function", mmf8, [(-np.pi, np.pi), (0.0, 9.0)], [1.1, 1.1]),
    _benchmark("MMF10", "CEC2020 MMO MMF10 bi-objective function", mmf10, [(0.1, 1.1), (0.1, 1.1)], [1.21, 13.2]),
    _benchmark("MMF11", "CEC2020 MMO MMF11 bi-objective function", mmf11, [(0.1, 1.1), (0.1, 1.1)], [1.21, 15.4]),
    _benchmark("MMF12", "CEC2020 MMO MMF12 bi-objective function", mmf12, [(0.0, 1.0), (0.0, 1.0)], [1.54, 1.1]),
    _benchmark(
        "MMF13",
        "CEC2020 MMO MMF13 bi-objective function",
        mmf13,
        [(0.1, 1.1), (0.1, 1.1), (0.1, 1.1)],
        [1.54, 15.4],
    ),
]
