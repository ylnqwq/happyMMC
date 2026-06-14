# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np


PI = np.pi
INF = 1.0e99
DATA_DIR = (
    Path(__file__).resolve().parent
    / "official_cec2022"
    / "Python-CEC2022"
    / "Python"
    / "input_data"
)

_CACHE = {}


def _load_shift(func_num, dimension):
    key = ("shift", func_num, dimension)
    if key not in _CACHE:
        data = np.loadtxt(DATA_DIR / f"shift_data_{func_num}.txt", dtype=float)
        data = np.asarray(data, dtype=float)
        if func_num < 9:
            _CACHE[key] = data.ravel()[:dimension]
        else:
            if data.ndim == 1:
                data = data.reshape(-1, dimension)
            _CACHE[key] = data[:, :dimension]
    return _CACHE[key]


def _load_rotation(func_num, dimension):
    key = ("rotation", func_num, dimension)
    if key not in _CACHE:
        data = np.loadtxt(DATA_DIR / f"M_{func_num}_D{dimension}.txt", dtype=float)
        flat = np.asarray(data, dtype=float).ravel()
        if func_num < 9:
            _CACHE[key] = flat.reshape(dimension, dimension)
        else:
            _CACHE[key] = flat.reshape(-1, dimension, dimension)
    return _CACHE[key]


def _load_shuffle(func_num, dimension):
    key = ("shuffle", func_num, dimension)
    if key not in _CACHE:
        data = np.loadtxt(DATA_DIR / f"shuffle_data_{func_num}_D{dimension}.txt", dtype=int)
        _CACHE[key] = np.asarray(data, dtype=int).ravel() - 1
    return _CACHE[key]


def _shift_rotate(x, shift=None, rotation=None, shift_rate=1.0, s_flag=True, r_flag=True):
    z = np.asarray(x, dtype=float).copy()
    if s_flag:
        z = z - np.asarray(shift, dtype=float)
    z = z * shift_rate
    if r_flag:
        z = np.asarray(rotation, dtype=float) @ z
    return z


def _ellips_func(x, shift=None, rotation=None, s_flag=True, r_flag=True):
    z = _shift_rotate(x, shift, rotation, 1.0, s_flag, r_flag)
    weights = 10.0 ** (6.0 * np.arange(len(z)) / (len(z) - 1))
    return np.sum(weights * z**2)


def _bent_cigar_func(x, shift=None, rotation=None, s_flag=True, r_flag=True):
    z = _shift_rotate(x, shift, rotation, 1.0, s_flag, r_flag)
    return z[0] ** 2 + 1.0e6 * np.sum(z[1:] ** 2)


def _discus_func(x, shift=None, rotation=None, s_flag=True, r_flag=True):
    z = _shift_rotate(x, shift, rotation, 1.0, s_flag, r_flag)
    return 1.0e6 * z[0] ** 2 + np.sum(z[1:] ** 2)


def _rosenbrock_func(x, shift=None, rotation=None, s_flag=True, r_flag=True):
    z = _shift_rotate(x, shift, rotation, 2.048 / 100.0, s_flag, r_flag) + 1.0
    return np.sum(100.0 * (z[:-1] ** 2 - z[1:]) ** 2 + (z[:-1] - 1.0) ** 2)


def _ackley_func(x, shift=None, rotation=None, s_flag=True, r_flag=True):
    z = _shift_rotate(x, shift, rotation, 1.0, s_flag, r_flag)
    sum1 = np.sum(z**2)
    sum2 = np.sum(np.cos(2.0 * PI * z))
    return np.e - 20.0 * np.exp(-0.2 * np.sqrt(sum1 / len(z))) - np.exp(sum2 / len(z)) + 20.0


def _griewank_func(x, shift=None, rotation=None, s_flag=True, r_flag=True):
    z = _shift_rotate(x, shift, rotation, 600.0 / 100.0, s_flag, r_flag)
    indexes = np.arange(1, len(z) + 1)
    return 1.0 + np.sum(z**2) / 4000.0 - np.prod(np.cos(z / np.sqrt(indexes)))


def _rastrigin_func(x, shift=None, rotation=None, s_flag=True, r_flag=True):
    z = _shift_rotate(x, shift, rotation, 5.12 / 100.0, s_flag, r_flag)
    return np.sum(z**2 - 10.0 * np.cos(2.0 * PI * z) + 10.0)


def _schwefel_func(x, shift=None, rotation=None, s_flag=True, r_flag=True):
    z = _shift_rotate(x, shift, rotation, 1000.0 / 100.0, s_flag, r_flag)
    z = z + 420.9687462275036
    total = 0.0

    for value in z:
        if value > 500.0:
            total -= (500.0 - np.fmod(value, 500.0)) * np.sin(np.sqrt(500.0 - np.fmod(value, 500.0)))
            total += ((value - 500.0) / 100.0) ** 2 / len(z)
        elif value < -500.0:
            total -= (-500.0 + np.fmod(abs(value), 500.0)) * np.sin(np.sqrt(500.0 - np.fmod(abs(value), 500.0)))
            total += ((value + 500.0) / 100.0) ** 2 / len(z)
        else:
            total -= value * np.sin(np.sqrt(abs(value)))

    return total + 418.9828872724338 * len(z)


def _grie_rosen_func(x, shift=None, rotation=None, s_flag=True, r_flag=True):
    z = _shift_rotate(x, shift, rotation, 5.0 / 100.0, s_flag, r_flag) + 1.0
    total = 0.0
    for i in range(len(z)):
        next_i = (i + 1) % len(z)
        temp = 100.0 * (z[i] ** 2 - z[next_i]) ** 2 + (z[i] - 1.0) ** 2
        total += temp**2 / 4000.0 - np.cos(temp) + 1.0
    return total


def _escaffer6_func(x, shift=None, rotation=None, s_flag=True, r_flag=True):
    z = _shift_rotate(x, shift, rotation, 1.0, s_flag, r_flag)
    total = 0.0
    for i in range(len(z)):
        next_i = (i + 1) % len(z)
        radius2 = z[i] ** 2 + z[next_i] ** 2
        temp1 = np.sin(np.sqrt(radius2)) ** 2
        temp2 = (1.0 + 0.001 * radius2) ** 2
        total += 0.5 + (temp1 - 0.5) / temp2
    return total


def _happycat_func(x, shift=None, rotation=None, s_flag=True, r_flag=True):
    z = _shift_rotate(x, shift, rotation, 5.0 / 100.0, s_flag, r_flag) - 1.0
    r2 = np.sum(z**2)
    sum_z = np.sum(z)
    return abs(r2 - len(z)) ** 0.25 + (0.5 * r2 + sum_z) / len(z) + 0.5


def _hgbat_func(x, shift=None, rotation=None, s_flag=True, r_flag=True):
    z = _shift_rotate(x, shift, rotation, 5.0 / 100.0, s_flag, r_flag) - 1.0
    r2 = np.sum(z**2)
    sum_z = np.sum(z)
    return abs(r2**2 - sum_z**2) ** 0.5 + (0.5 * r2 + sum_z) / len(z) + 0.5


def _schaffer_f7_func(x, shift=None, rotation=None, s_flag=True, r_flag=True, pair_source=None):
    if pair_source is not None:
        y = np.asarray(pair_source, dtype=float)[: len(x)]
    elif s_flag:
        y = (np.asarray(x, dtype=float) - np.asarray(shift, dtype=float))
    else:
        y = np.asarray(x, dtype=float)

    pair_radius = np.sqrt(y[:-1] ** 2 + y[1:] ** 2)
    terms = np.sqrt(pair_radius) * (1.0 + np.sin(50.0 * pair_radius**0.2) ** 2)
    return (np.sum(terms) / (len(y) - 1)) ** 2


def _step_rastrigin_func(x, shift, rotation):
    y = np.asarray(x, dtype=float).copy()
    mask = np.abs(y - shift) > 0.5
    y[mask] = shift[mask] + np.floor(2.0 * (y[mask] - shift[mask]) + 0.5) / 2.0
    return _rastrigin_func(y, shift, rotation, True, True)


def _levy_func(x, shift=None, rotation=None, s_flag=True, r_flag=True):
    z = _shift_rotate(x, shift, rotation, 1.0, s_flag, r_flag)
    w = 1.0 + z / 4.0
    term1 = np.sin(PI * w[0]) ** 2
    term3 = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * PI * w[-1]) ** 2)
    terms = (w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(PI * w[:-1] + 1.0) ** 2)
    return term1 + np.sum(terms) + term3


def _zakharov_func(x, shift=None, rotation=None, s_flag=True, r_flag=True):
    z = _shift_rotate(x, shift, rotation, 1.0, s_flag, r_flag)
    index = np.arange(1, len(z) + 1)
    sum1 = np.sum(z**2)
    sum2 = np.sum(0.5 * index * z)
    return sum1 + sum2**2 + sum2**4


def _katsuura_func(x, shift=None, rotation=None, s_flag=True, r_flag=True):
    z = _shift_rotate(x, shift, rotation, 5.0 / 100.0, s_flag, r_flag)
    dimension = len(z)
    product = 1.0
    tmp3 = dimension**1.2

    for i, value in enumerate(z):
        temp = 0.0
        for j in range(1, 33):
            tmp = 2.0**j * value
            temp += abs(tmp - np.floor(tmp + 0.5)) / (2.0**j)
        product *= (1.0 + (i + 1) * temp) ** (10.0 / tmp3)

    tmp1 = 10.0 / dimension / dimension
    return product * tmp1 - tmp1


def _split_groups(dimension, proportions):
    sizes = [int(np.ceil(p * dimension)) for p in proportions[:-1]]
    sizes.append(dimension - sum(sizes))
    starts = np.cumsum([0] + sizes[:-1])
    return [(start, start + size) for start, size in zip(starts, sizes)]


def _hf02(x):
    dimension = len(x)
    z = _shift_rotate(x, _load_shift(6, dimension), _load_rotation(6, dimension))
    y = z[_load_shuffle(6, dimension)]
    groups = _split_groups(dimension, [0.4, 0.4, 0.2])

    return (
        _bent_cigar_func(y[groups[0][0] : groups[0][1]], s_flag=False, r_flag=False)
        + _hgbat_func(y[groups[1][0] : groups[1][1]], s_flag=False, r_flag=False)
        + _rastrigin_func(y[groups[2][0] : groups[2][1]], s_flag=False, r_flag=False)
    )


def _hf10(x):
    dimension = len(x)
    z = _shift_rotate(x, _load_shift(7, dimension), _load_rotation(7, dimension))
    y = z[_load_shuffle(7, dimension)]
    groups = _split_groups(dimension, [0.1, 0.2, 0.2, 0.2, 0.1, 0.2])

    return (
        _hgbat_func(y[groups[0][0] : groups[0][1]], s_flag=False, r_flag=False)
        + _katsuura_func(y[groups[1][0] : groups[1][1]], s_flag=False, r_flag=False)
        + _ackley_func(y[groups[2][0] : groups[2][1]], s_flag=False, r_flag=False)
        + _rastrigin_func(y[groups[3][0] : groups[3][1]], s_flag=False, r_flag=False)
        + _schwefel_func(y[groups[4][0] : groups[4][1]], s_flag=False, r_flag=False)
        + _schaffer_f7_func(
            y[groups[5][0] : groups[5][1]],
            s_flag=False,
            r_flag=False,
            pair_source=y,
        )
    )


def _hf06(x):
    dimension = len(x)
    z = _shift_rotate(x, _load_shift(8, dimension), _load_rotation(8, dimension))
    y = z[_load_shuffle(8, dimension)]
    groups = _split_groups(dimension, [0.3, 0.2, 0.2, 0.1, 0.2])

    return (
        _katsuura_func(y[groups[0][0] : groups[0][1]], s_flag=False, r_flag=False)
        + _happycat_func(y[groups[1][0] : groups[1][1]], s_flag=False, r_flag=False)
        + _grie_rosen_func(y[groups[2][0] : groups[2][1]], s_flag=False, r_flag=False)
        + _schwefel_func(y[groups[3][0] : groups[3][1]], s_flag=False, r_flag=False)
        + _ackley_func(y[groups[4][0] : groups[4][1]], s_flag=False, r_flag=False)
    )


def _cf_cal(x, shifts, delta, bias, fit):
    weights = []
    for shift in shifts[: len(fit)]:
        distance2 = np.sum((np.asarray(x, dtype=float) - shift) ** 2)
        if np.isclose(distance2, 0.0):
            weights.append(INF)
        else:
            weights.append(distance2**-0.5 * np.exp(-distance2 / (2.0 * len(x) * delta[len(weights)] ** 2)))

    weights = np.asarray(weights, dtype=float)
    if np.max(weights) == 0.0:
        weights = np.ones_like(weights)

    fit = np.asarray(fit, dtype=float) + np.asarray(bias, dtype=float)
    return np.sum(weights / np.sum(weights) * fit)


def _cf01(x):
    dimension = len(x)
    shifts = _load_shift(9, dimension)
    rotations = _load_rotation(9, dimension)
    fit = [
        10000.0 * _rosenbrock_func(x, shifts[0], rotations[0]) / 1.0e4,
        10000.0 * _ellips_func(x, shifts[1], rotations[1]) / 1.0e10,
        10000.0 * _bent_cigar_func(x, shifts[2], rotations[2]) / 1.0e30,
        10000.0 * _discus_func(x, shifts[3], rotations[3]) / 1.0e10,
        10000.0 * _ellips_func(x, shifts[4], rotations[4], True, False) / 1.0e10,
    ]
    return _cf_cal(x, shifts, [10, 20, 30, 40, 50], [0, 200, 300, 100, 400], fit)


def _cf02(x):
    dimension = len(x)
    shifts = _load_shift(10, dimension)
    rotations = _load_rotation(10, dimension)
    fit = [
        _schwefel_func(x, shifts[0], rotations[0], True, False),
        _rastrigin_func(x, shifts[1], rotations[1]),
        _hgbat_func(x, shifts[2], rotations[2]),
    ]
    return _cf_cal(x, shifts, [20, 10, 10], [0, 200, 100], fit)


def _cf06(x):
    dimension = len(x)
    shifts = _load_shift(11, dimension)
    rotations = _load_rotation(11, dimension)
    fit = [
        10000.0 * _escaffer6_func(x, shifts[0], rotations[0]) / 2.0e7,
        _schwefel_func(x, shifts[1], rotations[1]),
        1000.0 * _griewank_func(x, shifts[2], rotations[2]) / 100.0,
        _rosenbrock_func(x, shifts[3], rotations[3]),
        10000.0 * _rastrigin_func(x, shifts[4], rotations[4]) / 1.0e3,
    ]
    return _cf_cal(x, shifts, [20, 20, 30, 30, 20], [0, 200, 300, 400, 200], fit)


def _cf07(x):
    dimension = len(x)
    shifts = _load_shift(12, dimension)
    rotations = _load_rotation(12, dimension)
    fit = [
        10000.0 * _hgbat_func(x, shifts[0], rotations[0]) / 1000.0,
        10000.0 * _rastrigin_func(x, shifts[1], rotations[1]) / 1.0e3,
        10000.0 * _schwefel_func(x, shifts[2], rotations[2]) / 4.0e3,
        10000.0 * _bent_cigar_func(x, shifts[3], rotations[3]) / 1.0e30,
        10000.0 * _ellips_func(x, shifts[4], rotations[4]) / 1.0e10,
        10000.0 * _escaffer6_func(x, shifts[5], rotations[5]) / 2.0e7,
    ]
    return _cf_cal(x, shifts, [10, 20, 30, 40, 50, 60], [0, 300, 500, 100, 400, 200], fit)


def cec2022_f1_zakharov(x):
    dimension = len(x)
    return _zakharov_func(x, _load_shift(1, dimension), _load_rotation(1, dimension)) + 300.0


def cec2022_f2_rosenbrock(x):
    dimension = len(x)
    return _rosenbrock_func(x, _load_shift(2, dimension), _load_rotation(2, dimension)) + 400.0


def cec2022_f3_schaffer_f7(x):
    dimension = len(x)
    return _schaffer_f7_func(x, _load_shift(3, dimension), _load_rotation(3, dimension)) + 600.0


def cec2022_f4_step_rastrigin(x):
    dimension = len(x)
    return _rastrigin_func(x, _load_shift(4, dimension), _load_rotation(4, dimension)) + 800.0


def cec2022_f5_levy(x):
    dimension = len(x)
    return _levy_func(x, _load_shift(5, dimension), _load_rotation(5, dimension)) + 900.0


def cec2022_f6_hybrid_2(x):
    return _hf02(x) + 1800.0


def cec2022_f7_hybrid_10(x):
    return _hf10(x) + 2000.0


def cec2022_f8_hybrid_6(x):
    return _hf06(x) + 2200.0


def cec2022_f9_composition_1(x):
    return _cf01(x) + 2300.0


def cec2022_f10_composition_2(x):
    return _cf02(x) + 2400.0


def cec2022_f11_composition_6(x):
    return _cf06(x) + 2600.0


def cec2022_f12_composition_7(x):
    return _cf07(x) + 2700.0


OFFICIAL_CEC2022_BENCHMARKS = [
    {
        "id": "CEC2022_F1",
        "name": "CEC2022 F1 平移旋转 Zakharov 函数",
        "function": cec2022_f1_zakharov,
        "optimal_value": 300.0,
    },
    {
        "id": "CEC2022_F2",
        "name": "CEC2022 F2 平移旋转 Rosenbrock 函数",
        "function": cec2022_f2_rosenbrock,
        "optimal_value": 400.0,
    },
    {
        "id": "CEC2022_F3",
        "name": "CEC2022 F3 平移旋转 Schaffer's F7 函数",
        "function": cec2022_f3_schaffer_f7,
        "optimal_value": 600.0,
    },
    {
        "id": "CEC2022_F4",
        "name": "CEC2022 F4 非连续 Rastrigin 函数",
        "function": cec2022_f4_step_rastrigin,
        "optimal_value": 800.0,
    },
    {
        "id": "CEC2022_F5",
        "name": "CEC2022 F5 平移旋转 Levy 函数",
        "function": cec2022_f5_levy,
        "optimal_value": 900.0,
    },
    {
        "id": "CEC2022_F6",
        "name": "CEC2022 F6 混合函数 2",
        "function": cec2022_f6_hybrid_2,
        "optimal_value": 1800.0,
    },
    {
        "id": "CEC2022_F7",
        "name": "CEC2022 F7 混合函数 10",
        "function": cec2022_f7_hybrid_10,
        "optimal_value": 2000.0,
    },
    {
        "id": "CEC2022_F8",
        "name": "CEC2022 F8 混合函数 6",
        "function": cec2022_f8_hybrid_6,
        "optimal_value": 2200.0,
    },
    {
        "id": "CEC2022_F9",
        "name": "CEC2022 F9 组合函数 1",
        "function": cec2022_f9_composition_1,
        "optimal_value": 2300.0,
    },
    {
        "id": "CEC2022_F10",
        "name": "CEC2022 F10 组合函数 2",
        "function": cec2022_f10_composition_2,
        "optimal_value": 2400.0,
    },
    {
        "id": "CEC2022_F11",
        "name": "CEC2022 F11 组合函数 6",
        "function": cec2022_f11_composition_6,
        "optimal_value": 2600.0,
    },
    {
        "id": "CEC2022_F12",
        "name": "CEC2022 F12 组合函数 7",
        "function": cec2022_f12_composition_7,
        "optimal_value": 2700.0,
    },
]
