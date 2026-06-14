# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np


PI = np.pi
DATA_DIR = (
    Path(__file__).resolve().parent
    / "official_cec2017"
    / "input_data"
)

_CACHE = {}


def _load_shift(func_num, dimension):
    key = ("shift", func_num, dimension)
    if key not in _CACHE:
        data = np.loadtxt(DATA_DIR / f"shift_data_{func_num}.txt", dtype=float).ravel()
        _CACHE[key] = data[:dimension]
    return _CACHE[key]


def _load_rotation(func_num, dimension):
    key = ("rotation", func_num, dimension)
    if key not in _CACHE:
        data = np.loadtxt(DATA_DIR / f"M_{func_num}_D{dimension}.txt", dtype=float)
        _CACHE[key] = data.reshape(dimension, dimension)
    return _CACHE[key]


def _shift_rotate(x, func_num, shift_rate):
    x = np.asarray(x, dtype=float)
    dimension = len(x)
    shift = _load_shift(func_num, dimension)
    rotation = _load_rotation(func_num, dimension)
    return rotation @ ((x - shift) * shift_rate)


def cec2017_f1_bent_cigar(x):
    z = _shift_rotate(x, func_num=1, shift_rate=1.0)
    return z[0] ** 2 + 1.0e6 * np.sum(z[1:] ** 2) + 100.0


def cec2017_f3_zakharov(x):
    z = _shift_rotate(x, func_num=3, shift_rate=1.0)
    index = np.arange(1, len(z) + 1)
    sum1 = np.sum(z**2)
    sum2 = np.sum(0.5 * index * z)
    return sum1 + sum2**2 + sum2**4 + 300.0


def cec2017_f4_rosenbrock(x):
    z = _shift_rotate(x, func_num=4, shift_rate=2.048 / 100.0)
    z = z + 1.0
    return np.sum(100.0 * (z[:-1] ** 2 - z[1:]) ** 2 + (z[:-1] - 1.0) ** 2) + 400.0


def cec2017_f5_rastrigin(x):
    z = _shift_rotate(x, func_num=5, shift_rate=5.12 / 100.0)
    return np.sum(z**2 - 10.0 * np.cos(2.0 * PI * z) + 10.0) + 500.0


def cec2017_f6_schaffer_f7(x):
    y = np.asarray(x, dtype=float) - _load_shift(func_num=6, dimension=len(x))
    rotation = _load_rotation(func_num=6, dimension=len(x))
    z = rotation @ y
    pair_radius = np.sqrt(z[:-1] ** 2 + z[1:] ** 2)
    terms = np.sqrt(pair_radius) * (1.0 + np.sin(50.0 * pair_radius**0.2) ** 2)
    return (np.sum(terms) / (len(x) - 1)) ** 2 + 600.0


OFFICIAL_CEC2017_BENCHMARKS = [
    {
        "id": "CEC2017_F1",
        "name": "CEC2017 F1 平移旋转 Bent Cigar 函数",
        "function": cec2017_f1_bent_cigar,
        "optimal_value": 100.0,
    },
    {
        "id": "CEC2017_F3",
        "name": "CEC2017 F3 平移旋转 Zakharov 函数",
        "function": cec2017_f3_zakharov,
        "optimal_value": 300.0,
    },
    {
        "id": "CEC2017_F4",
        "name": "CEC2017 F4 平移旋转 Rosenbrock 函数",
        "function": cec2017_f4_rosenbrock,
        "optimal_value": 400.0,
    },
    {
        "id": "CEC2017_F5",
        "name": "CEC2017 F5 平移旋转 Rastrigin 函数",
        "function": cec2017_f5_rastrigin,
        "optimal_value": 500.0,
    },
    {
        "id": "CEC2017_F6",
        "name": "CEC2017 F6 平移旋转 Schaffer's F7 函数",
        "function": cec2017_f6_schaffer_f7,
        "optimal_value": 600.0,
    },
]
