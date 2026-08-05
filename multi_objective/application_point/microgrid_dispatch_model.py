# -*- coding: utf-8 -*-
"""Microgrid dispatch model for the multi-objective application case."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


HOURS = 24

LOAD_KW = np.array(
    [42, 40, 38, 37, 39, 45, 52, 60, 68, 72, 70, 66,
     64, 62, 65, 71, 80, 92, 96, 88, 78, 68, 56, 48],
    dtype=float,
)
PV_KW = np.array(
    [0, 0, 0, 0, 0, 0, 3, 8, 15, 24, 32, 38,
     40, 36, 28, 18, 8, 2, 0, 0, 0, 0, 0, 0],
    dtype=float,
)
WT_KW = np.array(
    [28, 30, 32, 34, 36, 38, 35, 33, 30, 28, 26, 24,
     22, 20, 18, 16, 18, 22, 26, 30, 34, 36, 33, 30],
    dtype=float,
)

BUY_PRICE = np.array(
    [0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.68, 0.68, 0.68, 1.05, 1.05, 1.05,
     1.05, 1.05, 1.05, 0.68, 0.68, 1.05, 1.05, 1.05, 0.68, 0.68, 0.68, 0.38],
    dtype=float,
)
SELL_PRICE = np.array(
    [0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.45, 0.45, 0.45, 0.70, 0.70, 0.70,
     0.70, 0.70, 0.70, 0.45, 0.45, 0.70, 0.70, 0.70, 0.45, 0.45, 0.45, 0.30],
    dtype=float,
)

DIESEL_EMISSION_FACTOR = np.array([0.724, 0.0036, 0.0015], dtype=float)
GRID_EMISSION_FACTOR = np.array([0.997, 0.0045, 0.0018], dtype=float)
TREATMENT_COST = np.array([0.023, 6.0, 8.0], dtype=float)


@dataclass(frozen=True)
class MicrogridParams:
    diesel_min_kw: float = 10.0
    diesel_max_kw: float = 65.0
    battery_charge_max_kw: float = 30.0
    battery_discharge_max_kw: float = 30.0
    grid_buy_max_kw: float = 80.0
    grid_sell_max_kw: float = 60.0
    battery_capacity_kwh: float = 120.0
    soc_initial: float = 0.5
    soc_min: float = 0.2
    soc_max: float = 0.9
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    diesel_om_cost: float = 0.05
    diesel_fuel_cost: float = 0.45
    battery_om_cost: float = 0.02
    penalty_weight: float = 1.0e5

    @property
    def diesel_unit_cost(self) -> float:
        return self.diesel_om_cost + self.diesel_fuel_cost


PARAMS = MicrogridParams()
BOUNDS = (
    [(PARAMS.diesel_min_kw, PARAMS.diesel_max_kw)] * HOURS
    + [(-PARAMS.battery_charge_max_kw, PARAMS.battery_discharge_max_kw)] * HOURS
)


def split_solution(solution):
    values = np.asarray(solution, dtype=float)
    if values.size != 2 * HOURS:
        raise ValueError(f"Expected {2 * HOURS} decision variables, got {values.size}.")
    return values[:HOURS], values[HOURS:]


def repair_battery_power(raw_battery_kw, params=PARAMS):
    repaired = np.zeros(HOURS, dtype=float)
    energy = params.soc_initial * params.battery_capacity_kwh
    min_energy = params.soc_min * params.battery_capacity_kwh
    max_energy = params.soc_max * params.battery_capacity_kwh
    initial_energy = energy

    for hour in range(HOURS):
        remaining_hours = HOURS - hour - 1
        requested_power = raw_battery_kw[hour]

        max_discharge = min(
            params.battery_discharge_max_kw,
            max(0.0, (energy - min_energy) * params.discharge_efficiency),
        )
        max_charge = min(
            params.battery_charge_max_kw,
            max(0.0, (max_energy - energy) / params.charge_efficiency),
        )
        power = float(np.clip(requested_power, -max_charge, max_discharge))

        min_reachable_energy = max(
            min_energy,
            initial_energy - remaining_hours * params.battery_charge_max_kw * params.charge_efficiency,
        )
        max_reachable_energy = min(
            max_energy,
            initial_energy + remaining_hours * params.battery_discharge_max_kw / params.discharge_efficiency,
        )
        next_energy = energy - power / params.discharge_efficiency if power >= 0 else energy + -power * params.charge_efficiency

        if next_energy < min_reachable_energy:
            power = (
                (energy - min_reachable_energy) * params.discharge_efficiency
                if min_reachable_energy <= energy
                else -(min_reachable_energy - energy) / params.charge_efficiency
            )
        elif next_energy > max_reachable_energy:
            power = (
                (energy - max_reachable_energy) * params.discharge_efficiency
                if max_reachable_energy <= energy
                else -(max_reachable_energy - energy) / params.charge_efficiency
            )

        power = float(np.clip(power, -max_charge, max_discharge))
        repaired[hour] = power
        energy = energy - power / params.discharge_efficiency if power >= 0 else energy + -power * params.charge_efficiency

    return repaired


def repair_diesel_power(raw_diesel_kw, battery_kw, params=PARAMS):
    net_demand = LOAD_KW - PV_KW - WT_KW - battery_kw
    lower = np.maximum(params.diesel_min_kw, net_demand - params.grid_buy_max_kw)
    upper = np.minimum(params.diesel_max_kw, net_demand + params.grid_sell_max_kw)
    lower = np.minimum(lower, upper)
    return np.clip(raw_diesel_kw, lower, upper)


def battery_energy_profile(battery_kw, params=PARAMS):
    energy = np.empty(HOURS + 1, dtype=float)
    energy[0] = params.soc_initial * params.battery_capacity_kwh
    for hour, power in enumerate(battery_kw):
        energy[hour + 1] = (
            energy[hour] - power / params.discharge_efficiency
            if power >= 0
            else energy[hour] + -power * params.charge_efficiency
        )
    return energy


def evaluate_dispatch(solution, params=PARAMS):
    raw_diesel_kw, raw_battery_kw = split_solution(solution)
    battery_kw = repair_battery_power(raw_battery_kw, params)
    diesel_kw = repair_diesel_power(raw_diesel_kw, battery_kw, params)
    grid_kw = LOAD_KW - PV_KW - WT_KW - diesel_kw - battery_kw
    energy_kwh = battery_energy_profile(battery_kw, params)
    soc = energy_kwh / params.battery_capacity_kwh

    buy_kw = np.maximum(grid_kw, 0.0)
    sell_kw = np.maximum(-grid_kw, 0.0)

    economic_cost = float(
        params.diesel_unit_cost * np.sum(diesel_kw)
        + params.battery_om_cost * np.sum(np.abs(battery_kw))
        + np.sum(BUY_PRICE * buy_kw - SELL_PRICE * sell_kw)
    )

    diesel_environment_unit = float(np.dot(DIESEL_EMISSION_FACTOR, TREATMENT_COST))
    grid_environment_unit = float(np.dot(GRID_EMISSION_FACTOR, TREATMENT_COST))
    environment_cost = float(diesel_environment_unit * np.sum(diesel_kw) + grid_environment_unit * np.sum(buy_kw))

    penalty = 0.0
    penalty += np.sum(np.maximum(buy_kw - params.grid_buy_max_kw, 0.0) ** 2)
    penalty += np.sum(np.maximum(sell_kw - params.grid_sell_max_kw, 0.0) ** 2)
    penalty += (soc[-1] - params.soc_initial) ** 2
    penalty_value = float(params.penalty_weight * penalty)

    return {
        "diesel_kw": diesel_kw,
        "battery_kw": battery_kw,
        "grid_kw": grid_kw,
        "pv_kw": PV_KW.copy(),
        "wt_kw": WT_KW.copy(),
        "load_kw": LOAD_KW.copy(),
        "energy_kwh": energy_kwh,
        "soc": soc,
        "economic_cost": economic_cost,
        "environment_cost": environment_cost,
        "penalty": penalty_value,
    }


def objective_function(solution):
    dispatch = evaluate_dispatch(solution)
    penalty = dispatch["penalty"]
    return np.array(
        [
            dispatch["economic_cost"] + penalty,
            dispatch["environment_cost"] + penalty,
        ],
        dtype=float,
    )
