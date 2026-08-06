# -*- coding: utf-8 -*-
"""Microgrid dispatch model for the multi-objective application case."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


HOURS = 24

LOAD_KW = np.array(
    [42, 39, 34, 33, 35, 42, 48, 52, 54, 60, 58, 56,
     55, 58, 62, 68, 78, 88, 92, 88, 76, 64, 54, 46],
    dtype=float,
)
PV_KW = np.array(
    [0, 0, 0, 0, 0, 3, 5, 12, 24, 36, 44, 50,
     52, 46, 36, 24, 12, 5, 0, 0, 0, 0, 0, 0],
    dtype=float,
)
WT_KW = np.array(
    [32, 31, 33, 35, 37, 38, 36, 35, 34, 32, 30, 28,
     26, 24, 22, 20, 22, 25, 28, 32, 34, 36, 34, 33],
    dtype=float,
)

BUY_PRICE = np.array(
    [0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.52, 0.52, 0.52, 0.60, 0.60, 0.60,
     0.60, 0.60, 0.60, 0.52, 0.52, 0.60, 0.60, 0.60, 0.52, 0.52, 0.52, 0.42],
    dtype=float,
)
SELL_PRICE = np.array(
    [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.32, 0.32, 0.32, 0.38, 0.38, 0.38,
     0.38, 0.38, 0.38, 0.32, 0.32, 0.38, 0.38, 0.38, 0.32, 0.32, 0.32, 0.25],
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
    [(0.0, PARAMS.diesel_max_kw)] * HOURS
    + [(-PARAMS.battery_charge_max_kw, PARAMS.battery_discharge_max_kw)] * HOURS
)


def split_solution(solution):
    values = np.asarray(solution, dtype=float)
    if values.size != 2 * HOURS:
        raise ValueError(f"Expected {2 * HOURS} decision variables, got {values.size}.")
    return values[:HOURS], values[HOURS:]


def renewable_surplus_profile():
    return np.maximum(WT_KW + PV_KW - LOAD_KW, 0.0)


def repair_battery_power(raw_battery_kw, params=PARAMS):
    repaired = np.zeros(HOURS, dtype=float)
    energy = params.soc_initial * params.battery_capacity_kwh
    min_energy = params.soc_min * params.battery_capacity_kwh
    max_energy = params.soc_max * params.battery_capacity_kwh
    initial_energy = energy
    renewable_surplus_kw = renewable_surplus_profile()

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
            renewable_surplus_kw[hour],
        )
        power = float(np.clip(requested_power, -max_charge, max_discharge))

        future_charge_capacity = float(
            np.sum(np.minimum(params.battery_charge_max_kw, renewable_surplus_kw[hour + 1:]))
            * params.charge_efficiency
        )
        min_reachable_energy = max(
            min_energy,
            initial_energy - future_charge_capacity,
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
    raw_diesel_kw = np.asarray(raw_diesel_kw, dtype=float)
    battery_charge_kw = np.maximum(-battery_kw, 0.0)
    battery_discharge_kw = np.maximum(battery_kw, 0.0)
    renewable_surplus_kw = renewable_surplus_profile()
    allowed_sell_kw = np.minimum(
        params.grid_sell_max_kw,
        np.maximum(renewable_surplus_kw - battery_charge_kw, 0.0),
    )

    net_demand = LOAD_KW - PV_KW - WT_KW - battery_kw
    lower = np.maximum(0.0, net_demand - params.grid_buy_max_kw)
    grid_upper = net_demand + params.grid_sell_max_kw
    surplus_upper = LOAD_KW - WT_KW - PV_KW + battery_charge_kw + allowed_sell_kw - battery_discharge_kw
    upper = np.minimum.reduce([np.full(HOURS, params.diesel_max_kw), grid_upper, surplus_upper])

    diesel_kw = np.zeros(HOURS, dtype=float)
    for hour in range(HOURS):
        hour_upper = max(0.0, float(upper[hour]))
        hour_lower = min(max(0.0, float(lower[hour])), hour_upper)

        if hour_upper < params.diesel_min_kw or raw_diesel_kw[hour] < params.diesel_min_kw:
            diesel_kw[hour] = 0.0
        else:
            hour_lower = max(hour_lower, params.diesel_min_kw)
            diesel_kw[hour] = float(np.clip(raw_diesel_kw[hour], hour_lower, hour_upper))
    return diesel_kw


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
    charge_kw = np.maximum(-battery_kw, 0.0)
    renewable_surplus_kw = renewable_surplus_profile()

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
    penalty += np.sum(np.maximum(sell_kw + charge_kw - renewable_surplus_kw, 0.0) ** 2)
    penalty += (soc[-1] - params.soc_initial) ** 2
    penalty_value = float(params.penalty_weight * penalty)

    return {
        "diesel_kw": diesel_kw,
        "battery_kw": battery_kw,
        "grid_kw": grid_kw,
        "renewable_surplus_kw": renewable_surplus_kw,
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
