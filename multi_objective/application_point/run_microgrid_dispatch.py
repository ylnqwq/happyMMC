# -*- coding: utf-8 -*-
"""Run the multi-objective microgrid dispatch application case."""

from __future__ import annotations

import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODULE_DIR.parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from experiment_utils import env_bool, env_csv, env_float, env_int, env_output_dir, format_float, print_progress
from multi_objective.algorithms import MOABC, MODE, MOEAD, MOIABC, MOPSO
from multi_objective.application_point.microgrid_dispatch_model import (
    BOUNDS,
    DIESEL_EMISSION_FACTOR,
    GRID_EMISSION_FACTOR,
    PARAMS,
    TREATMENT_COST,
    evaluate_dispatch,
    objective_function,
)
from multi_objective.mo_utils import (
    calculate_hypervolume,
    igd_metric,
    igd_plus_metric,
    non_dominated_mask,
    spacing_metric,
)
from multi_objective.application_point.plot_microgrid_results import (
    convergence_main_start_index,
    plot_cost_breakdown,
    plot_igd_comparison_figures,
    plot_power_dispatch,
    plot_storage_dispatch,
)


OUTPUT_DIR = env_output_dir("APP_OUTPUT_DIR", MODULE_DIR / "results", MODULE_DIR)
SAVE_PLOTS = env_bool("APP_SAVE_PLOTS", True)
RUN_TIMES = env_int("APP_RUN_TIMES", 30)
SEED_BASE = env_int("APP_SEED", 20260723)
PARALLEL_WORKERS = env_int("APP_WORKERS", 8)
HV_REFERENCE_VALUE = env_float("APP_HV_REFERENCE_VALUE", 1.1)

BEE = env_int("APP_BEE", 80)
MAX_ITER = env_int("APP_MAX_ITER", 800)
LIMIT = env_int("APP_LIMIT", 160)
ARCHIVE_SIZE = env_int("APP_ARCHIVE_SIZE", 100)
TOURNAMENT_SIZE = env_int("APP_TOURNAMENT_SIZE", 3)
ELITE_RATE = env_float("APP_ELITE_RATE", 0.25)
ELIMINATION_RATE = env_float("APP_ELIMINATION_RATE", 0.25)
ARCHIVE_GUIDANCE_RATE = env_float("APP_ARCHIVE_GUIDANCE_RATE", 0.40)
ENABLED_ALGORITHMS = env_csv("APP_ALGORITHMS", ["MOIABC", "MOABC", "MOPSO", "MOEA/D", "MO-DE"])

ALGORITHM_COLORS = {
    "MOIABC": "#d66b5f",
    "MOABC": "#4e8fc7",
    "MOPSO": "#a7dce0",
    "MOEA/D": "#7fa6d9",
    "MO-DE": "#f1b183",
}
ALGORITHM_MARKERS = {
    "MOIABC": "P",
    "MOABC": "D",
    "MOPSO": "^",
    "MOEA/D": "s",
    "MO-DE": "o",
}

ALGORITHM_CONFIGS = {
    "MOIABC": {
        "runner": MOIABC.multi_objective_iabc,
        "params": {
            "bee": BEE,
            "max_iter": MAX_ITER,
            "limit": LIMIT,
            "archive_size": ARCHIVE_SIZE,
            "tournament_size": TOURNAMENT_SIZE,
            "elite_rate": ELITE_RATE,
            "elimination_rate": ELIMINATION_RATE,
            "archive_guidance_rate": ARCHIVE_GUIDANCE_RATE,
        },
    },
    "MOABC": {
        "runner": MOABC.multi_objective_abc,
        "params": {
            "bee": BEE,
            "max_iter": MAX_ITER,
            "limit": LIMIT,
            "archive_size": ARCHIVE_SIZE,
        },
    },
    "MOPSO": {
        "runner": MOPSO.mopso,
        "params": {
            "swarm_size": BEE,
            "max_iter": MAX_ITER,
            "archive_size": ARCHIVE_SIZE,
            "inertia": 0.4,
            "cognitive": 1.5,
            "social": 1.5,
        },
    },
    "MOEA/D": {
        "runner": MOEAD.moead,
        "params": {
            "population_size": BEE,
            "max_iter": MAX_ITER,
            "archive_size": ARCHIVE_SIZE,
            "neighborhood_size": 20,
            "mutation_factor": 0.5,
            "crossover_rate": 0.9,
        },
    },
    "MO-DE": {
        "runner": MODE.mode,
        "params": {
            "population_size": BEE,
            "max_iter": MAX_ITER,
            "archive_size": ARCHIVE_SIZE,
            "mutation_factor": 0.5,
            "crossover_rate": 0.9,
        },
    },
}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
LEGEND_FONT_SIZE = 16


def algorithm_dir_name(algorithm_name):
    return algorithm_name.replace("/", "_").replace("\\", "_")


def raw_cost_objectives(archive_solutions):
    values = []
    for solution in archive_solutions:
        dispatch = evaluate_dispatch(solution)
        values.append([dispatch["economic_cost"], dispatch["environment_cost"]])
    return np.asarray(values, dtype=float)


def normalize_objectives(objectives, ideal_point, nadir_point):
    objectives = np.asarray(objectives, dtype=float)
    span = np.asarray(nadir_point, dtype=float) - np.asarray(ideal_point, dtype=float)
    span = np.where(np.isclose(span, 0.0), 1.0, span)
    return (objectives - ideal_point) / span


def build_empirical_reference_front(run_results):
    all_solutions = np.vstack([item["archive_solutions"] for item in run_results])
    all_objectives = np.vstack([item["archive_objectives"] for item in run_results])

    _, unique_indexes = np.unique(all_objectives, axis=0, return_index=True)
    unique_indexes = np.sort(unique_indexes)
    all_solutions = all_solutions[unique_indexes]
    all_objectives = all_objectives[unique_indexes]

    mask = non_dominated_mask(all_objectives)
    reference_solutions = all_solutions[mask]
    reference_objectives = all_objectives[mask]
    order = np.argsort(reference_objectives[:, 0])
    return reference_solutions[order], reference_objectives[order], all_objectives


def select_compromise_index(objectives):
    normalized = normalize_objectives(objectives, np.min(objectives, axis=0), np.max(objectives, axis=0))
    distances = np.linalg.norm(normalized, axis=1)
    return int(np.argmin(distances)), distances


def write_pareto_csv(path, archive_solutions, archive_objectives):
    compromise_index, ideal_distances = select_compromise_index(archive_objectives)
    fieldnames = [
        "index",
        "is_compromise",
        "economic_cost",
        "environment_cost",
        "penalty",
        "penalized_economic_objective",
        "penalized_environment_objective",
        "ideal_distance",
        "objective_sum",
        "final_soc",
        "solution",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, (solution, objectives) in enumerate(zip(archive_solutions, archive_objectives), start=1):
            dispatch = evaluate_dispatch(solution)
            writer.writerow(
                {
                    "index": index,
                    "is_compromise": "1" if index - 1 == compromise_index else "0",
                    "economic_cost": f"{dispatch['economic_cost']:.6f}",
                    "environment_cost": f"{dispatch['environment_cost']:.6f}",
                    "penalty": f"{dispatch['penalty']:.6f}",
                    "penalized_economic_objective": f"{objectives[0]:.6f}",
                    "penalized_environment_objective": f"{objectives[1]:.6f}",
                    "ideal_distance": f"{ideal_distances[index - 1]:.6f}",
                    "objective_sum": f"{np.sum(objectives):.6f}",
                    "final_soc": f"{dispatch['soc'][-1]:.6f}",
                    "solution": " ".join(f"{value:.6f}" for value in solution),
                }
            )


def write_all_run_archives_csv(path, run_results, algorithm_name=None):
    fieldnames = [
        "algorithm",
        "run",
        "seed",
        "index",
        "economic_cost",
        "environment_cost",
        "penalty",
        "penalized_economic_objective",
        "penalized_environment_objective",
        "objective_sum",
        "final_soc",
        "solution",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for run_result in run_results:
            current_algorithm = algorithm_name or run_result.get("algorithm", "")
            for index, (solution, objectives) in enumerate(
                zip(run_result["archive_solutions"], run_result["archive_objectives"]),
                start=1,
            ):
                dispatch = evaluate_dispatch(solution)
                writer.writerow(
                    {
                        "algorithm": current_algorithm,
                        "run": run_result["run"],
                        "seed": run_result["seed"],
                        "index": index,
                        "economic_cost": f"{dispatch['economic_cost']:.6f}",
                        "environment_cost": f"{dispatch['environment_cost']:.6f}",
                        "penalty": f"{dispatch['penalty']:.6f}",
                        "penalized_economic_objective": f"{objectives[0]:.6f}",
                        "penalized_environment_objective": f"{objectives[1]:.6f}",
                        "objective_sum": f"{np.sum(objectives):.6f}",
                        "final_soc": f"{dispatch['soc'][-1]:.6f}",
                        "solution": " ".join(f"{value:.6f}" for value in solution),
                    }
                )


def write_all_algorithm_archives_csv(path, algorithm_run_results):
    combined_run_results = []
    for algorithm_name, run_results in algorithm_run_results.items():
        for run_result in run_results:
            item = dict(run_result)
            item["algorithm"] = algorithm_name
            combined_run_results.append(item)
    write_all_run_archives_csv(path, combined_run_results)


def write_dispatch_csv(path, solution):
    dispatch = evaluate_dispatch(solution)
    fieldnames = [
        "hour",
        "time_h",
        "load_kw",
        "wind_available_kw",
        "pv_available_kw",
        "pv_kw",
        "wt_kw",
        "wind_curtail_kw",
        "pv_curtail_kw",
        "total_curtail_kw",
        "diesel_kw",
        "battery_kw",
        "battery_discharge_kw",
        "battery_charge_kw",
        "battery_charge_state",
        "grid_kw",
        "grid_buy_kw",
        "grid_sell_kw",
        "renewable_surplus_kw",
        "surplus_allocation_violation_kw",
        "diesel_ramp_up_violation_kw",
        "diesel_ramp_down_violation_kw",
        "battery_ramp_violation_kw",
        "grid_exchange_ramp_violation_kw",
        "battery_charge_bound_violation_kw",
        "battery_discharge_bound_violation_kw",
        "battery_mutual_exclusion_violation_kw",
        "soc",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for hour in range(len(dispatch["load_kw"])):
            grid_kw = dispatch["grid_kw"][hour]
            charge_kw = dispatch["battery_charge_kw"][hour]
            discharge_kw = dispatch["battery_discharge_kw"][hour]
            sell_kw = max(-grid_kw, 0.0)
            violation_kw = max(sell_kw + charge_kw - dispatch["renewable_surplus_kw"][hour], 0.0)
            writer.writerow(
                {
                    "hour": hour + 1,
                    "time_h": f"{dispatch['time_h'][hour]:.6f}",
                    "load_kw": f"{dispatch['load_kw'][hour]:.6f}",
                    "wind_available_kw": f"{dispatch['wind_available_kw'][hour]:.6f}",
                    "pv_available_kw": f"{dispatch['pv_available_kw'][hour]:.6f}",
                    "pv_kw": f"{dispatch['pv_kw'][hour]:.6f}",
                    "wt_kw": f"{dispatch['wt_kw'][hour]:.6f}",
                    "wind_curtail_kw": f"{dispatch['wind_curtail_kw'][hour]:.6f}",
                    "pv_curtail_kw": f"{dispatch['pv_curtail_kw'][hour]:.6f}",
                    "total_curtail_kw": f"{dispatch['total_curtail_kw'][hour]:.6f}",
                    "diesel_kw": f"{dispatch['diesel_kw'][hour]:.6f}",
                    "battery_kw": f"{dispatch['battery_kw'][hour]:.6f}",
                    "battery_discharge_kw": f"{discharge_kw:.6f}",
                    "battery_charge_kw": f"{charge_kw:.6f}",
                    "battery_charge_state": f"{dispatch['battery_charge_state'][hour]:.0f}",
                    "grid_kw": f"{grid_kw:.6f}",
                    "grid_buy_kw": f"{max(grid_kw, 0.0):.6f}",
                    "grid_sell_kw": f"{max(-grid_kw, 0.0):.6f}",
                    "renewable_surplus_kw": f"{dispatch['renewable_surplus_kw'][hour]:.6f}",
                    "surplus_allocation_violation_kw": f"{violation_kw:.6f}",
                    "diesel_ramp_up_violation_kw": f"{dispatch['diesel_ramp_up_violation_kw'][hour]:.6f}",
                    "diesel_ramp_down_violation_kw": f"{dispatch['diesel_ramp_down_violation_kw'][hour]:.6f}",
                    "battery_ramp_violation_kw": f"{dispatch['battery_ramp_violation_kw'][hour]:.6f}",
                    "grid_exchange_ramp_violation_kw": f"{dispatch['grid_exchange_ramp_violation_kw'][hour]:.6f}",
                    "battery_charge_bound_violation_kw": f"{dispatch['battery_charge_bound_violation_kw'][hour]:.6f}",
                    "battery_discharge_bound_violation_kw": f"{dispatch['battery_discharge_bound_violation_kw'][hour]:.6f}",
                    "battery_mutual_exclusion_violation_kw": f"{dispatch['battery_mutual_exclusion_violation_kw'][hour]:.6f}",
                    "soc": f"{dispatch['soc'][hour + 1]:.6f}",
                }
            )


def write_power_curves_csv(path, solution):
    dispatch = evaluate_dispatch(solution)
    fieldnames = [
        "hour",
        "time_h",
        "load_kw",
        "wind_available_kw",
        "pv_available_kw",
        "wind_kw",
        "pv_kw",
        "wind_curtail_kw",
        "pv_curtail_kw",
        "total_curtail_kw",
        "diesel_kw",
        "battery_kw",
        "battery_discharge_kw",
        "battery_charge_kw",
        "battery_charge_state",
        "grid_net_kw",
        "grid_buy_kw",
        "grid_sell_kw",
        "renewable_surplus_kw",
        "surplus_allocation_violation_kw",
        "diesel_ramp_up_violation_kw",
        "diesel_ramp_down_violation_kw",
        "battery_ramp_violation_kw",
        "grid_exchange_ramp_violation_kw",
        "battery_charge_bound_violation_kw",
        "battery_discharge_bound_violation_kw",
        "battery_mutual_exclusion_violation_kw",
        "total_generation_kw",
        "total_supply_kw",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for hour in range(len(dispatch["load_kw"])):
            battery_kw = dispatch["battery_kw"][hour]
            grid_kw = dispatch["grid_kw"][hour]
            battery_discharge_kw = dispatch["battery_discharge_kw"][hour]
            battery_charge_kw = dispatch["battery_charge_kw"][hour]
            grid_buy_kw = max(grid_kw, 0.0)
            grid_sell_kw = max(-grid_kw, 0.0)
            renewable_surplus_kw = dispatch["renewable_surplus_kw"][hour]
            violation_kw = max(grid_sell_kw + battery_charge_kw - renewable_surplus_kw, 0.0)
            total_generation_kw = dispatch["wt_kw"][hour] + dispatch["pv_kw"][hour] + dispatch["diesel_kw"][hour]
            total_supply_kw = total_generation_kw + battery_discharge_kw + grid_buy_kw
            writer.writerow(
                {
                    "hour": hour + 1,
                    "time_h": f"{dispatch['time_h'][hour]:.6f}",
                    "load_kw": f"{dispatch['load_kw'][hour]:.6f}",
                    "wind_available_kw": f"{dispatch['wind_available_kw'][hour]:.6f}",
                    "pv_available_kw": f"{dispatch['pv_available_kw'][hour]:.6f}",
                    "wind_kw": f"{dispatch['wt_kw'][hour]:.6f}",
                    "pv_kw": f"{dispatch['pv_kw'][hour]:.6f}",
                    "wind_curtail_kw": f"{dispatch['wind_curtail_kw'][hour]:.6f}",
                    "pv_curtail_kw": f"{dispatch['pv_curtail_kw'][hour]:.6f}",
                    "total_curtail_kw": f"{dispatch['total_curtail_kw'][hour]:.6f}",
                    "diesel_kw": f"{dispatch['diesel_kw'][hour]:.6f}",
                    "battery_kw": f"{battery_kw:.6f}",
                    "battery_discharge_kw": f"{battery_discharge_kw:.6f}",
                    "battery_charge_kw": f"{battery_charge_kw:.6f}",
                    "battery_charge_state": f"{dispatch['battery_charge_state'][hour]:.0f}",
                    "grid_net_kw": f"{grid_kw:.6f}",
                    "grid_buy_kw": f"{grid_buy_kw:.6f}",
                    "grid_sell_kw": f"{grid_sell_kw:.6f}",
                    "renewable_surplus_kw": f"{renewable_surplus_kw:.6f}",
                    "surplus_allocation_violation_kw": f"{violation_kw:.6f}",
                    "diesel_ramp_up_violation_kw": f"{dispatch['diesel_ramp_up_violation_kw'][hour]:.6f}",
                    "diesel_ramp_down_violation_kw": f"{dispatch['diesel_ramp_down_violation_kw'][hour]:.6f}",
                    "battery_ramp_violation_kw": f"{dispatch['battery_ramp_violation_kw'][hour]:.6f}",
                    "grid_exchange_ramp_violation_kw": f"{dispatch['grid_exchange_ramp_violation_kw'][hour]:.6f}",
                    "battery_charge_bound_violation_kw": f"{dispatch['battery_charge_bound_violation_kw'][hour]:.6f}",
                    "battery_discharge_bound_violation_kw": f"{dispatch['battery_discharge_bound_violation_kw'][hour]:.6f}",
                    "battery_mutual_exclusion_violation_kw": f"{dispatch['battery_mutual_exclusion_violation_kw'][hour]:.6f}",
                    "total_generation_kw": f"{total_generation_kw:.6f}",
                    "total_supply_kw": f"{total_supply_kw:.6f}",
                }
            )


def write_soc_curve_csv(path, solution):
    dispatch = evaluate_dispatch(solution)
    fieldnames = [
        "hour",
        "time_h",
        "battery_kw",
        "battery_charge_state",
        "energy_start_kwh",
        "energy_end_kwh",
        "soc_start",
        "soc_end",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for hour in range(len(dispatch["battery_kw"])):
            writer.writerow(
                {
                    "hour": hour + 1,
                    "time_h": f"{dispatch['time_h'][hour]:.6f}",
                    "battery_kw": f"{dispatch['battery_kw'][hour]:.6f}",
                    "battery_charge_state": f"{dispatch['battery_charge_state'][hour]:.0f}",
                    "energy_start_kwh": f"{dispatch['energy_kwh'][hour]:.6f}",
                    "energy_end_kwh": f"{dispatch['energy_kwh'][hour + 1]:.6f}",
                    "soc_start": f"{dispatch['soc'][hour]:.6f}",
                    "soc_end": f"{dispatch['soc'][hour + 1]:.6f}",
                }
            )


def read_numeric_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as file:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(file)]


def write_metrics_csv(path, metric_rows):
    fieldnames = ["run", "seed", "archive_size", "IGD", "IGD+", "HV", "best_sum", "spacing", "elapsed_seconds"]
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in metric_rows:
            writer.writerow(
                {
                    "run": row["run"],
                    "seed": row["seed"],
                    "archive_size": row["archive_size"],
                    "IGD": format_float(row["IGD"]),
                    "IGD+": format_float(row["IGD+"]),
                    "HV": format_float(row["HV"]),
                    "best_sum": format_float(row["best_sum"]),
                    "spacing": format_float(row["spacing"]),
                    "elapsed_seconds": f"{row['elapsed_seconds']:.6f}",
                }
            )


def write_metrics_summary_csv(path, summary):
    fieldnames = ["metric", "mean", "std", "best"]
    metric_names = ["IGD", "IGD+", "HV", "best_sum", "spacing"]
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for metric_name in metric_names:
            writer.writerow(
                {
                    "metric": metric_name,
                    "mean": format_float(summary[f"mean_{metric_name}"]),
                    "std": format_float(summary[f"std_{metric_name}"]),
                    "best": format_float(summary[f"best_{metric_name}"]),
                }
            )


def write_convergence_history_csv(path, run_results):
    fieldnames = ["run", "seed", "iteration", "best_sum"]
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in run_results:
            for iteration, best_sum in enumerate(item["history"]):
                writer.writerow(
                    {
                        "run": item["run"],
                        "seed": item["seed"],
                        "iteration": iteration,
                        "best_sum": format_float(best_sum),
                    }
                )


def write_average_convergence_history_csv(path, run_results):
    histories = np.asarray([item["history"] for item in run_results], dtype=float)
    ddof = 1 if len(histories) > 1 else 0
    fieldnames = ["iteration", "mean_best_sum", "std_best_sum", "min_best_sum", "max_best_sum"]
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for iteration in range(histories.shape[1]):
            values = histories[:, iteration]
            writer.writerow(
                {
                    "iteration": iteration,
                    "mean_best_sum": format_float(np.mean(values)),
                    "std_best_sum": format_float(np.std(values, ddof=ddof)),
                    "min_best_sum": format_float(np.min(values)),
                    "max_best_sum": format_float(np.max(values)),
                }
            )


def plot_pareto(path, selected_solutions, selected_objectives, reference_solutions=None, algorithm_name="MOIABC"):
    selected_raw = raw_cost_objectives(selected_solutions)

    plt.figure(figsize=(7, 5.5))
    if reference_solutions is not None and len(reference_solutions) > len(selected_solutions):
        reference_raw = raw_cost_objectives(reference_solutions)
        plt.scatter(
            reference_raw[:, 0],
            reference_raw[:, 1],
            s=18,
            alpha=0.35,
            label="经验参考前沿",
        )
    plt.scatter(selected_raw[:, 0], selected_raw[:, 1], s=30, alpha=0.85, label=f"{algorithm_name} 外部档案")

    compromise_index, _ = select_compromise_index(selected_objectives)
    plt.scatter(
        selected_raw[compromise_index, 0],
        selected_raw[compromise_index, 1],
        s=90,
        marker="*",
        label="折中方案",
    )
    plt.xlabel("经济成本 / 元")
    plt.ylabel("环境成本 / 元")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_history(path, history, algorithm_name="MOIABC"):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(history)), history, linewidth=1.6)
    plt.xlabel("迭代次数 / 次")
    plt.ylabel("最优目标和")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_algorithm_pareto_comparison(path, algorithm_outputs, reference_solutions):
    plt.figure(figsize=(8.4, 7.7))
    if reference_solutions is not None and len(reference_solutions) > 0:
        reference_raw = raw_cost_objectives(reference_solutions)
        plt.scatter(
            reference_raw[:, 0],
            reference_raw[:, 1],
            s=16,
            c="#9aa1a8",
            alpha=0.28,
            label="经验参考前沿",
        )

    for algorithm_name, output in algorithm_outputs.items():
        raw_objectives = raw_cost_objectives(output["selected_solutions"])
        plt.scatter(
            raw_objectives[:, 0],
            raw_objectives[:, 1],
            s=28,
            alpha=0.82,
            marker=ALGORITHM_MARKERS.get(algorithm_name, "o"),
            color=ALGORITHM_COLORS.get(algorithm_name),
            label=f"{algorithm_name} 外部档案",
        )
        compromise_index = output["compromise_index"]
        plt.scatter(
            raw_objectives[compromise_index, 0],
            raw_objectives[compromise_index, 1],
            s=95,
            marker="*",
            color=ALGORITHM_COLORS.get(algorithm_name),
            edgecolor="#222222",
            linewidth=0.5,
            label=f"{algorithm_name} 折中方案",
        )

    plt.xlabel("经济成本 / 元")
    plt.ylabel("环境成本 / 元")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(fontsize=12, ncol=2, loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_algorithm_convergence_comparison(path, algorithm_run_results):
    series = []
    for algorithm_name, run_results in algorithm_run_results.items():
        histories = np.asarray([item["history"] for item in run_results], dtype=float)
        iterations = np.arange(histories.shape[1])
        mean_values = np.mean(histories, axis=0)
        min_values = np.min(histories, axis=0)
        max_values = np.max(histories, axis=0)
        start_index = convergence_main_start_index(iterations, mean_values, max_values)
        series.append(
            {
                "name": algorithm_name,
                "iterations": iterations,
                "mean": mean_values,
                "min": min_values,
                "max": max_values,
                "start_index": start_index,
                "color": ALGORITHM_COLORS.get(algorithm_name),
            }
        )

    final_iteration = max(float(item["iterations"][-1]) for item in series)
    fig, axis = plt.subplots(figsize=(10, 7.2))
    main_y_values = []
    for item in series:
        iterations = item["iterations"]
        mask = np.ones_like(iterations, dtype=bool)
        color = item["color"]
        axis.plot(iterations[mask], item["mean"][mask], linewidth=1.8, color=color, label=f"{item['name']} 平均值")
        axis.fill_between(iterations[mask], item["min"][mask], item["max"][mask], color=color, alpha=0.14, label=f"{item['name']} 范围")
        main_y_values.extend(item["mean"][mask])
        main_y_values.extend(item["min"][mask])
        main_y_values.extend(item["max"][mask])

    main_y_values = np.array([value for value in main_y_values if value > 0.0], dtype=float)
    axis.set_xlim(0, final_iteration)
    axis.set_yscale("log")
    axis.set_ylim(float(np.min(main_y_values)) * 0.75, 1.0e9)
    axis.set_xlabel("迭代次数 / 次")
    axis.set_ylabel("最优目标和")
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(fontsize=12, loc="upper center", ncol=5, frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close()


def run_once(task):
    algorithm_name, run_index = task
    if algorithm_name not in ALGORITHM_CONFIGS:
        valid_names = ", ".join(ALGORITHM_CONFIGS)
        raise ValueError(f"Unknown algorithm: {algorithm_name}. Valid algorithms: {valid_names}")

    config = ALGORITHM_CONFIGS[algorithm_name]
    seed = SEED_BASE + run_index
    start_time = time.perf_counter()
    archive_solutions, archive_objectives, history, used_seed = config["runner"](
        objective_function=objective_function,
        bounds=BOUNDS,
        seed=seed,
        **config["params"],
    )
    elapsed_seconds = time.perf_counter() - start_time
    return {
        "algorithm": algorithm_name,
        "run": run_index + 1,
        "seed": used_seed,
        "archive_solutions": archive_solutions,
        "archive_objectives": archive_objectives,
        "history": history,
        "elapsed_seconds": elapsed_seconds,
    }


def calculate_run_metrics(run_results, reference_objectives, all_objectives):
    ideal_point = np.min(all_objectives, axis=0)
    nadir_point = np.max(all_objectives, axis=0)
    normalized_reference = normalize_objectives(reference_objectives, ideal_point, nadir_point)
    hypervolume_reference = np.full(reference_objectives.shape[1], HV_REFERENCE_VALUE, dtype=float)

    metric_rows = []
    for item in run_results:
        objectives = item["archive_objectives"]
        normalized_objectives = normalize_objectives(objectives, ideal_point, nadir_point)
        objective_sums = np.sum(objectives, axis=1)
        metric_rows.append(
            {
                "run": item["run"],
                "seed": item["seed"],
                "archive_size": int(len(objectives)),
                "IGD": igd_metric(normalized_objectives, normalized_reference),
                "IGD+": igd_plus_metric(normalized_objectives, normalized_reference),
                "HV": calculate_hypervolume(normalized_objectives, hypervolume_reference),
                "best_sum": float(np.min(objective_sums)),
                "spacing": spacing_metric(objectives),
                "elapsed_seconds": item["elapsed_seconds"],
            }
        )
    return metric_rows


def summarize_metrics(metric_rows):
    summary = {}
    ddof = 1 if len(metric_rows) > 1 else 0
    for metric_name in ["IGD", "IGD+", "HV", "best_sum", "spacing"]:
        values = np.array([row[metric_name] for row in metric_rows], dtype=float)
        summary[f"mean_{metric_name}"] = float(np.mean(values))
        summary[f"std_{metric_name}"] = float(np.std(values, ddof=ddof))
        summary[f"best_{metric_name}"] = float(np.max(values) if metric_name == "HV" else np.min(values))
    return summary


def select_best_run(metric_rows):
    igd_plus_values = np.array([row["IGD+"] for row in metric_rows], dtype=float)
    hv_values = np.array([row["HV"] for row in metric_rows], dtype=float)
    best_sum_values = np.array([row["best_sum"] for row in metric_rows], dtype=float)
    order = np.lexsort((best_sum_values, -hv_values, igd_plus_values))
    return int(order[0])


def run_all(algorithm_name):
    worker_count = min(PARALLEL_WORKERS, RUN_TIMES)
    tasks = [(algorithm_name, run_index) for run_index in range(RUN_TIMES)]
    if worker_count <= 1:
        run_results = []
        for index, task in enumerate(tasks, start=1):
            run_results.append(run_once(task))
            print_progress(index, RUN_TIMES, prefix=f"{algorithm_name} Progress")
        print()
        return run_results

    run_results = []
    completed = 0
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(run_once, task) for task in tasks]
        for future in as_completed(futures):
            run_results.append(future.result())
            completed += 1
            print_progress(completed, RUN_TIMES, prefix=f"{algorithm_name} Progress")
    print()
    return sorted(run_results, key=lambda item: item["run"])


def build_algorithm_output(algorithm_name, run_results, reference_solutions, reference_objectives, all_objectives):
    metric_rows = calculate_run_metrics(run_results, reference_objectives, all_objectives)
    metrics_summary = summarize_metrics(metric_rows)
    best_run_position = select_best_run(metric_rows)
    selected_result = run_results[best_run_position]
    selected_metrics = metric_rows[best_run_position]

    selected_solutions = selected_result["archive_solutions"]
    selected_objectives = selected_result["archive_objectives"]
    compromise_index, ideal_distances = select_compromise_index(selected_objectives)
    compromise_dispatch = evaluate_dispatch(selected_solutions[compromise_index])
    raw_objectives = raw_cost_objectives(selected_solutions)
    total_available_renewable_kwh = float(
        np.sum(compromise_dispatch["wind_available_kw"] + compromise_dispatch["pv_available_kw"])
        * PARAMS.time_step_hours
    )
    total_curtailment_kwh = float(np.sum(compromise_dispatch["total_curtail_kw"]) * PARAMS.time_step_hours)
    renewable_utilization_rate = (
        1.0 - total_curtailment_kwh / total_available_renewable_kwh
        if total_available_renewable_kwh > 0.0
        else 1.0
    )

    summary = {
        "run_times": RUN_TIMES,
        "seed_base": SEED_BASE,
        "algorithm": algorithm_name,
        "dispatch_periods": len(compromise_dispatch["load_kw"]),
        "decision_variables": len(BOUNDS),
        "time_step_hours": PARAMS.time_step_hours,
        "selected_run": selected_result["run"],
        "selected_seed": selected_result["seed"],
        "bee": BEE,
        "max_iter": MAX_ITER,
        "limit": LIMIT,
        "configured_archive_size": ARCHIVE_SIZE,
        "algorithm_parameters": dict(ALGORITHM_CONFIGS[algorithm_name]["params"]),
        "enable_curtailment": PARAMS.enable_curtailment,
        "enable_extra_complexity": PARAMS.enable_extra_complexity,
        "diesel_quadratic_fuel_cost": PARAMS.diesel_quadratic_fuel_cost,
        "diesel_quadratic_emission_cost": PARAMS.diesel_quadratic_emission_cost,
        "diesel_emission_factor": DIESEL_EMISSION_FACTOR.tolist(),
        "grid_emission_factor": GRID_EMISSION_FACTOR.tolist(),
        "treatment_cost": TREATMENT_COST.tolist(),
        "diesel_environment_unit_cost": float(np.dot(DIESEL_EMISSION_FACTOR, TREATMENT_COST)),
        "grid_environment_unit_cost": float(np.dot(GRID_EMISSION_FACTOR, TREATMENT_COST)),
        "diesel_ramp_up_kw_per_h": PARAMS.diesel_ramp_up_kw_per_h,
        "diesel_ramp_down_kw_per_h": PARAMS.diesel_ramp_down_kw_per_h,
        "diesel_ramp_up_limit_kw_per_period": PARAMS.diesel_ramp_up_limit_kw,
        "diesel_ramp_down_limit_kw_per_period": PARAMS.diesel_ramp_down_limit_kw,
        "battery_ramp_kw_per_h": PARAMS.battery_ramp_kw_per_h,
        "battery_ramp_limit_kw_per_period": PARAMS.battery_ramp_limit_kw,
        "grid_exchange_ramp_kw_per_h": PARAMS.grid_exchange_ramp_kw_per_h,
        "grid_exchange_ramp_limit_kw_per_period": PARAMS.grid_exchange_ramp_limit_kw,
        "selected_archive_size": int(len(selected_objectives)),
        "reference_front_size": int(len(reference_objectives)),
        "metrics": {
            "selected_run": selected_metrics,
            "summary": metrics_summary,
        },
        "min_economic_cost": float(np.min(raw_objectives[:, 0])),
        "min_environment_cost": float(np.min(raw_objectives[:, 1])),
        "compromise_index": compromise_index + 1,
        "compromise_ideal_distance": float(ideal_distances[compromise_index]),
        "compromise_economic_cost": compromise_dispatch["economic_cost"],
        "compromise_environment_cost": compromise_dispatch["environment_cost"],
        "compromise_penalized_economic_objective": float(selected_objectives[compromise_index, 0]),
        "compromise_penalized_environment_objective": float(selected_objectives[compromise_index, 1]),
        "compromise_penalty": compromise_dispatch["penalty"],
        "compromise_final_soc": float(compromise_dispatch["soc"][-1]),
        "compromise_wind_curtailment_kwh": float(np.sum(compromise_dispatch["wind_curtail_kw"]) * PARAMS.time_step_hours),
        "compromise_pv_curtailment_kwh": float(np.sum(compromise_dispatch["pv_curtail_kw"]) * PARAMS.time_step_hours),
        "compromise_total_curtailment_kwh": total_curtailment_kwh,
        "compromise_renewable_utilization_rate": float(renewable_utilization_rate),
    }
    return {
        "algorithm": algorithm_name,
        "run_results": run_results,
        "metric_rows": metric_rows,
        "metrics_summary": metrics_summary,
        "selected_result": selected_result,
        "selected_metrics": selected_metrics,
        "selected_solutions": selected_solutions,
        "selected_objectives": selected_objectives,
        "compromise_index": compromise_index,
        "compromise_dispatch": compromise_dispatch,
        "summary": summary,
    }


def write_algorithm_outputs(output_dir, algorithm_output, reference_solutions, reference_objectives, save_algorithm_plots=True):
    output_dir.mkdir(parents=True, exist_ok=True)
    algorithm_name = algorithm_output["algorithm"]
    selected_result = algorithm_output["selected_result"]
    selected_solutions = algorithm_output["selected_solutions"]
    selected_objectives = algorithm_output["selected_objectives"]
    compromise_index = algorithm_output["compromise_index"]

    (output_dir / "multi_objective_summary.json").write_text(
        json.dumps(algorithm_output["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_metrics_csv(output_dir / "multi_objective_metrics.csv", algorithm_output["metric_rows"])
    write_metrics_summary_csv(output_dir / "multi_objective_metrics_summary.csv", algorithm_output["metrics_summary"])
    write_convergence_history_csv(output_dir / "multi_objective_convergence_history.csv", algorithm_output["run_results"])
    write_average_convergence_history_csv(output_dir / "multi_objective_average_convergence_history.csv", algorithm_output["run_results"])
    write_all_run_archives_csv(
        output_dir / "multi_objective_all_run_archives.csv",
        algorithm_output["run_results"],
        algorithm_name=algorithm_name,
    )
    write_pareto_csv(output_dir / "multi_objective_pareto.csv", selected_solutions, selected_objectives)
    write_pareto_csv(output_dir / "multi_objective_reference_pareto.csv", reference_solutions, reference_objectives)
    write_dispatch_csv(output_dir / "multi_objective_compromise_dispatch.csv", selected_solutions[compromise_index])
    write_power_curves_csv(output_dir / "compromise_power_curves.csv", selected_solutions[compromise_index])
    write_soc_curve_csv(output_dir / "compromise_soc_curve.csv", selected_solutions[compromise_index])

    if SAVE_PLOTS and save_algorithm_plots:
        power_rows = read_numeric_csv(output_dir / "compromise_power_curves.csv")
        soc_rows = read_numeric_csv(output_dir / "compromise_soc_curve.csv")
        plot_cost_breakdown(power_rows, output_dir / "compromise_cost_breakdown.png", algorithm_name=algorithm_name)
        plot_power_dispatch(power_rows, output_dir / "compromise_power_dispatch.png", algorithm_name=algorithm_name)
        if algorithm_name == "MOIABC":
            plot_storage_dispatch(
                power_rows,
                soc_rows,
                output_dir / "compromise_storage_dispatch.png",
                algorithm_name=algorithm_name,
            )


def write_algorithm_comparison_csv(path, algorithm_outputs):
    fieldnames = [
        "algorithm",
        "selected_run",
        "selected_seed",
        "selected_archive_size",
        "reference_front_size",
        "IGD",
        "IGD+",
        "HV",
        "best_sum",
        "spacing",
        "mean_IGD",
        "mean_IGD+",
        "mean_HV",
        "mean_best_sum",
        "mean_spacing",
        "compromise_economic_cost",
        "compromise_environment_cost",
        "compromise_penalty",
        "compromise_final_soc",
        "compromise_total_curtailment_kwh",
        "compromise_renewable_utilization_rate",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for algorithm_name, output in algorithm_outputs.items():
            selected_metrics = output["selected_metrics"]
            summary = output["summary"]
            metric_summary = output["metrics_summary"]
            writer.writerow(
                {
                    "algorithm": algorithm_name,
                    "selected_run": summary["selected_run"],
                    "selected_seed": summary["selected_seed"],
                    "selected_archive_size": summary["selected_archive_size"],
                    "reference_front_size": summary["reference_front_size"],
                    "IGD": format_float(selected_metrics["IGD"]),
                    "IGD+": format_float(selected_metrics["IGD+"]),
                    "HV": format_float(selected_metrics["HV"]),
                    "best_sum": format_float(selected_metrics["best_sum"]),
                    "spacing": format_float(selected_metrics["spacing"]),
                    "mean_IGD": format_float(metric_summary["mean_IGD"]),
                    "mean_IGD+": format_float(metric_summary["mean_IGD+"]),
                    "mean_HV": format_float(metric_summary["mean_HV"]),
                    "mean_best_sum": format_float(metric_summary["mean_best_sum"]),
                    "mean_spacing": format_float(metric_summary["mean_spacing"]),
                    "compromise_economic_cost": format_float(summary["compromise_economic_cost"]),
                    "compromise_environment_cost": format_float(summary["compromise_environment_cost"]),
                    "compromise_penalty": format_float(summary["compromise_penalty"]),
                    "compromise_final_soc": format_float(summary["compromise_final_soc"]),
                    "compromise_total_curtailment_kwh": format_float(summary["compromise_total_curtailment_kwh"]),
                    "compromise_renewable_utilization_rate": format_float(summary["compromise_renewable_utilization_rate"]),
                }
            )


def select_primary_algorithm(algorithm_names):
    return "MOIABC" if "MOIABC" in algorithm_names else algorithm_names[0]


def clear_existing_png_files(output_dir):
    if not output_dir.exists():
        return
    for path in output_dir.rglob("*.png"):
        path.unlink()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clear_existing_png_files(OUTPUT_DIR)
    if RUN_TIMES <= 0:
        raise ValueError("APP_RUN_TIMES must be greater than 0.")
    if PARALLEL_WORKERS <= 0:
        raise ValueError("APP_WORKERS must be greater than 0.")

    algorithm_names = []
    for algorithm_name in ENABLED_ALGORITHMS:
        if algorithm_name not in ALGORITHM_CONFIGS:
            valid_names = ", ".join(ALGORITHM_CONFIGS)
            raise ValueError(f"Unknown APP_ALGORITHMS item: {algorithm_name}. Valid algorithms: {valid_names}")
        if algorithm_name not in algorithm_names:
            algorithm_names.append(algorithm_name)
    if not algorithm_names:
        raise ValueError("APP_ALGORITHMS must contain at least one algorithm.")

    print("=" * 80)
    print("Multi-objective microgrid dispatch configuration")
    print(f"Algorithms: {', '.join(algorithm_names)}")
    print(f"Runs per algorithm: {RUN_TIMES}, seed_base: {SEED_BASE}, bee: {BEE}, max_iter: {MAX_ITER}, archive_size: {ARCHIVE_SIZE}")
    print(f"Dispatch periods: {len(BOUNDS) // (4 if PARAMS.enable_curtailment else 2)}, time_step: {PARAMS.time_step_hours:g} h")
    print(f"Decision variables: {len(BOUNDS)}")
    print(f"Curtailment enabled: {PARAMS.enable_curtailment}")
    print(f"Diesel ramp limit: +{PARAMS.diesel_ramp_up_limit_kw:g}/-{PARAMS.diesel_ramp_down_limit_kw:g} kW per period")
    print(f"Extra complexity enabled: {PARAMS.enable_extra_complexity}")
    if PARAMS.enable_extra_complexity:
        print(f"Battery ramp limit: {PARAMS.battery_ramp_limit_kw:g} kW per period")
        print(f"Grid exchange ramp limit: {PARAMS.grid_exchange_ramp_limit_kw:g} kW per period")
        print(f"Diesel quadratic cost coefficients: fuel={PARAMS.diesel_quadratic_fuel_cost:g}, emission={PARAMS.diesel_quadratic_emission_cost:g}")
    print(f"Parallel workers: {min(PARALLEL_WORKERS, RUN_TIMES)}")

    algorithm_run_results = {}
    for algorithm_name in algorithm_names:
        print("-" * 80)
        print(f"Running {algorithm_name}")
        algorithm_run_results[algorithm_name] = run_all(algorithm_name)

    all_run_results = [item for run_results in algorithm_run_results.values() for item in run_results]
    reference_solutions, reference_objectives, all_objectives = build_empirical_reference_front(all_run_results)
    algorithm_outputs = {
        algorithm_name: build_algorithm_output(
            algorithm_name,
            run_results,
            reference_solutions,
            reference_objectives,
            all_objectives,
        )
        for algorithm_name, run_results in algorithm_run_results.items()
    }

    for algorithm_name, output in algorithm_outputs.items():
        write_algorithm_outputs(OUTPUT_DIR / algorithm_dir_name(algorithm_name), output, reference_solutions, reference_objectives)

    primary_algorithm = select_primary_algorithm(algorithm_names)
    write_algorithm_outputs(
        OUTPUT_DIR,
        algorithm_outputs[primary_algorithm],
        reference_solutions,
        reference_objectives,
        save_algorithm_plots=False,
    )
    write_algorithm_comparison_csv(OUTPUT_DIR / "microgrid_algorithm_comparison.csv", algorithm_outputs)
    write_all_algorithm_archives_csv(OUTPUT_DIR / "microgrid_all_algorithm_archives.csv", algorithm_run_results)
    (OUTPUT_DIR / "microgrid_algorithm_comparison_summary.json").write_text(
        json.dumps(
            {
                "algorithms": algorithm_names,
                "primary_algorithm": primary_algorithm,
                "run_times_per_algorithm": RUN_TIMES,
                "seed_base": SEED_BASE,
                "reference_front_size": int(len(reference_objectives)),
                "summaries": {name: output["summary"] for name, output in algorithm_outputs.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if SAVE_PLOTS:
        plot_algorithm_pareto_comparison(OUTPUT_DIR / "algorithm_pareto_comparison.png", algorithm_outputs, reference_solutions)
        plot_algorithm_convergence_comparison(
            OUTPUT_DIR / "algorithm_average_convergence_comparison.png",
            algorithm_run_results,
        )
        plot_igd_comparison_figures(
            {name: output["metric_rows"] for name, output in algorithm_outputs.items()},
            OUTPUT_DIR,
        )

    print("=" * 80)
    print("Multi-objective microgrid application case finished")
    print(f"Results: {OUTPUT_DIR}")
    print(f"Primary algorithm files at root: {primary_algorithm}")
    print(f"Reference front size: {len(reference_objectives)}")
    for algorithm_name, output in algorithm_outputs.items():
        selected_result = output["selected_result"]
        selected_metrics = output["selected_metrics"]
        compromise_dispatch = output["compromise_dispatch"]
        print("-" * 80)
        print(f"Algorithm: {algorithm_name}")
        print(f"Selected run: {selected_result['run']}, seed: {selected_result['seed']}")
        print(f"Selected archive size: {len(output['selected_objectives'])}")
        print(f"IGD: {selected_metrics['IGD']:.6e}")
        print(f"IGD+: {selected_metrics['IGD+']:.6e}")
        print(f"HV: {selected_metrics['HV']:.6e}")
        print(f"best_sum: {selected_metrics['best_sum']:.6e}")
        print(f"spacing: {selected_metrics['spacing']:.6e}")
        print(f"Compromise economic cost: {compromise_dispatch['economic_cost']:.6f}")
        print(f"Compromise environment cost: {compromise_dispatch['environment_cost']:.6f}")
        print(f"Compromise penalty: {compromise_dispatch['penalty']:.6f}")
        print(f"Compromise total curtailment: {output['summary']['compromise_total_curtailment_kwh']:.6f}")
        print(f"Compromise renewable utilization rate: {output['summary']['compromise_renewable_utilization_rate']:.6f}")


if __name__ == "__main__":
    main()
