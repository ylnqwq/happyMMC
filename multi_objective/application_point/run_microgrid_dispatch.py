# -*- coding: utf-8 -*-
"""Run the multi-objective microgrid dispatch application case."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODULE_DIR.parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from multi_objective.algorithms import MOIABC
from multi_objective.application_point.microgrid_dispatch_model import (
    BOUNDS,
    evaluate_dispatch,
    objective_function,
)


OUTPUT_DIR = MODULE_DIR / "results"
SAVE_PLOTS = os.environ.get("APP_SAVE_PLOTS", "1") != "0"
SEED = int(os.environ.get("APP_SEED", "20260625"))
BEE = int(os.environ.get("APP_BEE", "75"))
MAX_ITER = int(os.environ.get("APP_MAX_ITER", "750"))
LIMIT = int(os.environ.get("APP_LIMIT", "150"))
ARCHIVE_SIZE = int(os.environ.get("APP_ARCHIVE_SIZE", "100"))

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False


def write_pareto_csv(path, archive_solutions, archive_objectives):
    fieldnames = [
        "index",
        "economic_cost",
        "environment_cost",
        "penalty",
        "penalized_economic_objective",
        "penalized_environment_objective",
        "weighted_penalized_objective_0_5",
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
                    "economic_cost": f"{dispatch['economic_cost']:.6f}",
                    "environment_cost": f"{dispatch['environment_cost']:.6f}",
                    "penalty": f"{dispatch['penalty']:.6f}",
                    "penalized_economic_objective": f"{objectives[0]:.6f}",
                    "penalized_environment_objective": f"{objectives[1]:.6f}",
                    "weighted_penalized_objective_0_5": f"{(objectives[0] + 0.5 * objectives[1]):.6f}",
                    "final_soc": f"{dispatch['soc'][-1]:.6f}",
                    "solution": " ".join(f"{value:.6f}" for value in solution),
                }
            )


def write_dispatch_csv(path, solution):
    dispatch = evaluate_dispatch(solution)
    fieldnames = [
        "hour",
        "load_kw",
        "pv_kw",
        "wt_kw",
        "diesel_kw",
        "battery_kw",
        "grid_kw",
        "soc",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for hour in range(len(dispatch["load_kw"])):
            writer.writerow(
                {
                    "hour": hour + 1,
                    "load_kw": f"{dispatch['load_kw'][hour]:.6f}",
                    "pv_kw": f"{dispatch['pv_kw'][hour]:.6f}",
                    "wt_kw": f"{dispatch['wt_kw'][hour]:.6f}",
                    "diesel_kw": f"{dispatch['diesel_kw'][hour]:.6f}",
                    "battery_kw": f"{dispatch['battery_kw'][hour]:.6f}",
                    "grid_kw": f"{dispatch['grid_kw'][hour]:.6f}",
                    "soc": f"{dispatch['soc'][hour + 1]:.6f}",
                }
            )


def plot_pareto(path, archive_objectives):
    plt.figure(figsize=(7, 5.5))
    plt.scatter(archive_objectives[:, 0], archive_objectives[:, 1], s=28, alpha=0.8)
    plt.xlabel("Economic cost")
    plt.ylabel("Environment cost")
    plt.title("MOIABC Pareto front for microgrid dispatch")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_history(path, history):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(history)), history, linewidth=1.6)
    plt.xlabel("Iteration")
    plt.ylabel("Best objective sum")
    plt.title("MOIABC convergence curve")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    archive_solutions, archive_objectives, history, used_seed = MOIABC.multi_objective_iabc(
        objective_function=objective_function,
        bounds=BOUNDS,
        bee=BEE,
        max_iter=MAX_ITER,
        limit=LIMIT,
        archive_size=ARCHIVE_SIZE,
        tournament_size=3,
        elite_rate=0.05,
        elimination_rate=0.10,
        seed=SEED,
    )

    objective_sums = np.sum(archive_objectives, axis=1)
    best_sum_index = int(np.argmin(objective_sums))
    compromise_index = int(np.argmin(archive_objectives[:, 0] + 0.5 * archive_objectives[:, 1]))
    compromise_dispatch = evaluate_dispatch(archive_solutions[compromise_index])
    raw_dispatches = [evaluate_dispatch(solution) for solution in archive_solutions]
    raw_economic_costs = np.array([item["economic_cost"] for item in raw_dispatches], dtype=float)
    raw_environment_costs = np.array([item["environment_cost"] for item in raw_dispatches], dtype=float)

    summary = {
        "seed": used_seed,
        "bee": BEE,
        "max_iter": MAX_ITER,
        "limit": LIMIT,
        "archive_size": int(len(archive_objectives)),
        "best_penalized_sum": float(objective_sums[best_sum_index]),
        "min_economic_cost": float(np.min(raw_economic_costs)),
        "min_environment_cost": float(np.min(raw_environment_costs)),
        "compromise_index": compromise_index + 1,
        "compromise_economic_cost": compromise_dispatch["economic_cost"],
        "compromise_environment_cost": compromise_dispatch["environment_cost"],
        "compromise_penalized_economic_objective": float(archive_objectives[compromise_index, 0]),
        "compromise_penalized_environment_objective": float(archive_objectives[compromise_index, 1]),
        "compromise_penalty": compromise_dispatch["penalty"],
        "compromise_final_soc": float(compromise_dispatch["soc"][-1]),
    }

    (OUTPUT_DIR / "multi_objective_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_pareto_csv(OUTPUT_DIR / "multi_objective_pareto.csv", archive_solutions, archive_objectives)
    write_dispatch_csv(OUTPUT_DIR / "multi_objective_compromise_dispatch.csv", archive_solutions[compromise_index])

    if SAVE_PLOTS:
        plot_pareto(OUTPUT_DIR / "multi_objective_pareto.png", archive_objectives)
        plot_history(OUTPUT_DIR / "multi_objective_history.png", history)

    print("=" * 80)
    print("Multi-objective microgrid application case finished")
    print(f"Results: {OUTPUT_DIR}")
    print(f"Seed: {used_seed}")
    print(f"Archive size: {len(archive_objectives)}")
    print(f"Min economic cost: {np.min(raw_economic_costs):.6f}")
    print(f"Min environment cost: {np.min(raw_environment_costs):.6f}")
    print(f"Compromise economic cost: {compromise_dispatch['economic_cost']:.6f}")
    print(f"Compromise environment cost: {compromise_dispatch['environment_cost']:.6f}")
    print(f"Compromise penalty: {compromise_dispatch['penalty']:.6f}")


if __name__ == "__main__":
    main()
