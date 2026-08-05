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

from experiment_utils import env_bool, env_float, env_int, env_output_dir, format_float, print_progress
from multi_objective.algorithms import MOIABC
from multi_objective.application_point.microgrid_dispatch_model import (
    BOUNDS,
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

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False


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
        "grid_buy_kw",
        "grid_sell_kw",
        "soc",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for hour in range(len(dispatch["load_kw"])):
            grid_kw = dispatch["grid_kw"][hour]
            writer.writerow(
                {
                    "hour": hour + 1,
                    "load_kw": f"{dispatch['load_kw'][hour]:.6f}",
                    "pv_kw": f"{dispatch['pv_kw'][hour]:.6f}",
                    "wt_kw": f"{dispatch['wt_kw'][hour]:.6f}",
                    "diesel_kw": f"{dispatch['diesel_kw'][hour]:.6f}",
                    "battery_kw": f"{dispatch['battery_kw'][hour]:.6f}",
                    "grid_kw": f"{grid_kw:.6f}",
                    "grid_buy_kw": f"{max(grid_kw, 0.0):.6f}",
                    "grid_sell_kw": f"{max(-grid_kw, 0.0):.6f}",
                    "soc": f"{dispatch['soc'][hour + 1]:.6f}",
                }
            )


def write_power_curves_csv(path, solution):
    dispatch = evaluate_dispatch(solution)
    fieldnames = [
        "hour",
        "load_kw",
        "wind_kw",
        "pv_kw",
        "diesel_kw",
        "battery_kw",
        "battery_discharge_kw",
        "battery_charge_kw",
        "grid_net_kw",
        "grid_buy_kw",
        "grid_sell_kw",
        "total_generation_kw",
        "total_supply_kw",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for hour in range(len(dispatch["load_kw"])):
            battery_kw = dispatch["battery_kw"][hour]
            grid_kw = dispatch["grid_kw"][hour]
            battery_discharge_kw = max(battery_kw, 0.0)
            battery_charge_kw = max(-battery_kw, 0.0)
            grid_buy_kw = max(grid_kw, 0.0)
            grid_sell_kw = max(-grid_kw, 0.0)
            total_generation_kw = dispatch["wt_kw"][hour] + dispatch["pv_kw"][hour] + dispatch["diesel_kw"][hour]
            total_supply_kw = total_generation_kw + battery_discharge_kw + grid_buy_kw
            writer.writerow(
                {
                    "hour": hour + 1,
                    "load_kw": f"{dispatch['load_kw'][hour]:.6f}",
                    "wind_kw": f"{dispatch['wt_kw'][hour]:.6f}",
                    "pv_kw": f"{dispatch['pv_kw'][hour]:.6f}",
                    "diesel_kw": f"{dispatch['diesel_kw'][hour]:.6f}",
                    "battery_kw": f"{battery_kw:.6f}",
                    "battery_discharge_kw": f"{battery_discharge_kw:.6f}",
                    "battery_charge_kw": f"{battery_charge_kw:.6f}",
                    "grid_net_kw": f"{grid_kw:.6f}",
                    "grid_buy_kw": f"{grid_buy_kw:.6f}",
                    "grid_sell_kw": f"{grid_sell_kw:.6f}",
                    "total_generation_kw": f"{total_generation_kw:.6f}",
                    "total_supply_kw": f"{total_supply_kw:.6f}",
                }
            )


def write_soc_curve_csv(path, solution):
    dispatch = evaluate_dispatch(solution)
    fieldnames = [
        "hour",
        "battery_kw",
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
                    "battery_kw": f"{dispatch['battery_kw'][hour]:.6f}",
                    "energy_start_kwh": f"{dispatch['energy_kwh'][hour]:.6f}",
                    "energy_end_kwh": f"{dispatch['energy_kwh'][hour + 1]:.6f}",
                    "soc_start": f"{dispatch['soc'][hour]:.6f}",
                    "soc_end": f"{dispatch['soc'][hour + 1]:.6f}",
                }
            )


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


def plot_pareto(path, selected_solutions, selected_objectives, reference_solutions=None):
    selected_raw = raw_cost_objectives(selected_solutions)

    plt.figure(figsize=(7, 5.5))
    if reference_solutions is not None and len(reference_solutions) > len(selected_solutions):
        reference_raw = raw_cost_objectives(reference_solutions)
        plt.scatter(
            reference_raw[:, 0],
            reference_raw[:, 1],
            s=18,
            alpha=0.35,
            label="Empirical reference front",
        )
    plt.scatter(selected_raw[:, 0], selected_raw[:, 1], s=30, alpha=0.85, label="Selected MOIABC archive")

    compromise_index, _ = select_compromise_index(selected_objectives)
    plt.scatter(
        selected_raw[compromise_index, 0],
        selected_raw[compromise_index, 1],
        s=90,
        marker="*",
        label="Compromise solution",
    )
    plt.xlabel("Economic cost")
    plt.ylabel("Environment cost")
    plt.title("MOIABC Pareto front for microgrid dispatch")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
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


def run_once(run_index):
    seed = SEED_BASE + run_index
    start_time = time.perf_counter()
    archive_solutions, archive_objectives, history, used_seed = MOIABC.multi_objective_iabc(
        objective_function=objective_function,
        bounds=BOUNDS,
        bee=BEE,
        max_iter=MAX_ITER,
        limit=LIMIT,
        archive_size=ARCHIVE_SIZE,
        tournament_size=TOURNAMENT_SIZE,
        elite_rate=ELITE_RATE,
        elimination_rate=ELIMINATION_RATE,
        archive_guidance_rate=ARCHIVE_GUIDANCE_RATE,
        seed=seed,
    )
    elapsed_seconds = time.perf_counter() - start_time
    return {
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


def run_all():
    worker_count = min(PARALLEL_WORKERS, RUN_TIMES)
    if worker_count <= 1:
        run_results = []
        for run_index in range(RUN_TIMES):
            run_results.append(run_once(run_index))
            print_progress(run_index + 1, RUN_TIMES, prefix="Progress")
        print()
        return run_results

    run_results = []
    completed = 0
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(run_once, run_index) for run_index in range(RUN_TIMES)]
        for future in as_completed(futures):
            run_results.append(future.result())
            completed += 1
            print_progress(completed, RUN_TIMES, prefix="Progress")
    print()
    return sorted(run_results, key=lambda item: item["run"])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if RUN_TIMES <= 0:
        raise ValueError("APP_RUN_TIMES must be greater than 0.")
    if PARALLEL_WORKERS <= 0:
        raise ValueError("APP_WORKERS must be greater than 0.")

    print("=" * 80)
    print("MOIABC microgrid dispatch configuration")
    print(f"Runs: {RUN_TIMES}, seed_base: {SEED_BASE}, bee: {BEE}, max_iter: {MAX_ITER}, archive_size: {ARCHIVE_SIZE}")
    print(f"Parallel workers: {min(PARALLEL_WORKERS, RUN_TIMES)}")

    run_results = run_all()

    reference_solutions, reference_objectives, all_objectives = build_empirical_reference_front(run_results)
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

    summary = {
        "run_times": RUN_TIMES,
        "seed_base": SEED_BASE,
        "selected_run": selected_result["run"],
        "selected_seed": selected_result["seed"],
        "bee": BEE,
        "max_iter": MAX_ITER,
        "limit": LIMIT,
        "configured_archive_size": ARCHIVE_SIZE,
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
    }

    (OUTPUT_DIR / "multi_objective_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_metrics_csv(OUTPUT_DIR / "multi_objective_metrics.csv", metric_rows)
    write_metrics_summary_csv(OUTPUT_DIR / "multi_objective_metrics_summary.csv", metrics_summary)
    write_convergence_history_csv(OUTPUT_DIR / "multi_objective_convergence_history.csv", run_results)
    write_average_convergence_history_csv(OUTPUT_DIR / "multi_objective_average_convergence_history.csv", run_results)
    write_pareto_csv(OUTPUT_DIR / "multi_objective_pareto.csv", selected_solutions, selected_objectives)
    write_pareto_csv(OUTPUT_DIR / "multi_objective_reference_pareto.csv", reference_solutions, reference_objectives)
    write_dispatch_csv(OUTPUT_DIR / "multi_objective_compromise_dispatch.csv", selected_solutions[compromise_index])
    write_power_curves_csv(OUTPUT_DIR / "compromise_power_curves.csv", selected_solutions[compromise_index])
    write_soc_curve_csv(OUTPUT_DIR / "compromise_soc_curve.csv", selected_solutions[compromise_index])

    if SAVE_PLOTS:
        plot_pareto(OUTPUT_DIR / "multi_objective_pareto.png", selected_solutions, selected_objectives, reference_solutions)
        plot_history(OUTPUT_DIR / "multi_objective_history.png", selected_result["history"])

    print("=" * 80)
    print("Multi-objective microgrid application case finished")
    print(f"Results: {OUTPUT_DIR}")
    print(f"Selected run: {selected_result['run']}, seed: {selected_result['seed']}")
    print(f"Selected archive size: {len(selected_objectives)}")
    print(f"Reference front size: {len(reference_objectives)}")
    print(f"IGD: {selected_metrics['IGD']:.6e}")
    print(f"IGD+: {selected_metrics['IGD+']:.6e}")
    print(f"HV: {selected_metrics['HV']:.6e}")
    print(f"best_sum: {selected_metrics['best_sum']:.6e}")
    print(f"spacing: {selected_metrics['spacing']:.6e}")
    print(f"Compromise economic cost: {compromise_dispatch['economic_cost']:.6f}")
    print(f"Compromise environment cost: {compromise_dispatch['environment_cost']:.6f}")
    print(f"Compromise penalty: {compromise_dispatch['penalty']:.6f}")


if __name__ == "__main__":
    main()
