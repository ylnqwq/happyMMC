# -*- coding: utf-8 -*-

import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODULE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from single_objective.algorithms import IABC
from single_objective.single_objective_benchmarks import CEC2022_BENCHMARKS


RUN_TIMES = 30
SEED_BASE = 20240621
BOUNDS = [(-100, 100)] * 10
OUTPUT_DIR = MODULE_DIR / "sensitivity_results"
PARALLEL_WORKERS = 4

ENABLED_SUITES = ["CEC2022"]
ENABLED_FUNCTION_IDS = []

BENCHMARK_SUITES = {
    "CEC2022": CEC2022_BENCHMARKS,
}

COMMON_PARAMS = {
    "bee": 75,
    "max_iter": 200,
    "limit": 150,
    "tournament_size": 3,
}

ELITE_RATES = [0.05, 0.10, 0.15, 0.20, 0.25]
ELIMINATION_RATES = [0.05, 0.10, 0.15, 0.20, 0.25]


def get_enabled_benchmarks():
    benchmarks = []
    for suite_name in ENABLED_SUITES:
        if suite_name not in BENCHMARK_SUITES:
            valid_names = ", ".join(BENCHMARK_SUITES)
            raise ValueError(f"Unknown suite: {suite_name}. Valid suites: {valid_names}")
        benchmarks.extend(BENCHMARK_SUITES[suite_name])

    if ENABLED_FUNCTION_IDS:
        enabled_ids = set(ENABLED_FUNCTION_IDS)
        benchmarks = [item for item in benchmarks if item["id"] in enabled_ids]

    if not benchmarks:
        raise ValueError("No benchmark selected. Check ENABLED_SUITES or ENABLED_FUNCTION_IDS.")
    return benchmarks


def parameter_grid():
    for elite_rate in ELITE_RATES:
        for elimination_rate in ELIMINATION_RATES:
            yield elite_rate, elimination_rate


def format_float(value, precision=16):
    return f"{float(value):.{precision}f}"


def run_once(benchmark, seed, elite_rate, elimination_rate):
    bounds = benchmark.get("bounds", BOUNDS)
    start_time = time.perf_counter()
    _, best_value, _, used_seed = IABC.iabc(
        objective_function=benchmark["function"],
        bounds=bounds,
        seed=seed,
        elite_rate=elite_rate,
        elimination_rate=elimination_rate,
        **COMMON_PARAMS,
    )
    elapsed_time = time.perf_counter() - start_time
    return {
        "benchmark_id": benchmark["id"],
        "elite_rate": elite_rate,
        "elimination_rate": elimination_rate,
        "seed": used_seed,
        "best_value": best_value,
        "error": best_value - benchmark["optimal_value"],
        "time": elapsed_time,
    }


def run_once_task(task):
    benchmark, seed, elite_rate, elimination_rate = task
    return run_once(benchmark, seed, elite_rate, elimination_rate)


def print_configuration(benchmarks):
    total_combinations = len(ELITE_RATES) * len(ELIMINATION_RATES)
    total_runs = len(benchmarks) * total_combinations * RUN_TIMES

    print("\n" + "=" * 80)
    print("IABC parameter sensitivity configuration")
    print(f"Suites: {', '.join(ENABLED_SUITES)}")
    print(f"Benchmark count: {len(benchmarks)}")
    print(f"Benchmarks: {', '.join(item['id'] for item in benchmarks)}")
    print(f"Run times: {RUN_TIMES}")
    print(f"Seed base: {SEED_BASE}")
    print(
        "Common params: "
        f"bee={COMMON_PARAMS['bee']}, "
        f"max_iter={COMMON_PARAMS['max_iter']}, "
        f"limit={COMMON_PARAMS['limit']}, "
        f"tournament_size={COMMON_PARAMS['tournament_size']}"
    )
    print(f"elite_rate values: {ELITE_RATES}")
    print(f"elimination_rate values: {ELIMINATION_RATES}")
    print(f"Parameter combinations: {total_combinations}")
    print(f"Total IABC runs: {total_runs}")
    print(f"Parallel workers: {PARALLEL_WORKERS}")


def save_rows(filename, rows):
    if not rows:
        return
    with open(filename, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(rows):
    grouped = {}
    for row in rows:
        key = (row["elite_rate"], row["elimination_rate"], row["benchmark_id"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (elite_rate, elimination_rate, benchmark_id), items in grouped.items():
        best_values = np.array([item["best_value"] for item in items], dtype=float)
        errors = np.array([item["error"] for item in items], dtype=float)
        times = np.array([item["time"] for item in items], dtype=float)
        ddof = 1 if len(items) > 1 else 0
        summary_rows.append(
            {
                "elite_rate": elite_rate,
                "elimination_rate": elimination_rate,
                "benchmark_id": benchmark_id,
                "run_times": len(items),
                "best": float(np.min(best_values)),
                "mean": float(np.mean(best_values)),
                "std": float(np.std(best_values, ddof=ddof)),
                "mean_error": float(np.mean(errors)),
                "median_error": float(np.median(errors)),
                "std_error": float(np.std(errors, ddof=ddof)),
                "mean_time": float(np.mean(times)),
            }
        )
    return summary_rows


def rank_parameter_sets(summary_rows):
    by_benchmark = {}
    for row in summary_rows:
        by_benchmark.setdefault(row["benchmark_id"], []).append(row)

    rank_sums = {}
    best_counts = {}
    mean_error_sums = {}
    for benchmark_rows in by_benchmark.values():
        ordered = sorted(benchmark_rows, key=lambda item: item["mean_error"])
        for rank, row in enumerate(ordered, start=1):
            key = (row["elite_rate"], row["elimination_rate"])
            rank_sums[key] = rank_sums.get(key, 0.0) + rank
            mean_error_sums[key] = mean_error_sums.get(key, 0.0) + row["mean_error"]
            best_counts.setdefault(key, 0)
        if ordered:
            best_key = (ordered[0]["elite_rate"], ordered[0]["elimination_rate"])
            best_counts[best_key] = best_counts.get(best_key, 0) + 1

    benchmark_count = len(by_benchmark)
    rank_rows = []
    for elite_rate, elimination_rate in parameter_grid():
        key = (elite_rate, elimination_rate)
        rank_rows.append(
            {
                "elite_rate": elite_rate,
                "elimination_rate": elimination_rate,
                "average_rank": rank_sums.get(key, 0.0) / benchmark_count,
                "best_count": best_counts.get(key, 0),
                "benchmark_count": benchmark_count,
                "average_mean_error": mean_error_sums.get(key, 0.0) / benchmark_count,
            }
        )
    return sorted(rank_rows, key=lambda item: (item["average_rank"], -item["best_count"]))


def print_progress(done, total, prefix="", width=32):
    ratio = done / total
    completed = int(width * ratio)
    bar = "#" * completed + "-" * (width - completed)
    print(f"\r{prefix} [{bar}] {done}/{total} {ratio * 100:6.2f}%", end="", flush=True)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    benchmarks = get_enabled_benchmarks()
    print_configuration(benchmarks)

    seeds = [int(seed) for seed in np.random.SeedSequence(SEED_BASE).generate_state(RUN_TIMES)]
    total_tasks = len(benchmarks) * len(ELITE_RATES) * len(ELIMINATION_RATES) * RUN_TIMES
    rows = []
    done = 0
    start_time = time.perf_counter()

    tasks = [
        (benchmark, seed, elite_rate, elimination_rate)
        for benchmark in benchmarks
        for elite_rate, elimination_rate in parameter_grid()
        for seed in seeds
    ]

    worker_count = min(PARALLEL_WORKERS, len(tasks))
    if worker_count <= 1:
        for task in tasks:
            benchmark, _, elite_rate, elimination_rate = task
            rows.append(run_once_task(task))
            done += 1
            prefix = f"{benchmark['id']} e={elite_rate:.2f} d={elimination_rate:.2f}"
            print_progress(done, total_tasks, prefix=prefix)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(run_once_task, task) for task in tasks]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                done += 1
                prefix = f"{row['benchmark_id']} e={row['elite_rate']:.2f} d={row['elimination_rate']:.2f}"
                print_progress(done, total_tasks, prefix=prefix)

    print()
    detail_rows = [
        {
            **row,
            "best_value": format_float(row["best_value"]),
            "error": format_float(row["error"]),
            "time": format_float(row["time"], precision=6),
        }
        for row in rows
    ]
    summary_rows = summarize_results(rows)
    rank_rows = rank_parameter_sets(summary_rows)

    save_rows(OUTPUT_DIR / "sensitivity_detail_results.csv", detail_rows)
    save_rows(OUTPUT_DIR / "sensitivity_summary_by_function.csv", summary_rows)
    save_rows(OUTPUT_DIR / "sensitivity_average_rank.csv", rank_rows)

    print("\n" + "=" * 80)
    print("IABC parameter sensitivity finished")
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} seconds")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    if rank_rows:
        best = rank_rows[0]
        print(
            "Recommended params: "
            f"elite_rate={best['elite_rate']}, "
            f"elimination_rate={best['elimination_rate']}, "
            f"average_rank={best['average_rank']:.3f}, "
            f"best_count={best['best_count']}/{best['benchmark_count']}"
        )


if __name__ == "__main__":
    main()
