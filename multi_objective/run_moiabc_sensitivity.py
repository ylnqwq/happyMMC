# -*- coding: utf-8 -*-

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

from multi_objective.algorithms import MOIABC
from multi_objective.mo_utils import calculate_hypervolume, spacing_metric
from multi_objective.multiobjective_benchmarks import CEC2009_UF_BENCHMARKS, CEC2020_MMO_BENCHMARKS, ZDT_BENCHMARKS
from experiment_utils import env_csv, env_int, env_output_dir, format_float, print_progress, save_rows_to_csv, select_enabled_items


RUN_TIMES = env_int("MOIABC_SENSITIVITY_RUN_TIMES", 30)
SEED_BASE = env_int("MOIABC_SENSITIVITY_SEED_BASE", 20260719)
OUTPUT_DIR = env_output_dir("MOIABC_SENSITIVITY_OUTPUT_DIR", MODULE_DIR / "moiabc_sensitivity_results", MODULE_DIR)
PARALLEL_WORKERS = env_int("MOIABC_SENSITIVITY_WORKERS", 4)

ENABLED_SUITES = env_csv("MOIABC_SENSITIVITY_SUITES", ["ZDT", "CEC2009_UF", "CEC2020_MMO"])
ENABLED_FUNCTION_IDS = env_csv("MOIABC_SENSITIVITY_FUNCTION_IDS")

BENCHMARK_SUITES = {
    "ZDT": ZDT_BENCHMARKS,
    "CEC2009_UF": CEC2009_UF_BENCHMARKS,
    "CEC2020_MMO": CEC2020_MMO_BENCHMARKS,
}

COMMON_PARAMS = {
    "bee": 80,
    "max_iter": 800,
    "limit": 160,
    "archive_size": 100,
    "tournament_size": 3,
}

ELITE_RATES = [0.05, 0.10, 0.15, 0.20, 0.25]
ELIMINATION_RATES = [0.05, 0.10, 0.15, 0.20, 0.25]
RANK_METRICS = [
    ("hypervolume", True),
    ("spacing", False),
    ("best_sum", False),
]


def get_enabled_benchmarks():
    return select_enabled_items(
        ENABLED_SUITES,
        BENCHMARK_SUITES,
        ENABLED_FUNCTION_IDS,
        suite_label="benchmark suite",
        empty_message="No benchmark selected. Check ENABLED_SUITES or ENABLED_FUNCTION_IDS.",
    )


def parameter_grid():
    for elite_rate in ELITE_RATES:
        for elimination_rate in ELIMINATION_RATES:
            yield elite_rate, elimination_rate


def run_once(benchmark, seed, elite_rate, elimination_rate):
    start_time = time.perf_counter()
    archive_solutions, archive_objectives, _, used_seed = MOIABC.multi_objective_iabc(
        objective_function=benchmark["function"],
        bounds=benchmark["bounds"],
        seed=seed,
        elite_rate=elite_rate,
        elimination_rate=elimination_rate,
        **COMMON_PARAMS,
    )
    elapsed_time = time.perf_counter() - start_time

    objective_sums = np.sum(archive_objectives, axis=1)
    best_sum = float(np.min(objective_sums))

    return {
        "benchmark_id": benchmark["id"],
        "elite_rate": elite_rate,
        "elimination_rate": elimination_rate,
        "seed": used_seed,
        "archive_size": len(archive_objectives),
        "best_sum": best_sum,
        "mean_sum": float(np.mean(objective_sums)),
        "spacing": spacing_metric(archive_objectives),
        "hypervolume": calculate_hypervolume(archive_objectives, benchmark["reference_point"]),
        "time": elapsed_time,
    }


def run_once_task(task):
    benchmark, seed, elite_rate, elimination_rate = task
    return run_once(benchmark, seed, elite_rate, elimination_rate)


def print_configuration(benchmarks):
    total_combinations = len(ELITE_RATES) * len(ELIMINATION_RATES)
    total_runs = len(benchmarks) * total_combinations * RUN_TIMES

    print("\n" + "=" * 80)
    print("MOIABC parameter sensitivity configuration")
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
        f"archive_size={COMMON_PARAMS['archive_size']}, "
        f"tournament_size={COMMON_PARAMS['tournament_size']}"
    )
    print(f"elite_rate values: {ELITE_RATES}")
    print(f"elimination_rate values: {ELIMINATION_RATES}")
    print(f"Parameter combinations: {total_combinations}")
    print(f"Total MOIABC runs: {total_runs}")
    print(f"Parallel workers: {PARALLEL_WORKERS}")


def summarize_results(rows):
    grouped = {}
    for row in rows:
        key = (row["elite_rate"], row["elimination_rate"], row["benchmark_id"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (elite_rate, elimination_rate, benchmark_id), items in grouped.items():
        archive_sizes = np.array([item["archive_size"] for item in items], dtype=float)
        best_sums = np.array([item["best_sum"] for item in items], dtype=float)
        mean_sums = np.array([item["mean_sum"] for item in items], dtype=float)
        spacings = np.array([item["spacing"] for item in items], dtype=float)
        hypervolumes = np.array([item["hypervolume"] for item in items], dtype=float)
        times = np.array([item["time"] for item in items], dtype=float)
        ddof = 1 if len(items) > 1 else 0
        summary_rows.append(
            {
                "elite_rate": elite_rate,
                "elimination_rate": elimination_rate,
                "benchmark_id": benchmark_id,
                "run_times": len(items),
                "mean_archive_size": float(np.mean(archive_sizes)),
                "mean_best_sum": float(np.mean(best_sums)),
                "std_best_sum": float(np.std(best_sums, ddof=ddof)),
                "best_sum": float(np.min(best_sums)),
                "mean_sum": float(np.mean(mean_sums)),
                "mean_spacing": float(np.mean(spacings)),
                "std_spacing": float(np.std(spacings, ddof=ddof)),
                "mean_hypervolume": float(np.mean(hypervolumes)),
                "std_hypervolume": float(np.std(hypervolumes, ddof=ddof)),
                "best_hypervolume": float(np.max(hypervolumes)),
                "mean_time": float(np.mean(times)),
            }
        )
    return summary_rows


def rank_parameter_sets(summary_rows):
    by_benchmark_metric = {}
    for row in summary_rows:
        for metric_name, higher_is_better in RANK_METRICS:
            key = (row["benchmark_id"], metric_name)
            by_benchmark_metric.setdefault(key, []).append((row, higher_is_better))

    rank_sums = {}
    best_counts = {}
    metric_sums = {}
    for (benchmark_id, metric_name), metric_rows in by_benchmark_metric.items():
        higher_is_better = metric_rows[0][1]
        value_key = f"mean_{metric_name}" if metric_name != "best_sum" else "mean_best_sum"
        ordered = sorted(
            [item[0] for item in metric_rows],
            key=lambda item: item[value_key],
            reverse=higher_is_better,
        )
        for rank, row in enumerate(ordered, start=1):
            key = (row["elite_rate"], row["elimination_rate"])
            rank_sums[key] = rank_sums.get(key, 0.0) + rank
            best_counts.setdefault(key, 0)
            metric_sums.setdefault(
                key,
                {
                    "mean_hypervolume": 0.0,
                    "mean_spacing": 0.0,
                    "mean_best_sum": 0.0,
                },
            )
            metric_sums[key][value_key] += row[value_key]
        if ordered:
            best_key = (ordered[0]["elite_rate"], ordered[0]["elimination_rate"])
            best_counts[best_key] = best_counts.get(best_key, 0) + 1

    benchmark_count = len({row["benchmark_id"] for row in summary_rows})
    metric_count = len(RANK_METRICS)
    divisor = benchmark_count * metric_count
    rank_rows = []
    for elite_rate, elimination_rate in parameter_grid():
        key = (elite_rate, elimination_rate)
        sums = metric_sums.get(
            key,
            {
                "mean_hypervolume": 0.0,
                "mean_spacing": 0.0,
                "mean_best_sum": 0.0,
            },
        )
        average_rank = rank_sums.get(key)
        rank_rows.append(
            {
                "elite_rate": elite_rate,
                "elimination_rate": elimination_rate,
                "average_rank": float("inf") if average_rank is None else average_rank / divisor,
                "best_count": best_counts.get(key, 0),
                "benchmark_count": benchmark_count,
                "metric_count": metric_count,
                "average_hypervolume": sums["mean_hypervolume"] / benchmark_count,
                "average_spacing": sums["mean_spacing"] / benchmark_count,
                "average_best_sum": sums["mean_best_sum"] / benchmark_count,
            }
        )
    return sorted(rank_rows, key=lambda item: (item["average_rank"], -item["best_count"]))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
            "best_sum": format_float(row["best_sum"]),
            "mean_sum": format_float(row["mean_sum"]),
            "spacing": format_float(row["spacing"]),
            "hypervolume": format_float(row["hypervolume"]),
            "time": format_float(row["time"], precision=6),
        }
        for row in rows
    ]
    summary_rows = summarize_results(rows)
    rank_rows = rank_parameter_sets(summary_rows)

    save_rows_to_csv(OUTPUT_DIR / "moiabc_sensitivity_detail_results.csv", detail_rows)
    save_rows_to_csv(OUTPUT_DIR / "moiabc_sensitivity_summary_by_function.csv", summary_rows)
    save_rows_to_csv(OUTPUT_DIR / "moiabc_sensitivity_average_rank.csv", rank_rows)

    print("\n" + "=" * 80)
    print("MOIABC parameter sensitivity finished")
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} seconds")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    if rank_rows:
        best = rank_rows[0]
        print(
            "Recommended params: "
            f"elite_rate={best['elite_rate']}, "
            f"elimination_rate={best['elimination_rate']}, "
            f"average_rank={best['average_rank']:.3f}, "
            f"best_count={best['best_count']}/{best['benchmark_count'] * best['metric_count']}"
        )


if __name__ == "__main__":
    main()
