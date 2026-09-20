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

from multi_objective.algorithms import MOABC, MOIABC
from multi_objective.mo_utils import (
    attach_igd_metrics,
    best_sum_history_value,
    calculate_hypervolume,
    spacing_metric,
    update_archive,
    validate_bounds,
)
from multi_objective.multiobjective_benchmarks import CEC2009_UF_BENCHMARKS, CEC2020_MMO_BENCHMARKS, ZDT_BENCHMARKS
from multi_objective.statistical_tests import (
    print_average_rank_overview,
    print_wilcoxon_overview,
    save_average_rank_results,
    save_wilcoxon_results,
)
from multi_objective.experiment_utils import (
    env_bool,
    env_csv,
    env_float,
    env_int,
    env_output_dir,
    format_float,
    print_progress,
    safe_filename_stem,
    save_rows_to_csv,
    select_enabled_items,
    select_named_items,
)


RUN_TIMES = env_int("MOIABC_ABLATION_RUN_TIMES", 30)
SEED_BASE = env_int("MOIABC_ABLATION_SEED_BASE", 20260723)
OUTPUT_DIR = env_output_dir("MOIABC_ABLATION_OUTPUT_DIR", MODULE_DIR / "moiabc_ablation_results", MODULE_DIR)
PARALLEL_WORKERS = env_int("MOIABC_ABLATION_WORKERS", 8)
SAVE_ARCHIVE_POINTS = env_bool("MOIABC_ABLATION_SAVE_ARCHIVE_POINTS", False)
IGD_REFERENCE_POINTS = env_int("MOIABC_ABLATION_IGD_REFERENCE_POINTS", 2000)
IGD_REFERENCE_CANDIDATES = env_int("MOIABC_ABLATION_IGD_REFERENCE_CANDIDATES", 12000)

ENABLED_SUITES = env_csv("MOIABC_ABLATION_SUITES", ["ZDT", "CEC2009_UF", "CEC2020_MMO"])
ENABLED_FUNCTION_IDS = env_csv("MOIABC_ABLATION_FUNCTION_IDS")
ENABLED_VARIANTS = env_csv("MOIABC_ABLATION_VARIANTS")

BENCHMARK_SUITES = {
    "ZDT": ZDT_BENCHMARKS,
    "CEC2009_UF": CEC2009_UF_BENCHMARKS,
    "CEC2020_MMO": CEC2020_MMO_BENCHMARKS,
}

COMMON_PARAMS = {
    "bee": env_int("MOIABC_ABLATION_BEE", 80),
    "max_iter": env_int("MOIABC_ABLATION_MAX_ITER", 800),
    "limit": env_int("MOIABC_ABLATION_LIMIT", 160),
    "archive_size": env_int("MOIABC_ABLATION_ARCHIVE_SIZE", 100),
    "tournament_size": env_int("MOIABC_ABLATION_TOURNAMENT_SIZE", 3),
    "elite_rate": env_float("MOIABC_ABLATION_ELITE_RATE", 0.25),
    "elimination_rate": env_float("MOIABC_ABLATION_ELIMINATION_RATE", 0.25),
    "archive_guidance_rate": env_float("MOIABC_ABLATION_ARCHIVE_GUIDANCE_RATE", 0.40),
}

VARIANTS = [
    {
        "name": "MOIABC",
        "use_good_point_init": True,
        "use_tournament_selection": True,
        "use_elite_enhancement": True,
        "use_archive_guidance": True,
        "use_worst_elimination": True,
        "use_adaptive_elimination": True,
    },
    {
        "name": "MOIABC-no-good-point-init",
        "use_good_point_init": False,
        "use_tournament_selection": True,
        "use_elite_enhancement": True,
        "use_archive_guidance": True,
        "use_worst_elimination": True,
        "use_adaptive_elimination": True,
    },
    {
        "name": "MOIABC-no-tournament-selection",
        "use_good_point_init": True,
        "use_tournament_selection": False,
        "use_elite_enhancement": True,
        "use_archive_guidance": True,
        "use_worst_elimination": True,
        "use_adaptive_elimination": True,
    },
    {
        "name": "MOIABC-no-archive-guidance",
        "use_good_point_init": True,
        "use_tournament_selection": True,
        "use_elite_enhancement": True,
        "use_archive_guidance": False,
        "use_worst_elimination": True,
        "use_adaptive_elimination": True,
    },
    {
        "name": "MOIABC-no-elite-enhancement",
        "use_good_point_init": True,
        "use_tournament_selection": True,
        "use_elite_enhancement": False,
        "use_archive_guidance": False,
        "use_worst_elimination": True,
        "use_adaptive_elimination": True,
    },
    {
        "name": "MOIABC-no-adaptive-elimination",
        "use_good_point_init": True,
        "use_tournament_selection": True,
        "use_elite_enhancement": True,
        "use_archive_guidance": True,
        "use_worst_elimination": True,
        "use_adaptive_elimination": False,
    },
    {
        "name": "MOIABC-no-worst-elimination",
        "use_good_point_init": True,
        "use_tournament_selection": True,
        "use_elite_enhancement": True,
        "use_archive_guidance": True,
        "use_worst_elimination": False,
        "use_adaptive_elimination": False,
    },
    {
        "name": "MOABC-equivalent",
        "use_good_point_init": False,
        "use_tournament_selection": False,
        "use_elite_enhancement": False,
        "use_archive_guidance": False,
        "use_worst_elimination": False,
        "use_adaptive_elimination": False,
    },
]

STATISTICAL_TEST_METRICS = [
    ("hypervolume", True),
    ("spacing", False),
    ("best_sum", False),
    ("igd", False),
    ("igd_plus", False),
]


def get_enabled_benchmarks():
    return select_enabled_items(
        ENABLED_SUITES,
        BENCHMARK_SUITES,
        ENABLED_FUNCTION_IDS,
        suite_label="benchmark suite",
        empty_message="No benchmark selected. Check MOIABC_ABLATION_SUITES or MOIABC_ABLATION_FUNCTION_IDS.",
    )


def get_enabled_variants():
    variants = select_named_items(VARIANTS, ENABLED_VARIANTS, item_label="variant")
    if "MOIABC" not in [variant["name"] for variant in variants]:
        raise ValueError("MOIABC must be enabled as the reference variant.")
    return variants


def run_moiabc_variant(objective_function, bounds, seed, variant, params):
    used_seed = int(seed)
    np.random.seed(used_seed)

    bounds = validate_bounds(bounds)
    if variant["use_good_point_init"]:
        food_sources, objectives, trials = MOIABC.initialize_food_sources(
            params["bee"],
            bounds,
            objective_function,
        )
    else:
        food_sources, objectives, trials = MOABC.initialize_food_sources(
            params["bee"],
            bounds,
            objective_function,
        )

    archive_solutions = np.empty((0, len(bounds)), dtype=float)
    archive_objectives = np.empty((0, objectives.shape[1]), dtype=float)
    archive_solutions, archive_objectives = update_archive(
        archive_solutions,
        archive_objectives,
        food_sources,
        objectives,
        params["archive_size"],
    )
    history = [best_sum_history_value(archive_objectives)]

    for iteration in range(params["max_iter"]):
        MOIABC.employed_bee_phase(
            food_sources,
            objectives,
            trials,
            bounds,
            objective_function,
        )

        if variant["use_tournament_selection"]:
            MOIABC.onlooker_bee_phase(
                food_sources,
                objectives,
                trials,
                bounds,
                objective_function,
                tournament_size=params["tournament_size"],
            )
        else:
            MOABC.onlooker_bee_phase(food_sources, objectives, trials, bounds, objective_function)

        MOIABC.scout_bee_phase(food_sources, objectives, trials, bounds, objective_function, params["limit"])

        if variant["use_elite_enhancement"]:
            guidance_rate = params["archive_guidance_rate"] if variant["use_archive_guidance"] else 0.0
            MOIABC.elite_enhancement_phase(
                food_sources,
                objectives,
                trials,
                bounds,
                objective_function,
                elite_rate=params["elite_rate"],
                archive_solutions=archive_solutions,
                archive_guidance_rate=guidance_rate,
            )

        if variant["use_worst_elimination"]:
            population_best_value = float(np.min(np.sum(objectives, axis=1)))
            current_best_value = min(history[-1], population_best_value)
            if variant.get("use_adaptive_elimination", True):
                current_elimination_rate = MOIABC.get_current_elimination_rate(
                    params["elimination_rate"],
                    iteration,
                    params["max_iter"],
                    initial_best_value=history[0],
                    current_best_value=current_best_value,
                )
            else:
                current_elimination_rate = params["elimination_rate"]
            MOIABC.worst_elimination_phase(
                food_sources,
                objectives,
                trials,
                bounds,
                objective_function,
                elimination_rate=current_elimination_rate,
            )

        archive_solutions, archive_objectives = update_archive(
            archive_solutions,
            archive_objectives,
            food_sources,
            objectives,
            params["archive_size"],
        )
        history.append(best_sum_history_value(archive_objectives))

    return archive_solutions, archive_objectives, history, used_seed


def run_variant(variant, benchmark, seed):
    start_time = time.perf_counter()
    archive_solutions, archive_objectives, history, used_seed = run_moiabc_variant(
        objective_function=benchmark["function"],
        bounds=benchmark["bounds"],
        seed=seed,
        variant=variant,
        params=COMMON_PARAMS,
    )
    elapsed_time = time.perf_counter() - start_time

    objective_sums = np.sum(archive_objectives, axis=1)
    return {
        "benchmark_id": benchmark["id"],
        "benchmark_name": benchmark["name"],
        "algorithm": variant["name"],
        "seed": used_seed,
        "archive_solutions": archive_solutions,
        "archive_objectives": archive_objectives,
        "archive_size": len(archive_objectives),
        "best_sum": float(np.min(objective_sums)),
        "mean_sum": float(np.mean(objective_sums)),
        "min_f1": float(np.min(archive_objectives[:, 0])),
        "min_f2": float(np.min(archive_objectives[:, 1])),
        "min_f3": float(np.min(archive_objectives[:, 2])) if archive_objectives.shape[1] >= 3 else "",
        "spacing": spacing_metric(archive_objectives),
        "hypervolume": calculate_hypervolume(archive_objectives, benchmark["reference_point"]),
        "time": elapsed_time,
        "history": history,
    }


def run_variant_task(task):
    variant, benchmark, seed, run_index = task
    result = run_variant(variant, benchmark, seed)
    result["run_index"] = run_index
    return result


def calculate_statistics(results):
    archive_sizes = np.array([item["archive_size"] for item in results], dtype=float)
    best_sums = np.array([item["best_sum"] for item in results], dtype=float)
    mean_sums = np.array([item["mean_sum"] for item in results], dtype=float)
    spacings = np.array([item["spacing"] for item in results], dtype=float)
    hypervolumes = np.array([item["hypervolume"] for item in results], dtype=float)
    igd_values = np.array([item["igd"] for item in results], dtype=float)
    igd_plus_values = np.array([item["igd_plus"] for item in results], dtype=float)
    times = np.array([item["time"] for item in results], dtype=float)
    ddof = 1 if len(results) > 1 else 0

    best_sum_index = int(np.argmin(best_sums))
    best_hv_index = int(np.argmax(hypervolumes))
    best_igd_index = int(np.argmin(igd_values))
    best_igd_plus_index = int(np.argmin(igd_plus_values))
    return {
        "run_times": len(results),
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
        "mean_igd": float(np.mean(igd_values)),
        "std_igd": float(np.std(igd_values, ddof=ddof)),
        "best_igd": float(np.min(igd_values)),
        "mean_igd_plus": float(np.mean(igd_plus_values)),
        "std_igd_plus": float(np.std(igd_plus_values, ddof=ddof)),
        "best_igd_plus": float(np.min(igd_plus_values)),
        "mean_time": float(np.mean(times)),
        "best_sum_seed": results[best_sum_index]["seed"],
        "best_hypervolume_seed": results[best_hv_index]["seed"],
        "best_igd_seed": results[best_igd_index]["seed"],
        "best_igd_plus_seed": results[best_igd_plus_index]["seed"],
    }


def save_detail_results(filename, all_results):
    rows = []
    for grouped_results in all_results.values():
        for results in grouped_results.values():
            for item in results:
                rows.append(
                    {
                        "run": item["run_index"],
                        "benchmark_id": item["benchmark_id"],
                        "benchmark_name": item["benchmark_name"],
                        "variant": item["algorithm"],
                        "seed": item["seed"],
                        "archive_size": item["archive_size"],
                        "best_sum": format_float(item["best_sum"]),
                        "mean_sum": format_float(item["mean_sum"]),
                        "min_f1": format_float(item["min_f1"]),
                        "min_f2": format_float(item["min_f2"]),
                        "min_f3": format_float(item["min_f3"]),
                        "spacing": format_float(item["spacing"]),
                        "hypervolume": format_float(item["hypervolume"]),
                        "reference_front_size": item["reference_front_size"],
                        "used_reference_front_size": item["used_reference_front_size"],
                        "igd": format_float(item["igd"]),
                        "igd_plus": format_float(item["igd_plus"]),
                        "time": format_float(item["time"], precision=6),
                    }
                )
    save_rows_to_csv(filename, rows)


def save_summary_results(filename, all_results):
    rows = []
    for benchmark_id, grouped_results in all_results.items():
        for variant_name, results in grouped_results.items():
            rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "variant": variant_name,
                    **calculate_statistics(results),
                }
            )
    save_rows_to_csv(filename, rows)
    return rows


def save_archive_points(filename, all_results):
    objective_count = max(
        item["archive_objectives"].shape[1]
        for grouped_results in all_results.values()
        for results in grouped_results.values()
        for item in results
    )
    rows = []
    for grouped_results in all_results.values():
        for results in grouped_results.values():
            for item in results:
                for point_index, objective in enumerate(item["archive_objectives"], start=1):
                    row = {
                        "run": item["run_index"],
                        "benchmark_id": item["benchmark_id"],
                        "variant": item["algorithm"],
                        "seed": item["seed"],
                        "point_index": point_index,
                    }
                    for objective_index in range(objective_count):
                        row[f"f{objective_index + 1}"] = (
                            objective[objective_index] if objective_index < len(objective) else ""
                        )
                    rows.append(row)
    save_rows_to_csv(filename, rows)


def print_statistics(benchmark, grouped_results):
    print("\n" + "=" * 80)
    print(benchmark["name"])
    print(f"Reference point: {benchmark['reference_point'].tolist()}")
    for variant_name, results in grouped_results.items():
        stats = calculate_statistics(results)
        print(f"\n{variant_name}")
        print("-" * 45)
        print(f"run_times: {stats['run_times']}")
        print(f"mean_archive_size: {stats['mean_archive_size']:.2f}")
        print(f"mean_best_sum: {stats['mean_best_sum']:.16e}")
        print(f"std_best_sum: {stats['std_best_sum']:.16e}")
        print(f"mean_spacing: {stats['mean_spacing']:.16e}")
        print(f"mean_hypervolume: {stats['mean_hypervolume']:.16e}")
        print(f"std_hypervolume: {stats['std_hypervolume']:.16e}")
        print(f"mean_igd: {stats['mean_igd']:.16e}")
        print(f"std_igd: {stats['std_igd']:.16e}")
        print(f"mean_igd_plus: {stats['mean_igd_plus']:.16e}")
        print(f"std_igd_plus: {stats['std_igd_plus']:.16e}")
        print(f"mean_time: {stats['mean_time']:.6f}s")


def run_benchmark(benchmark, variants, seeds):
    grouped_results = {variant["name"]: [] for variant in variants}
    tasks = [
        (variant, benchmark, int(seed), run_index)
        for run_index, seed in enumerate(seeds, start=1)
        for variant in variants
    ]
    total_tasks = len(tasks)

    print("\n" + "=" * 80)
    print(f"Starting benchmark: {benchmark['name']}")
    print_progress(0, total_tasks, prefix=benchmark["id"])

    worker_count = min(PARALLEL_WORKERS, total_tasks)
    if worker_count <= 1:
        for task_index, task in enumerate(tasks, start=1):
            result = run_variant_task(task)
            grouped_results[result["algorithm"]].append(result)
            print_progress(task_index, total_tasks, prefix=benchmark["id"])
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(run_variant_task, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                grouped_results[result["algorithm"]].append(result)
                completed += 1
                print_progress(completed, total_tasks, prefix=benchmark["id"])

    print()
    for results in grouped_results.values():
        results.sort(key=lambda item: item["run_index"])

    attach_igd_metrics(
        grouped_results,
        reference_points=IGD_REFERENCE_POINTS,
        reference_candidates=IGD_REFERENCE_CANDIDATES,
    )
    print_statistics(benchmark, grouped_results)
    return grouped_results


def print_run_configuration(benchmarks, variants):
    total_runs = len(benchmarks) * len(variants) * RUN_TIMES
    print("\n" + "=" * 80)
    print("MOIABC ablation experiment configuration")
    print(f"Suites: {', '.join(ENABLED_SUITES)}")
    print(f"Benchmark count: {len(benchmarks)}")
    print(f"Benchmarks: {', '.join(benchmark['id'] for benchmark in benchmarks)}")
    print(f"Variant count: {len(variants)}")
    print(f"Variants: {', '.join(variant['name'] for variant in variants)}")
    print(f"Run times: {RUN_TIMES}")
    print(f"Seed base: {SEED_BASE}")
    print(f"Parallel workers: {PARALLEL_WORKERS}")
    print(f"Save archive points: {'yes' if SAVE_ARCHIVE_POINTS else 'no'}")
    print(f"IGD reference points: {IGD_REFERENCE_POINTS if IGD_REFERENCE_POINTS > 0 else 'all'}")
    print(f"IGD reference candidates: {IGD_REFERENCE_CANDIDATES if IGD_REFERENCE_CANDIDATES > 0 else 'all'}")
    print(f"Total runs: {total_runs}")
    print(
        "Params: "
        f"bee={COMMON_PARAMS['bee']}, "
        f"max_iter={COMMON_PARAMS['max_iter']}, "
        f"limit={COMMON_PARAMS['limit']}, "
        f"archive_size={COMMON_PARAMS['archive_size']}, "
        f"tournament_size={COMMON_PARAMS['tournament_size']}, "
        f"elite_rate={COMMON_PARAMS['elite_rate']}, "
        f"elimination_rate={COMMON_PARAMS['elimination_rate']}, "
        f"archive_guidance_rate={COMMON_PARAMS['archive_guidance_rate']}"
    )


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    benchmarks = get_enabled_benchmarks()
    variants = get_enabled_variants()
    print_run_configuration(benchmarks, variants)

    seeds = [int(seed) for seed in np.random.SeedSequence(SEED_BASE).generate_state(RUN_TIMES)]
    all_results = {}
    total_start_time = time.perf_counter()
    for benchmark in benchmarks:
        all_results[benchmark["id"]] = run_benchmark(benchmark, variants, seeds)

    detail_file = OUTPUT_DIR / "moiabc_ablation_detail_results.csv"
    summary_file = OUTPUT_DIR / "moiabc_ablation_summary_by_function.csv"
    archive_points_file = OUTPUT_DIR / "moiabc_ablation_archive_points.csv"
    wilcoxon_file = OUTPUT_DIR / "wilcoxon_ablation_vs_moiabc_results.csv"
    average_rank_file = OUTPUT_DIR / "moiabc_ablation_average_rank.csv"

    save_detail_results(detail_file, all_results)
    save_summary_results(summary_file, all_results)
    if SAVE_ARCHIVE_POINTS:
        save_archive_points(archive_points_file, all_results)

    variant_names = [variant["name"] for variant in variants]
    wilcoxon_rows = []
    for base_variant in [name for name in variant_names if name != "MOIABC"]:
        wilcoxon_rows.extend(
            save_wilcoxon_results(
                OUTPUT_DIR / f"wilcoxon_{safe_filename_stem(base_variant)}_vs_moiabc_results.csv",
                all_results,
                base_algorithm=base_variant,
                improved_algorithm="MOIABC",
                metrics=STATISTICAL_TEST_METRICS,
            )
        )
    save_rows_to_csv(wilcoxon_file, wilcoxon_rows)
    rank_rows = save_average_rank_results(
        average_rank_file,
        all_results,
        algorithms=variant_names,
        metrics=STATISTICAL_TEST_METRICS,
    )

    print_wilcoxon_overview(wilcoxon_rows)
    print_average_rank_overview(rank_rows)

    total_time = time.perf_counter() - total_start_time
    print("\n" + "=" * 80)
    print("MOIABC ablation experiment finished")
    print(f"Elapsed time: {total_time:.2f} seconds")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Detail results: {detail_file.resolve()}")
    print(f"Summary results: {summary_file.resolve()}")
    if SAVE_ARCHIVE_POINTS:
        print(f"Archive points: {archive_points_file.resolve()}")
    print(f"Wilcoxon results: {wilcoxon_file.resolve()}")
    print(f"Average rank results: {average_rank_file.resolve()}")


if __name__ == "__main__":
    main()
