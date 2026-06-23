# -*- coding: utf-8 -*-

import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODULE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from single_objective.algorithms import ABC, ACO, GA, IABC, IABC_MSS, NDBP_ABC
from single_objective.single_objective_benchmarks import CEC2022_BENCHMARKS
from single_objective.statistical_tests import (
    print_average_rank_overview,
    print_wilcoxon_overview,
    save_average_rank_results,
    save_wilcoxon_results,
)


RUN_TIMES = 1
BOUNDS = [(-100, 100)] * 10
OUTPUT_DIR = Path(__file__).resolve().parent / "comparison_results"

# 全局测试开关：
# 1. ENABLED_SUITES 控制要跑哪些测试集，可选 "CEC2022"。
#    例：只跑 CEC2022 -> ENABLED_SUITES = ["CEC2022"]
# 2. ENABLED_FUNCTION_IDS 控制要跑哪些具体函数，空列表表示不过滤。
#    例：只跑 CEC2022_F1 和 CEC2022_F6 -> ENABLED_FUNCTION_IDS = ["CEC2022_F1", "CEC2022_F6"]
ENABLED_SUITES = ["CEC2022"]
ENABLED_FUNCTION_IDS = []
# 可选算法 ABC, GA, ACO, IABC-MSS, NDBP-ABC, IABC
ENABLED_ALGORITHMS = []

BENCHMARK_SUITES = {
    "CEC2022": CEC2022_BENCHMARKS,
}

COMMON_PARAMS = {
    "bee": 75,
    "max_iter": 750,
    "limit": 150,
}

ALGORITHMS = [
    {
        "name": "ABC",
        "runner": ABC.artificial_bee_colony,
        "params": COMMON_PARAMS,
    },
    {
        "name": "GA",
        "runner": GA.genetic_algorithm,
        "params": {
            "population_size": COMMON_PARAMS["bee"],
            "max_iter": COMMON_PARAMS["max_iter"],
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
        },
    },
    {
        "name": "ACO",
        "runner": ACO.ant_colony_optimization,
        "params": {
            "ant": COMMON_PARAMS["bee"],
            "max_iter": COMMON_PARAMS["max_iter"],
            "levels": 50,
            "evaporation_rate": 0.2,
            "deposit_weight": 1.0,
        },
    },
    {
        "name": "IABC-MSS",
        "runner": IABC_MSS.iabc_mss,
        "params": {
            "bee": COMMON_PARAMS["bee"],
            "max_iter": COMMON_PARAMS["max_iter"],
            "limit": COMMON_PARAMS["limit"],
        },
    },
    {
        "name": "NDBP-ABC",
        "runner": NDBP_ABC.ndbp_abc,
        "params": {
            "bee": COMMON_PARAMS["bee"],
            "max_iter": COMMON_PARAMS["max_iter"],
            "limit": COMMON_PARAMS["limit"],
        },
    },
    {
        "name": "IABC",
        "runner": IABC.iabc,
        "params": {
            **COMMON_PARAMS,
            "tournament_size": 3,
            "elite_rate": 0.25,
            "elimination_rate": 0.15,
        },
    },
]
STATISTICAL_TEST_METRICS = [
    ("best_value", False),
    ("error", False),
]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
LEGEND_FONT_SIZE = 14


def format_float(value, precision=16):
    return f"{float(value):.{precision}f}"


def run_algorithm(algorithm, benchmark, seed):
    bounds = benchmark.get("bounds", BOUNDS)
    start_time = time.perf_counter()
    best_solution, best_value, history, used_seed = algorithm["runner"](
        objective_function=benchmark["function"],
        bounds=bounds,
        seed=seed,
        **algorithm["params"],
    )
    elapsed_time = time.perf_counter() - start_time

    return {
        "benchmark_id": benchmark["id"],
        "benchmark_name": benchmark["name"],
        "algorithm": algorithm["name"],
        "seed": used_seed,
        "best_value": best_value,
        "error": best_value - benchmark["optimal_value"],
        "time": elapsed_time,
        "history": history,
        "best_solution": best_solution,
    }


def calculate_statistics(results):
    best_values = np.array([item["best_value"] for item in results], dtype=float)
    errors = np.array([item["error"] for item in results], dtype=float)
    times = np.array([item["time"] for item in results], dtype=float)
    ddof = 1 if len(results) > 1 else 0

    best_index = np.argmin(best_values)
    worst_index = np.argmax(best_values)

    return {
        "run_times": len(results),
        "best": np.min(best_values),
        "worst": np.max(best_values),
        "mean": np.mean(best_values),
        "std": np.std(best_values, ddof=ddof),
        "median": np.median(best_values),
        "mean_error": np.mean(errors),
        "std_error": np.std(errors, ddof=ddof),
        "mean_time": np.mean(times),
        "best_seed": results[best_index]["seed"],
        "worst_seed": results[worst_index]["seed"],
    }


def print_statistics(benchmark, grouped_results):
    print("\n" + "=" * 80)
    print(benchmark["name"])
    print(f"理论最优值: {benchmark['optimal_value']}")

    for algorithm_name, results in grouped_results.items():
        stats = calculate_statistics(results)
        print(f"\n{algorithm_name} 统计结果")
        print("-" * 45)
        print(f"运行次数: {stats['run_times']}")
        print(f"最优值: {format_float(stats['best'])}")
        print(f"最差值: {format_float(stats['worst'])}")
        print(f"平均值: {format_float(stats['mean'])}")
        print(f"标准差: {format_float(stats['std'])}")
        print(f"中位数: {format_float(stats['median'])}")
        print(f"平均误差: {format_float(stats['mean_error'])}")
        print(f"误差标准差: {format_float(stats['std_error'])}")
        print(f"平均耗时: {stats['mean_time']:.6f} 秒")
        print(f"最优种子: {stats['best_seed']}")
        print(f"最差种子: {stats['worst_seed']}")


def save_results_to_csv(filename, grouped_results):
    rows = []
    for results in grouped_results.values():
        for run_index, item in enumerate(results, start=1):
            rows.append(
                {
                    "run": run_index,
                    "benchmark_id": item["benchmark_id"],
                    "benchmark_name": item["benchmark_name"],
                    "algorithm": item["algorithm"],
                    "seed": item["seed"],
                    "best_value": format_float(item["best_value"]),
                    "error": format_float(item["error"]),
                    "time": format_float(item["time"], precision=6),
                }
            )

    with open(filename, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "run",
                "benchmark_id",
                "benchmark_name",
                "algorithm",
                "seed",
                "best_value",
                "error",
                "time",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_rows_to_csv(filename, rows):
    if not rows:
        return

    with open(filename, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_best_value_curve(grouped_results, benchmark, filename):
    plt.figure(figsize=(10, 5))
    for algorithm_name, results in grouped_results.items():
        run_indexes = np.arange(1, len(results) + 1)
        best_values = [item["best_value"] for item in results]
        plt.plot(run_indexes, best_values, marker="o", markersize=3, linewidth=1.2, label=algorithm_name)

    plt.axhline(benchmark["optimal_value"], color="red", linestyle="--", linewidth=1.2, label="理论最优值")
    plt.xlabel("独立运行次数")
    plt.ylabel("最优目标函数值")
    plt.title(f"{benchmark['id']} 独立运行最优值对比")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_error_boxplot(grouped_results, benchmark, filename):
    labels = list(grouped_results.keys())
    errors = [[item["error"] for item in grouped_results[label]] for label in labels]

    plt.figure(figsize=(7, 5))
    plt.boxplot(errors, showmeans=True)
    plt.xticks(np.arange(1, len(labels) + 1), labels)
    plt.ylabel("与理论最优值的误差")
    plt.title(f"{benchmark['id']} 误差分布")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_average_convergence_curve(grouped_results, benchmark, filename):
    plt.figure(figsize=(10, 5))
    for algorithm_name, results in grouped_results.items():
        history = np.array([item["history"] for item in results], dtype=float)
        mean_history = np.mean(history, axis=0)
        plt.plot(mean_history, linewidth=2, label=algorithm_name)

    plt.axhline(benchmark["optimal_value"], color="red", linestyle="--", linewidth=1.2, label="理论最优值")
    plt.xlabel("迭代次数")
    plt.ylabel("平均最优目标函数值")
    plt.title(f"{benchmark['id']} 平均收敛曲线")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def save_plots(grouped_results, benchmark):
    benchmark_id = benchmark["id"].lower()
    plot_best_value_curve(grouped_results, benchmark, OUTPUT_DIR / f"{benchmark_id}_best_value_curve.png")
    plot_error_boxplot(grouped_results, benchmark, OUTPUT_DIR / f"{benchmark_id}_error_boxplot.png")
    plot_average_convergence_curve(grouped_results, benchmark, OUTPUT_DIR / f"{benchmark_id}_average_convergence.png")


def get_enabled_benchmarks():
    benchmarks = []
    for suite_name in ENABLED_SUITES:
        if suite_name not in BENCHMARK_SUITES:
            valid_names = ", ".join(BENCHMARK_SUITES)
            raise ValueError(f"未知测试集: {suite_name}，可选测试集: {valid_names}")
        benchmarks.extend(BENCHMARK_SUITES[suite_name])

    if ENABLED_FUNCTION_IDS:
        enabled_ids = set(ENABLED_FUNCTION_IDS)
        benchmarks = [item for item in benchmarks if item["id"] in enabled_ids]

    if not benchmarks:
        raise ValueError("没有选中任何测试函数，请检查 ENABLED_SUITES 或 ENABLED_FUNCTION_IDS")

    return benchmarks


def get_enabled_algorithms():
    if not ENABLED_ALGORITHMS:
        return ALGORITHMS

    enabled_names = set(ENABLED_ALGORITHMS)
    algorithms = [algorithm for algorithm in ALGORITHMS if algorithm["name"] in enabled_names]
    missing_names = enabled_names - {algorithm["name"] for algorithm in ALGORITHMS}
    if missing_names:
        valid_names = ", ".join(algorithm["name"] for algorithm in ALGORITHMS)
        raise ValueError(f"未知算法: {', '.join(sorted(missing_names))}，可选算法: {valid_names}")
    if not algorithms:
        raise ValueError("没有选中任何算法，请检查 ENABLED_ALGORITHMS")
    return algorithms


def print_run_configuration(benchmarks, algorithms):
    print("\n" + "=" * 80)
    print("单目标实验运行配置")
    print(f"测试集: {', '.join(ENABLED_SUITES)}")
    print(f"测试函数数量: {len(benchmarks)}")
    print(f"测试函数: {', '.join(benchmark['id'] for benchmark in benchmarks)}")
    print(f"算法数量: {len(algorithms)}")
    print(f"算法: {', '.join(algorithm['name'] for algorithm in algorithms)}")
    print(f"独立运行次数: {RUN_TIMES}")
    print(
        "公共参数: "
        f"bee={COMMON_PARAMS['bee']}, "
        f"max_iter={COMMON_PARAMS['max_iter']}, "
        f"limit={COMMON_PARAMS['limit']}"
    )


def print_progress(current, total, prefix="", width=32):
    ratio = current / total
    completed = int(width * ratio)
    bar = "#" * completed + "-" * (width - completed)
    print(f"\r{prefix} [{bar}] {current}/{total} {ratio * 100:6.2f}%", end="", flush=True)


def run_benchmark(benchmark, algorithms):
    seeds = np.random.SeedSequence().generate_state(RUN_TIMES)
    grouped_results = {algorithm["name"]: [] for algorithm in algorithms}

    print("\n" + "=" * 80)
    print(f"开始测试: {benchmark['name']}")
    print_progress(0, RUN_TIMES, prefix=benchmark["id"])

    for run_index, seed in enumerate(seeds, start=1):
        seed = int(seed)

        for algorithm in algorithms:
            result = run_algorithm(algorithm, benchmark, seed)
            grouped_results[algorithm["name"]].append(result)

        print_progress(run_index, RUN_TIMES, prefix=benchmark["id"])

    print()

    print_statistics(benchmark, grouped_results)
    save_results_to_csv(OUTPUT_DIR / f"{benchmark['id'].lower()}_results.csv", grouped_results)
    save_plots(grouped_results, benchmark)
    return grouped_results


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    benchmarks = get_enabled_benchmarks()
    algorithms = get_enabled_algorithms()
    print_run_configuration(benchmarks, algorithms)

    all_results = {}
    total_start_time = time.perf_counter()
    for benchmark in benchmarks:
        all_results[benchmark["id"]] = run_benchmark(benchmark, algorithms)

    wilcoxon_rows = []
    enabled_algorithm_names = [algorithm["name"] for algorithm in algorithms]
    if "IABC" in enabled_algorithm_names:
        for base_algorithm in [name for name in enabled_algorithm_names if name != "IABC"]:
            wilcoxon_rows.extend(
                save_wilcoxon_results(
                    OUTPUT_DIR / f"wilcoxon_{base_algorithm.lower()}_vs_iabc_results.csv",
                    all_results,
                    base_algorithm=base_algorithm,
                    improved_algorithm="IABC",
                    metrics=STATISTICAL_TEST_METRICS,
                )
            )
    save_rows_to_csv(OUTPUT_DIR / "wilcoxon_test_results.csv", wilcoxon_rows)
    rank_rows = save_average_rank_results(
        OUTPUT_DIR / "average_rank_results.csv",
        all_results,
        algorithms=enabled_algorithm_names,
        metrics=STATISTICAL_TEST_METRICS,
    )
    print_wilcoxon_overview(wilcoxon_rows)
    print_average_rank_overview(rank_rows)

    total_time = time.perf_counter() - total_start_time
    print("\n" + "=" * 80)
    print(f"全部测试完成，总耗时: {total_time:.2f} 秒")
    print(f"结果文件已保存到: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
