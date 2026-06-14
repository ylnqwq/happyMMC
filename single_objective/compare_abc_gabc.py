# -*- coding: utf-8 -*-

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ABC
import GABC
from cec2017_official import OFFICIAL_CEC2017_BENCHMARKS
from cec2022_official import OFFICIAL_CEC2022_BENCHMARKS


RUN_TIMES = 1
BOUNDS = [(-100, 100)] * 10
OUTPUT_DIR = Path(__file__).resolve().parent / "comparison_results"

# 全局测试开关：
# 1. ENABLED_SUITES 控制要跑哪些测试集，可选 "CEC2017"、"CEC2022"。
#    例：只跑 CEC2022 -> ENABLED_SUITES = ["CEC2022"]
# 2. ENABLED_FUNCTION_IDS 控制要跑哪些具体函数，空列表表示不过滤。
#    例：只跑 CEC2022_F1 和 CEC2022_F6 -> ENABLED_FUNCTION_IDS = ["CEC2022_F1", "CEC2022_F6"]
ENABLED_SUITES = ["CEC2017","CEC2022"]
ENABLED_FUNCTION_IDS = []

BENCHMARK_SUITES = {
    "CEC2017": OFFICIAL_CEC2017_BENCHMARKS,
    "CEC2022": OFFICIAL_CEC2022_BENCHMARKS,
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
        "name": "GABC",
        "runner": GABC.gabc,
        "params": {
            **COMMON_PARAMS,
            "tournament_size": 3,
            "elite_rate": 0.15,
            "elimination_rate": 0.15,
        },
    },
]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False


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
        print(f"最优值: {stats['best']:.16e}")
        print(f"最差值: {stats['worst']:.16e}")
        print(f"平均值: {stats['mean']:.16e}")
        print(f"标准差: {stats['std']:.16e}")
        print(f"中位数: {stats['median']:.16e}")
        print(f"平均误差: {stats['mean_error']:.16e}")
        print(f"误差标准差: {stats['std_error']:.16e}")
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
                    "best_value": item["best_value"],
                    "error": item["error"],
                    "time": item["time"],
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
    plt.legend()
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
    plt.legend()
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


def print_progress(current, total, prefix="", width=32):
    ratio = current / total
    completed = int(width * ratio)
    bar = "#" * completed + "-" * (width - completed)
    print(f"\r{prefix} [{bar}] {current}/{total} {ratio * 100:6.2f}%", end="", flush=True)


def run_benchmark(benchmark):
    seeds = np.random.SeedSequence().generate_state(RUN_TIMES)
    grouped_results = {algorithm["name"]: [] for algorithm in ALGORITHMS}

    print("\n" + "=" * 80)
    print(f"开始测试: {benchmark['name']}")
    print_progress(0, RUN_TIMES, prefix=benchmark["id"])

    for run_index, seed in enumerate(seeds, start=1):
        seed = int(seed)

        for algorithm in ALGORITHMS:
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

    total_start_time = time.perf_counter()
    for benchmark in benchmarks:
        run_benchmark(benchmark)

    total_time = time.perf_counter() - total_start_time
    print("\n" + "=" * 80)
    print(f"全部测试完成，总耗时: {total_time:.2f} 秒")
    print(f"结果文件已保存到: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
