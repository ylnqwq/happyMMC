# -*- coding: utf-8 -*-

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import MOABC
import MOGABC
from mo_utils import non_dominated_mask, spacing_metric
from multiobjective_benchmarks import ZDT_BENCHMARKS


RUN_TIMES = 10
OUTPUT_DIR = Path(__file__).resolve().parent / "mo_comparison_results"

# 全局测试开关：
# 1. ENABLED_SUITES 控制要跑哪些测试集，可选 "ZDT"。
# 2. ENABLED_FUNCTION_IDS 控制要跑哪些具体函数，空列表表示不过滤。
#    例：只跑 ZDT1 和 ZDT4 -> ENABLED_FUNCTION_IDS = ["ZDT1", "ZDT4"]
ENABLED_SUITES = ["ZDT"]
ENABLED_FUNCTION_IDS = []

BENCHMARK_SUITES = {
    "ZDT": ZDT_BENCHMARKS,
}

COMMON_PARAMS = {
    "bee": 75,
    "max_iter": 750,
    "limit": 150,
    "archive_size": 100,
}

ALGORITHMS = [
    {
        "name": "MOABC",
        "runner": MOABC.multi_objective_abc,
        "params": COMMON_PARAMS,
    },
    {
        "name": "MOGABC",
        "runner": MOGABC.multi_objective_gabc,
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
LEGEND_FONT_SIZE = 14


def two_objective_hypervolume(objectives, reference_point):
    objectives = np.asarray(objectives, dtype=float)
    reference_point = np.asarray(reference_point, dtype=float)
    valid_mask = np.all(objectives < reference_point, axis=1)
    points = objectives[valid_mask]
    if len(points) == 0:
        return 0.0

    points = points[non_dominated_mask(points)]
    points = points[np.argsort(points[:, 0])]

    hypervolume = 0.0
    for index, point in enumerate(points):
        next_x = points[index + 1, 0] if index + 1 < len(points) else reference_point[0]
        width = max(0.0, next_x - point[0])
        height = max(0.0, reference_point[1] - point[1])
        hypervolume += width * height
    return float(hypervolume)


def run_algorithm(algorithm, benchmark, seed):
    start_time = time.perf_counter()
    archive_solutions, archive_objectives, history, used_seed = algorithm["runner"](
        objective_function=benchmark["function"],
        bounds=benchmark["bounds"],
        seed=seed,
        **algorithm["params"],
    )
    elapsed_time = time.perf_counter() - start_time

    objective_sums = np.sum(archive_objectives, axis=1)
    best_sum_index = int(np.argmin(objective_sums))

    return {
        "benchmark_id": benchmark["id"],
        "benchmark_name": benchmark["name"],
        "algorithm": algorithm["name"],
        "seed": used_seed,
        "archive_solutions": archive_solutions,
        "archive_objectives": archive_objectives,
        "archive_size": len(archive_objectives),
        "best_sum": float(objective_sums[best_sum_index]),
        "mean_sum": float(np.mean(objective_sums)),
        "min_f1": float(np.min(archive_objectives[:, 0])),
        "min_f2": float(np.min(archive_objectives[:, 1])),
        "spacing": spacing_metric(archive_objectives),
        "hypervolume": two_objective_hypervolume(archive_objectives, benchmark["reference_point"]),
        "time": elapsed_time,
        "history": history,
    }


def calculate_statistics(results):
    archive_sizes = np.array([item["archive_size"] for item in results], dtype=float)
    best_sums = np.array([item["best_sum"] for item in results], dtype=float)
    spacings = np.array([item["spacing"] for item in results], dtype=float)
    hypervolumes = np.array([item["hypervolume"] for item in results], dtype=float)
    times = np.array([item["time"] for item in results], dtype=float)
    ddof = 1 if len(results) > 1 else 0

    best_sum_index = int(np.argmin(best_sums))
    best_hv_index = int(np.argmax(hypervolumes))

    return {
        "run_times": len(results),
        "mean_archive_size": np.mean(archive_sizes),
        "mean_best_sum": np.mean(best_sums),
        "std_best_sum": np.std(best_sums, ddof=ddof),
        "best_sum": np.min(best_sums),
        "mean_spacing": np.mean(spacings),
        "mean_hypervolume": np.mean(hypervolumes),
        "std_hypervolume": np.std(hypervolumes, ddof=ddof),
        "best_hypervolume": np.max(hypervolumes),
        "mean_time": np.mean(times),
        "best_sum_seed": results[best_sum_index]["seed"],
        "best_hypervolume_seed": results[best_hv_index]["seed"],
    }


def print_statistics(benchmark, grouped_results):
    print("\n" + "=" * 80)
    print(benchmark["name"])
    print("说明: 多目标结果是一组 Pareto 非支配解，不再只有一个理论最优值。")
    print(f"超体积参考点: {benchmark['reference_point'].tolist()}")

    for algorithm_name, results in grouped_results.items():
        stats = calculate_statistics(results)
        print(f"\n{algorithm_name} 统计结果")
        print("-" * 45)
        print(f"运行次数: {stats['run_times']}")
        print(f"平均非支配解数量: {stats['mean_archive_size']:.2f}")
        print(f"平均最小目标和: {stats['mean_best_sum']:.16e}")
        print(f"目标和标准差: {stats['std_best_sum']:.16e}")
        print(f"最好目标和: {stats['best_sum']:.16e}")
        print(f"平均间距指标: {stats['mean_spacing']:.16e}")
        print(f"平均超体积: {stats['mean_hypervolume']:.16e}")
        print(f"超体积标准差: {stats['std_hypervolume']:.16e}")
        print(f"最好超体积: {stats['best_hypervolume']:.16e}")
        print(f"平均耗时: {stats['mean_time']:.6f} 秒")
        print(f"最好目标和种子: {stats['best_sum_seed']}")
        print(f"最好超体积种子: {stats['best_hypervolume_seed']}")


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
                    "archive_size": item["archive_size"],
                    "best_sum": item["best_sum"],
                    "mean_sum": item["mean_sum"],
                    "min_f1": item["min_f1"],
                    "min_f2": item["min_f2"],
                    "spacing": item["spacing"],
                    "hypervolume": item["hypervolume"],
                    "time": item["time"],
                }
            )

    with open(filename, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_archive_points(filename, grouped_results):
    fieldnames = ["algorithm", "run", "seed", "point_index", "f1", "f2"]
    with open(filename, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for results in grouped_results.values():
            for run_index, item in enumerate(results, start=1):
                for point_index, objective in enumerate(item["archive_objectives"], start=1):
                    writer.writerow(
                        {
                            "algorithm": item["algorithm"],
                            "run": run_index,
                            "seed": item["seed"],
                            "point_index": point_index,
                            "f1": objective[0],
                            "f2": objective[1],
                        }
                    )


def plot_pareto_scatter(grouped_results, benchmark, filename):
    plt.figure(figsize=(7, 6))
    for algorithm_name, results in grouped_results.items():
        objectives = np.vstack([item["archive_objectives"] for item in results])
        objectives = objectives[non_dominated_mask(objectives)]
        plt.scatter(objectives[:, 0], objectives[:, 1], s=14, alpha=0.65, label=algorithm_name)

    plt.xlabel("目标 f1")
    plt.ylabel("目标 f2")
    plt.title(f"{benchmark['id']} Pareto 非支配解散点图")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_average_history(grouped_results, benchmark, filename):
    plt.figure(figsize=(10, 5))
    for algorithm_name, results in grouped_results.items():
        history = np.array([item["history"] for item in results], dtype=float)
        mean_history = np.mean(history, axis=0)
        plt.plot(mean_history, linewidth=2, label=algorithm_name)

    plt.xlabel("迭代次数")
    plt.ylabel("档案中最小目标和")
    plt.title(f"{benchmark['id']} 平均收敛参考曲线")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def save_plots(grouped_results, benchmark):
    benchmark_id = benchmark["id"].lower()
    plot_pareto_scatter(grouped_results, benchmark, OUTPUT_DIR / f"{benchmark_id}_pareto_scatter.png")
    plot_average_history(grouped_results, benchmark, OUTPUT_DIR / f"{benchmark_id}_average_history.png")


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
    save_archive_points(OUTPUT_DIR / f"{benchmark['id'].lower()}_archive_points.csv", grouped_results)
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
    print(f"全部多目标测试完成，总耗时: {total_time:.2f} 秒")
    print(f"结果文件已保存到: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
