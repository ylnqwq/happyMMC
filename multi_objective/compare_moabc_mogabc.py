# -*- coding: utf-8 -*-

import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

MODULE_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODULE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from multi_objective.algorithms import MOABC, MOIABC, MOPSO, NSGA2, Zhao_IMOABC, Zhou_IMOABC
from multi_objective.mo_utils import non_dominated_mask, spacing_metric
from multi_objective.multiobjective_benchmarks import ZDT_BENCHMARKS
from multi_objective.statistical_tests import (
    print_average_rank_overview,
    print_wilcoxon_overview,
    save_average_rank_results,
    save_wilcoxon_results,
)


RUN_TIMES = 10
OUTPUT_DIR = Path(__file__).resolve().parent / "mo_comparison_results"

# 全局测试开关：
# 1. ENABLED_SUITES 控制要跑哪些测试集，可选 "ZDT"。
# 2. ENABLED_FUNCTION_IDS 控制要跑哪些具体函数，空列表表示不过滤。
#    例：只跑 ZDT1 和 ZDT4 -> ENABLED_FUNCTION_IDS = ["ZDT1", "ZDT4"]
ENABLED_SUITES = ["ZDT"]
ENABLED_FUNCTION_IDS = []
# 可选算法 MOABC, NSGA-II, MOPSO, Zhou-IMOABC, Zhao-IMOABC, MOIABC
ENABLED_ALGORITHMS = ["MOABC","MOIABC"]

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
        "name": "NSGA-II",
        "runner": NSGA2.nsga2,
        "params": {
            "population_size": COMMON_PARAMS["bee"],
            "max_iter": COMMON_PARAMS["max_iter"],
            "archive_size": COMMON_PARAMS["archive_size"],
            "crossover_rate": 0.9,
        },
    },
    {
        "name": "MOPSO",
        "runner": MOPSO.mopso,
        "params": {
            "swarm_size": COMMON_PARAMS["bee"],
            "max_iter": COMMON_PARAMS["max_iter"],
            "archive_size": COMMON_PARAMS["archive_size"],
            "inertia": 0.4,
            "cognitive": 1.5,
            "social": 1.5,
        },
    },
    {
        "name": "Zhou-IMOABC",
        "runner": Zhou_IMOABC.zhou_imoabc,
        "params": COMMON_PARAMS,
    },
    {
        "name": "Zhao-IMOABC",
        "runner": Zhao_IMOABC.zhao_imoabc,
        "params": {
            **COMMON_PARAMS,
            "elimination_rate": 0.1,
        },
    },
    {
        "name": "MOIABC",
        "runner": MOIABC.multi_objective_iabc,
        "params": {
            **COMMON_PARAMS,
            "tournament_size": 3,
            "elite_rate": 0.15,
            "elimination_rate": 0.15,
        },
    },
]
STATISTICAL_TEST_METRICS = [
    ("hypervolume", True),
    ("spacing", False),
    ("best_sum", False),
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


def three_objective_hypervolume(objectives, reference_point):
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
        if width <= 0.0:
            continue
        slice_points = points[: index + 1, 1:3]
        hypervolume += width * two_objective_hypervolume(slice_points, reference_point[1:3])
    return float(hypervolume)


def calculate_hypervolume(objectives, reference_point):
    objective_count = np.asarray(objectives).shape[1]
    if objective_count == 2:
        return two_objective_hypervolume(objectives, reference_point)
    if objective_count == 3:
        return three_objective_hypervolume(objectives, reference_point)
    raise ValueError(f"暂不支持 {objective_count} 目标超体积计算")


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
        "min_f3": float(np.min(archive_objectives[:, 2])) if archive_objectives.shape[1] >= 3 else "",
        "spacing": spacing_metric(archive_objectives),
        "hypervolume": calculate_hypervolume(archive_objectives, benchmark["reference_point"]),
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
                    "min_f3": item["min_f3"],
                    "spacing": item["spacing"],
                    "hypervolume": item["hypervolume"],
                    "time": item["time"],
                }
            )

    with open(filename, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_rows_to_csv(filename, rows):
    if not rows:
        return

    with open(filename, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_archive_points(filename, grouped_results):
    objective_count = max(
        item["archive_objectives"].shape[1] for results in grouped_results.values() for item in results
    )
    fieldnames = ["algorithm", "run", "seed", "point_index"]
    fieldnames.extend(f"f{index + 1}" for index in range(objective_count))

    with open(filename, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for results in grouped_results.values():
            for run_index, item in enumerate(results, start=1):
                for point_index, objective in enumerate(item["archive_objectives"], start=1):
                    row = {
                        "algorithm": item["algorithm"],
                        "run": run_index,
                        "seed": item["seed"],
                        "point_index": point_index,
                    }
                    for objective_index, value in enumerate(objective, start=1):
                        row[f"f{objective_index}"] = value
                    writer.writerow(row)


def plot_pareto_scatter(grouped_results, benchmark, filename):
    if benchmark["objective_count"] == 3:
        figure = plt.figure(figsize=(7, 6))
        axis = figure.add_subplot(111, projection="3d")
        for algorithm_name, results in grouped_results.items():
            objectives = np.vstack([item["archive_objectives"] for item in results])
            objectives = objectives[non_dominated_mask(objectives)]
            axis.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2], s=14, alpha=0.65, label=algorithm_name)

        axis.set_xlabel("目标 f1")
        axis.set_ylabel("目标 f2")
        axis.set_zlabel("目标 f3")
        axis.set_title(f"{benchmark['id']} Pareto 非支配解散点图")
        axis.legend(fontsize=LEGEND_FONT_SIZE)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        return

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
    print("多目标实验运行配置")
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
        f"limit={COMMON_PARAMS['limit']}, "
        f"archive_size={COMMON_PARAMS['archive_size']}"
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
    save_archive_points(OUTPUT_DIR / f"{benchmark['id'].lower()}_archive_points.csv", grouped_results)
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
    if "MOIABC" in enabled_algorithm_names:
        for base_algorithm in [name for name in enabled_algorithm_names if name != "MOIABC"]:
            wilcoxon_rows.extend(
                save_wilcoxon_results(
                    OUTPUT_DIR / f"wilcoxon_{base_algorithm.lower()}_vs_moiabc_results.csv",
                    all_results,
                    base_algorithm=base_algorithm,
                    improved_algorithm="MOIABC",
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
    print(f"全部多目标测试完成，总耗时: {total_time:.2f} 秒")
    print(f"结果文件已保存到: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
