# -*- coding: utf-8 -*-

import sys
import time
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

MODULE_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODULE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from multi_objective.algorithms import CMMODE, MOABC, MODE, MOEAD, MOIABC, MOPSO, Yang_IGWO, Zhou_IMOABC
from multi_objective.mo_utils import calculate_hypervolume, non_dominated_mask, spacing_metric
from multi_objective.multiobjective_benchmarks import CEC2009_UF_BENCHMARKS, CEC2020_MMO_BENCHMARKS, ZDT_BENCHMARKS
from multi_objective.statistical_tests import (
    print_average_rank_overview,
    print_wilcoxon_overview,
    save_average_rank_results,
    save_wilcoxon_results,
)
from experiment_utils import (
    env_bool,
    env_csv,
    env_int,
    env_output_dir,
    print_progress,
    safe_filename_stem,
    save_rows_to_csv,
    select_enabled_items,
)


RUN_TIMES = env_int("MO_COMPARISON_RUN_TIMES", 30)
SEED_BASE = env_int("MO_COMPARISON_SEED_BASE", 20260723)
PARALLEL_WORKERS = env_int("MO_COMPARISON_WORKERS", 8)
SAVE_ARCHIVE_POINTS = env_bool("MO_COMPARISON_SAVE_ARCHIVE_POINTS", True)
SAVE_PLOTS = env_bool("MO_COMPARISON_SAVE_PLOTS", True)

# 全局测试开关：
# 1. ENABLED_SUITES 控制要跑哪些测试集，可选 "ZDT"、"CEC2009_UF"、"CEC2020_MMO"。
# 2. ENABLED_FUNCTION_IDS 控制要跑哪些具体函数，空列表表示不过滤。
#    例：只跑 ZDT1、UF1 和 MMF1 -> ENABLED_FUNCTION_IDS = ["ZDT1", "UF1", "MMF1"]
ENABLED_SUITES = env_csv("MO_COMPARISON_SUITES", ["ZDT", "CEC2009_UF", "CEC2020_MMO"])
ENABLED_FUNCTION_IDS = env_csv("MO_COMPARISON_FUNCTION_IDS")

STANDARD_EXPERIMENT_GROUP = {
    "name": "standard_algorithms",
    "output_dir": env_output_dir(
        "MO_COMPARISON_STANDARD_OUTPUT_DIR",
        MODULE_DIR / "mo_comparison_results_standard_algorithms",
        MODULE_DIR,
    ),
    "algorithms": ["MO-DE", "MOEA/D", "MOPSO", "MOABC", "MOIABC"],
}
IMPROVED_EXPERIMENT_GROUP = {
    "name": "improved_algorithms",
    "output_dir": env_output_dir(
        "MO_COMPARISON_IMPROVED_OUTPUT_DIR",
        MODULE_DIR / "mo_comparison_results_improved_algorithms",
        MODULE_DIR,
    ),
    "algorithms": ["Zhou-IMOABC", "Yang-IGWO", "CMMODE", "MOIABC"],
}
EXPERIMENT_GROUPS = [
    STANDARD_EXPERIMENT_GROUP,
    IMPROVED_EXPERIMENT_GROUP,
]
LEGACY_OUTPUT_DIRS = [
    MODULE_DIR / "mo_comparison_results",
    MODULE_DIR / "mo_comparison_results_baselines_no_moiabc",
]

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
}

MOIABC_BEST_PARAMS = {
    "tournament_size": 3,
    "elite_rate": 0.25,
    "elimination_rate": 0.25,
    "archive_guidance_rate": 0.40,
}

ALGORITHMS = [
    {
        "name": "MOABC",
        "runner": MOABC.multi_objective_abc,
        "params": COMMON_PARAMS,
    },
    {
        "name": "MO-DE",
        "runner": MODE.mode,
        "params": {
            "population_size": COMMON_PARAMS["bee"],
            "max_iter": COMMON_PARAMS["max_iter"],
            "archive_size": COMMON_PARAMS["archive_size"],
            "mutation_factor": 0.5,
            "crossover_rate": 0.9,
        },
    },
    {
        "name": "MOEA/D",
        "runner": MOEAD.moead,
        "params": {
            "population_size": COMMON_PARAMS["bee"],
            "max_iter": COMMON_PARAMS["max_iter"],
            "archive_size": COMMON_PARAMS["archive_size"],
            "neighborhood_size": 20,
            "mutation_factor": 0.5,
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
        "name": "Yang-IGWO",
        "runner": Yang_IGWO.yang_igwo,
        "params": {
            **COMMON_PARAMS,
            "convergence_exponent": 1.0,
        },
    },
    {
        "name": "CMMODE",
        "runner": CMMODE.cmmode,
        "params": {
            **COMMON_PARAMS,
            "mutation_factor": 0.5,
            "crossover_rate": 0.9,
            "elite_search_rate": 0.5,
        },
    },
    {
        "name": "MOIABC",
        "runner": MOIABC.multi_objective_iabc,
        "params": {
            **COMMON_PARAMS,
            **MOIABC_BEST_PARAMS,
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


def run_algorithm_task(task):
    algorithm, benchmark, seed, run_index = task
    result = run_algorithm(algorithm, benchmark, seed)
    result["run_index"] = run_index
    return result


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

    save_rows_to_csv(filename, rows)


def save_archive_points(filename, grouped_results):
    objective_count = max(
        item["archive_objectives"].shape[1] for results in grouped_results.values() for item in results
    )
    rows = []
    for results in grouped_results.values():
        for run_index, item in enumerate(results, start=1):
            for point_index, objective in enumerate(item["archive_objectives"], start=1):
                row = {
                    "algorithm": item["algorithm"],
                    "run": run_index,
                    "seed": item["seed"],
                    "point_index": point_index,
                }
                for objective_index in range(objective_count):
                    row[f"f{objective_index + 1}"] = (
                        objective[objective_index] if objective_index < len(objective) else ""
                    )
                rows.append(row)
    save_rows_to_csv(filename, rows)


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


def save_plots(grouped_results, benchmark, output_dir):
    benchmark_id = benchmark["id"].lower()
    plot_pareto_scatter(grouped_results, benchmark, output_dir / f"{benchmark_id}_pareto_scatter.png")
    plot_average_history(grouped_results, benchmark, output_dir / f"{benchmark_id}_average_history.png")


def get_enabled_benchmarks():
    return select_enabled_items(
        ENABLED_SUITES,
        BENCHMARK_SUITES,
        ENABLED_FUNCTION_IDS,
        suite_label="benchmark suite",
        empty_message="没有选中任何测试函数，请检查 ENABLED_SUITES 或 ENABLED_FUNCTION_IDS",
    )


def get_enabled_algorithms(algorithm_names):
    by_name = {algorithm["name"]: algorithm for algorithm in ALGORITHMS}
    missing_names = [name for name in algorithm_names if name not in by_name]
    if missing_names:
        valid_names = ", ".join(by_name)
        raise ValueError(f"Unknown algorithm: {', '.join(missing_names)}. Valid algorithms: {valid_names}")
    return [by_name[name] for name in algorithm_names]


def print_run_configuration(group, benchmarks, algorithms):
    print("\n" + "=" * 80)
    print("多目标实验运行配置")
    print(f"实验组: {group['name']}")
    print(f"输出目录: {group['output_dir'].resolve()}")
    print(f"测试集: {', '.join(ENABLED_SUITES)}")
    print(f"测试函数数量: {len(benchmarks)}")
    print(f"测试函数: {', '.join(benchmark['id'] for benchmark in benchmarks)}")
    print(f"算法数量: {len(algorithms)}")
    print(f"算法: {', '.join(algorithm['name'] for algorithm in algorithms)}")
    print(f"独立运行次数: {RUN_TIMES}")
    print(f"随机种子基准: {SEED_BASE}")
    print(f"并行进程数: {PARALLEL_WORKERS}")
    print(f"保存档案点: {'是' if SAVE_ARCHIVE_POINTS else '否'}")
    print(f"保存图像: {'是' if SAVE_PLOTS else '否'}")
    print(
        "公共参数: "
        f"bee={COMMON_PARAMS['bee']}, "
        f"max_iter={COMMON_PARAMS['max_iter']}, "
        f"limit={COMMON_PARAMS['limit']}, "
        f"archive_size={COMMON_PARAMS['archive_size']}"
    )


def run_benchmark(benchmark, algorithms, output_dir):
    seeds = np.random.SeedSequence(SEED_BASE).generate_state(RUN_TIMES)
    grouped_results = {algorithm["name"]: [] for algorithm in algorithms}
    tasks = []

    print("\n" + "=" * 80)
    print(f"开始测试: {benchmark['name']}")
    total_tasks = RUN_TIMES * len(algorithms)
    print_progress(0, total_tasks, prefix=benchmark["id"])

    for run_index, seed in enumerate(seeds, start=1):
        seed = int(seed)
        for algorithm in algorithms:
            tasks.append((algorithm, benchmark, seed, run_index))

    worker_count = min(PARALLEL_WORKERS, len(tasks))
    if worker_count <= 1:
        for task_index, task in enumerate(tasks, start=1):
            result = run_algorithm_task(task)
            grouped_results[result["algorithm"]].append(result)
            print_progress(task_index, total_tasks, prefix=benchmark["id"])
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(run_algorithm_task, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                grouped_results[result["algorithm"]].append(result)
                done += 1
                print_progress(done, total_tasks, prefix=benchmark["id"])

    print()
    for results in grouped_results.values():
        results.sort(key=lambda item: item["run_index"])

    print_statistics(benchmark, grouped_results)
    save_results_to_csv(output_dir / f"{benchmark['id'].lower()}_results.csv", grouped_results)
    if SAVE_ARCHIVE_POINTS:
        save_archive_points(output_dir / f"{benchmark['id'].lower()}_archive_points.csv", grouped_results)
    if SAVE_PLOTS:
        save_plots(grouped_results, benchmark, output_dir)
    return grouped_results


def prepare_output_dir(output_dir):
    resolved_module_dir = MODULE_DIR.resolve()
    resolved_output_dir = output_dir.resolve()
    if resolved_output_dir == resolved_module_dir or resolved_module_dir not in resolved_output_dir.parents:
        raise ValueError(f"Refuse to clear output directory outside multi_objective: {resolved_output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def clear_legacy_output_dirs():
    group_dirs = {group["output_dir"].resolve() for group in EXPERIMENT_GROUPS}
    for output_dir in LEGACY_OUTPUT_DIRS:
        if output_dir.resolve() in group_dirs or not output_dir.exists():
            continue
        prepare_output_dir(output_dir)
        output_dir.rmdir()


def run_experiment_group(group, benchmarks):
    output_dir = group["output_dir"]
    prepare_output_dir(output_dir)
    algorithms = get_enabled_algorithms(group["algorithms"])
    print_run_configuration(group, benchmarks, algorithms)

    all_results = {}
    total_start_time = time.perf_counter()
    for benchmark in benchmarks:
        all_results[benchmark["id"]] = run_benchmark(benchmark, algorithms, output_dir)

    wilcoxon_rows = []
    enabled_algorithm_names = [algorithm["name"] for algorithm in algorithms]
    if "MOIABC" in enabled_algorithm_names:
        for base_algorithm in [name for name in enabled_algorithm_names if name != "MOIABC"]:
            wilcoxon_rows.extend(
                save_wilcoxon_results(
                    output_dir / f"wilcoxon_{safe_filename_stem(base_algorithm)}_vs_moiabc_results.csv",
                    all_results,
                    base_algorithm=base_algorithm,
                    improved_algorithm="MOIABC",
                    metrics=STATISTICAL_TEST_METRICS,
                )
            )
    save_rows_to_csv(output_dir / "wilcoxon_test_results.csv", wilcoxon_rows)
    rank_rows = save_average_rank_results(
        output_dir / "average_rank_results.csv",
        all_results,
        algorithms=enabled_algorithm_names,
        metrics=STATISTICAL_TEST_METRICS,
    )
    print_wilcoxon_overview(wilcoxon_rows)
    print_average_rank_overview(rank_rows)

    total_time = time.perf_counter() - total_start_time
    print("\n" + "=" * 80)
    print(f"{group['name']} 多目标测试完成，总耗时: {total_time:.2f} 秒")
    print(f"结果文件已保存到: {output_dir.resolve()}")


def run_single_experiment_group(group):
    benchmarks = get_enabled_benchmarks()
    run_experiment_group(group, benchmarks)


def main():
    benchmarks = get_enabled_benchmarks()
    clear_legacy_output_dirs()
    for group in EXPERIMENT_GROUPS:
        run_experiment_group(group, benchmarks)


if __name__ == "__main__":
    main()
