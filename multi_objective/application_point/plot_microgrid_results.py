# -*- coding: utf-8 -*-
"""Plot figures from saved microgrid dispatch result CSV files."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODULE_DIR.parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiment_utils import env_output_dir
from multi_objective.application_point.microgrid_dispatch_model import BUY_PRICE, PARAMS, SELL_PRICE


RESULTS_DIR = env_output_dir("APP_RESULTS_DIR", MODULE_DIR / "results", MODULE_DIR)
OUTPUT_DIR = env_output_dir("APP_PLOT_OUTPUT_DIR", RESULTS_DIR, MODULE_DIR)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
LEGEND_FONT_SIZE = 16
ALGORITHM_ORDER = ["MOIABC", "MOABC", "MOPSO", "MOEA/D", "MO-DE"]
ALGORITHM_COLORS = {
    "MOIABC": "#d66b5f",
    "MOABC": "#4e8fc7",
    "MOPSO": "#a7dce0",
    "MOEA/D": "#7fa6d9",
    "MO-DE": "#f1b183",
}
ALGORITHM_MARKERS = {
    "MOIABC": "P",
    "MOABC": "D",
    "MOPSO": "^",
    "MOEA/D": "s",
    "MO-DE": "o",
}


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as file:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(file)]


def read_dict_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def set_common_style(axis):
    axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def algorithm_title_prefix(algorithm_name):
    return "" if algorithm_name is None else f"{algorithm_name} "


def algorithm_sort_key(algorithm_name):
    try:
        return (0, ALGORITHM_ORDER.index(algorithm_name))
    except ValueError:
        return (1, algorithm_name)


def read_algorithm_name(input_dir):
    summary_path = input_dir / "multi_objective_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return summary.get("algorithm", input_dir.name)
    return input_dir.name.replace("MOEA_D", "MOEA/D")


def convergence_main_start_index(iterations, mean_values, max_values):
    final_mean = mean_values[-1]
    main_threshold = max(final_mean * 1.5, final_mean + 50.0)
    main_candidates = np.where(max_values <= main_threshold)[0]
    if len(main_candidates):
        main_start_index = int(main_candidates[0])
    else:
        mean_candidates = np.where(mean_values <= main_threshold)[0]
        main_start_index = int(mean_candidates[0]) if len(mean_candidates) else max(0, len(iterations) // 20)
    return min(main_start_index, len(iterations) - 2)


def plot_power_dispatch(rows, path, algorithm_name=None):
    hours = np.array([row.get("time_h", row["hour"]) for row in rows], dtype=float)
    time_step = float(np.median(np.diff(hours))) if len(hours) > 1 else PARAMS.time_step_hours
    bar_width = time_step * 0.72
    positive_series = [
        ("风电", np.array([row["wind_kw"] for row in rows]), "#4c78a8"),
        ("光伏", np.array([row["pv_kw"] for row in rows]), "#f2c14e"),
        ("微型燃气轮机", np.array([row["diesel_kw"] for row in rows]), "#8f6f4e"),
        ("主网购电", np.array([row["grid_buy_kw"] for row in rows]), "#59a14f"),
        ("储能放电", np.array([row["battery_discharge_kw"] for row in rows]), "#e15759"),
    ]
    negative_series = [
        ("储能充电", -np.array([row["battery_charge_kw"] for row in rows]), "#b07aa1"),
        ("主网售电", -np.array([row["grid_sell_kw"] for row in rows]), "#76b7b2"),
    ]
    load = np.array([row["load_kw"] for row in rows], dtype=float)

    fig, axis = plt.subplots(figsize=(12, 7))
    bottom = np.zeros_like(hours)
    for label, values, color in positive_series:
        axis.bar(hours, values, bottom=bottom, width=bar_width, label=label, color=color, edgecolor="white", linewidth=0.35)
        bottom += values

    bottom = np.zeros_like(hours)
    for label, values, color in negative_series:
        axis.bar(hours, values, bottom=bottom, width=bar_width, label=label, color=color, edgecolor="white", linewidth=0.35)
        bottom += values

    axis.plot(hours, load, color="#222222", marker="o", markersize=4, linewidth=1.8, label="负荷")
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set_xlim(0, 24.5)
    axis.set_xticks(np.arange(0, 25, 2))
    axis.set_xlabel("时间 / h")
    axis.set_ylabel("功率 / kW")
    set_common_style(axis)
    axis.legend(
        ncol=2,
        fontsize=LEGEND_FONT_SIZE,
        frameon=True,
        loc="upper left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_soc_curve(rows, path, algorithm_name=None):
    end_hours = np.array([row.get("time_h", row["hour"]) for row in rows], dtype=float)
    hours = np.concatenate([[0.0], end_hours])
    soc = np.array([rows[0]["soc_start"]] + [row["soc_end"] for row in rows], dtype=float)

    fig, axis = plt.subplots(figsize=(9, 6))
    axis.plot(hours, soc, color="#2f6f9f", marker="o", markersize=4, linewidth=2.0)
    axis.axhline(PARAMS.soc_min, color="#d95f02", linestyle="--", linewidth=1.2, label="荷电状态下限")
    axis.axhline(PARAMS.soc_max, color="#1b9e77", linestyle="--", linewidth=1.2, label="荷电状态上限")
    axis.set_xlim(0, 24)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(np.arange(0, 25, 2))
    axis.set_xlabel("时段 / h")
    axis.set_ylabel("荷电状态")
    set_common_style(axis)
    axis.legend(
        fontsize=LEGEND_FONT_SIZE,
        ncol=2,
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_average_convergence(rows, path, algorithm_name=None):
    iterations = np.array([row["iteration"] for row in rows], dtype=float)
    mean_values = np.array([row["mean_best_sum"] for row in rows], dtype=float)
    min_values = np.array([row["min_best_sum"] for row in rows], dtype=float)
    max_values = np.array([row["max_best_sum"] for row in rows], dtype=float)

    main_start_index = convergence_main_start_index(iterations, mean_values, max_values)

    main_iterations = iterations[main_start_index:]
    main_mean_values = mean_values[main_start_index:]
    main_min_values = min_values[main_start_index:]
    main_max_values = max_values[main_start_index:]

    fig, axis = plt.subplots(figsize=(10, 6))
    axis.fill_between(main_iterations, main_min_values, main_max_values, color="#b8c7d9", alpha=0.35, label="最小-最大范围")
    axis.plot(main_iterations, main_mean_values, color="#2f4b7c", linewidth=1.9, label="平均最优目标和")
    axis.set_xlim(main_iterations[0], main_iterations[-1])
    main_low = float(np.min(main_min_values))
    main_high = float(np.max(main_max_values))
    margin = max((main_high - main_low) * 0.08, 1.0)
    axis.set_ylim(main_low - margin, main_high + margin)
    axis.set_xlabel("迭代次数 / 次")
    axis.set_ylabel("最优目标和")
    set_common_style(axis)
    axis.legend(
        loc="upper right",
        ncol=2,
        fontsize=LEGEND_FONT_SIZE,
    )

    inset = axis.inset_axes([0.30, 0.16, 0.50, 0.42])
    early_slice = slice(0, main_start_index + 1)
    inset.fill_between(iterations[early_slice], min_values[early_slice], max_values[early_slice], color="#b8c7d9", alpha=0.35)
    inset.plot(iterations[early_slice], mean_values[early_slice], color="#2f4b7c", linewidth=1.2)
    inset.set_yscale("log")
    inset.set_xlabel("前期迭代", fontsize=9)
    inset.set_ylabel("目标和", fontsize=9)
    inset.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    inset.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_cost_breakdown(rows, path, algorithm_name=None):
    diesel_linear_cost = sum(row["diesel_kw"] for row in rows) * PARAMS.diesel_unit_cost * PARAMS.time_step_hours
    diesel_quadratic_cost = 0.0
    if PARAMS.enable_extra_complexity:
        diesel_quadratic_cost = (
            sum(row["diesel_kw"] ** 2 for row in rows)
            * PARAMS.diesel_quadratic_fuel_cost
            * PARAMS.time_step_hours
        )
    diesel_cost = diesel_linear_cost + diesel_quadratic_cost
    battery_cost = sum(abs(row["battery_kw"]) for row in rows) * PARAMS.battery_om_cost * PARAMS.time_step_hours
    buy_cost = sum(row["grid_buy_kw"] * BUY_PRICE[index] for index, row in enumerate(rows)) * PARAMS.time_step_hours
    sell_revenue = sum(row["grid_sell_kw"] * SELL_PRICE[index] for index, row in enumerate(rows)) * PARAMS.time_step_hours
    net_cost = diesel_cost + battery_cost + buy_cost - sell_revenue

    labels = ["微型燃气轮机", "储能", "主网购电", "售电收益", "净成本"]
    values = [diesel_cost, battery_cost, buy_cost, -sell_revenue, net_cost]
    colors = ["#8f6f4e", "#b07aa1", "#59a14f", "#76b7b2", "#e15759"]

    fig, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set_ylabel("成本 / 元")
    set_common_style(axis)
    for bar, value in zip(bars, values):
        va = "bottom" if value >= 0 else "top"
        offset = 3 if value >= 0 else -3
        axis.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width() / 2, value), xytext=(0, offset),
                      textcoords="offset points", ha="center", va=va, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_curtailment_curve(rows, path, algorithm_name=None):
    if "wind_curtail_kw" not in rows[0] or "pv_curtail_kw" not in rows[0]:
        return

    hours = np.array([row.get("time_h", row["hour"]) for row in rows], dtype=float)
    time_step = float(np.median(np.diff(hours))) if len(hours) > 1 else PARAMS.time_step_hours
    wind_curtail = np.array([row["wind_curtail_kw"] for row in rows], dtype=float)
    pv_curtail = np.array([row["pv_curtail_kw"] for row in rows], dtype=float)

    fig, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(hours, wind_curtail, width=time_step * 0.72, color="#4c78a8", edgecolor="white", linewidth=0.35, label="弃风")
    axis.bar(
        hours,
        pv_curtail,
        bottom=wind_curtail,
        width=time_step * 0.72,
        color="#f2c14e",
        edgecolor="white",
        linewidth=0.35,
        label="弃光",
    )
    axis.set_xlim(0, 24.5)
    axis.set_xticks(np.arange(0, 25, 2))
    axis.set_xlabel("时间 / h")
    axis.set_ylabel("弃电功率 / kW")
    set_common_style(axis)
    axis.legend(fontsize=LEGEND_FONT_SIZE, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_pareto_front(pareto_rows, reference_rows, path, algorithm_name=None):
    economic_costs = np.array([float(row["economic_cost"]) for row in pareto_rows], dtype=float)
    environment_costs = np.array([float(row["environment_cost"]) for row in pareto_rows], dtype=float)
    compromise_mask = np.array([row.get("is_compromise", "0") == "1" for row in pareto_rows], dtype=bool)

    fig, axis = plt.subplots(figsize=(7, 5.5))
    if reference_rows and len(reference_rows) > len(pareto_rows):
        reference_economic = np.array([float(row["economic_cost"]) for row in reference_rows], dtype=float)
        reference_environment = np.array([float(row["environment_cost"]) for row in reference_rows], dtype=float)
        axis.scatter(reference_economic, reference_environment, s=18, alpha=0.35, label="经验参考前沿")

    axis.scatter(economic_costs, environment_costs, s=30, alpha=0.85, label=f"{algorithm_title_prefix(algorithm_name)}外部档案")
    if np.any(compromise_mask):
        axis.scatter(
            economic_costs[compromise_mask][0],
            environment_costs[compromise_mask][0],
            s=90,
            marker="*",
            label="折中方案",
        )

    axis.set_xlabel("经济成本 / 元")
    axis.set_ylabel("环境成本 / 元")
    set_common_style(axis)
    axis.legend(fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_selected_history(history_rows, summary_path, path):
    selected_run = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        selected_run = int(summary["selected_run"])

    if selected_run is None:
        selected_run = int(float(history_rows[0]["run"]))
    selected_rows = [row for row in history_rows if int(float(row["run"])) == selected_run]
    if not selected_rows:
        selected_rows = history_rows

    iterations = np.array([float(row["iteration"]) for row in selected_rows], dtype=float)
    best_sums = np.array([float(row["best_sum"]) for row in selected_rows], dtype=float)

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(iterations, best_sums, linewidth=1.6, label="最优目标和")
    axis.set_xlabel("迭代次数 / 次")
    axis.set_ylabel("最优目标和")
    set_common_style(axis)
    axis.legend(fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_igd_igd_plus_comparison(metric_sets, path):
    algorithms = sorted(metric_sets, key=algorithm_sort_key)
    metrics = [
        ("IGD", "IGD 指标值"),
        ("IGD+", "IGD+ 指标值"),
    ]
    colors = [ALGORITHM_COLORS.get(algorithm_name, "#9aa1a8") for algorithm_name in algorithms]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
    for axis, (metric_name, ylabel) in zip(axes, metrics):
        data = []
        for algorithm_name in algorithms:
            values = np.array([float(row[metric_name]) for row in metric_sets[algorithm_name]], dtype=float)
            values = np.where(values <= 0.0, 1.0e-12, values)
            data.append(values)

        box = axis.boxplot(
            data,
            labels=algorithms,
            patch_artist=True,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "#222222", "markeredgecolor": "#222222", "markersize": 4},
            medianprops={"color": "#222222", "linewidth": 1.2},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)

        for index, values in enumerate(data, start=1):
            jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
            axis.scatter(
                np.full(len(values), index, dtype=float) + jitter,
                values,
                s=18,
                alpha=0.65,
                color=colors[(index - 1) % len(colors)],
                edgecolor="none",
            )

        axis.set_yscale("log")
        axis.set_xlabel("算法")
        axis.set_ylabel(ylabel)
        set_common_style(axis)

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_algorithm_pareto_comparison(pareto_sets, reference_rows, path):
    fig, axis = plt.subplots(figsize=(8, 6))
    if reference_rows:
        reference_economic = np.array([float(row["economic_cost"]) for row in reference_rows], dtype=float)
        reference_environment = np.array([float(row["environment_cost"]) for row in reference_rows], dtype=float)
        axis.scatter(
            reference_economic,
            reference_environment,
            s=16,
            c="#9aa1a8",
            alpha=0.28,
            label="经验参考前沿",
        )

    for algorithm_name in sorted(pareto_sets, key=algorithm_sort_key):
        rows = pareto_sets[algorithm_name]
        economic_costs = np.array([float(row["economic_cost"]) for row in rows], dtype=float)
        environment_costs = np.array([float(row["environment_cost"]) for row in rows], dtype=float)
        compromise_mask = np.array([row.get("is_compromise", "0") == "1" for row in rows], dtype=bool)
        color = ALGORITHM_COLORS.get(algorithm_name)

        axis.scatter(
            economic_costs,
            environment_costs,
            s=28,
            alpha=0.82,
            marker=ALGORITHM_MARKERS.get(algorithm_name, "o"),
            color=color,
            label=f"{algorithm_name} 外部档案",
        )
        if np.any(compromise_mask):
            axis.scatter(
                economic_costs[compromise_mask][0],
                environment_costs[compromise_mask][0],
                s=95,
                marker="*",
                color=color,
                edgecolor="#222222",
                linewidth=0.5,
                label=f"{algorithm_name} 折中方案",
            )

    axis.set_xlabel("经济成本 / 元")
    axis.set_ylabel("环境成本 / 元")
    set_common_style(axis)
    axis.legend(fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_algorithm_convergence_comparison(convergence_sets, path):
    series = []
    for algorithm_name in sorted(convergence_sets, key=algorithm_sort_key):
        rows = convergence_sets[algorithm_name]
        iterations = np.array([float(row["iteration"]) for row in rows], dtype=float)
        mean_values = np.array([float(row["mean_best_sum"]) for row in rows], dtype=float)
        min_values = np.array([float(row["min_best_sum"]) for row in rows], dtype=float)
        max_values = np.array([float(row["max_best_sum"]) for row in rows], dtype=float)
        start_index = convergence_main_start_index(iterations, mean_values, max_values)
        series.append(
            {
                "name": algorithm_name,
                "iterations": iterations,
                "mean": mean_values,
                "min": min_values,
                "max": max_values,
                "start_index": start_index,
                "color": ALGORITHM_COLORS.get(algorithm_name),
            }
        )

    main_start_iteration = max(item["iterations"][item["start_index"]] for item in series)
    fig, axis = plt.subplots(figsize=(10, 6))
    main_y_values = []

    for item in series:
        iterations = item["iterations"]
        mask = iterations >= main_start_iteration
        color = item["color"]

        axis.fill_between(
            iterations[mask],
            item["min"][mask],
            item["max"][mask],
            color=color,
            alpha=0.08,
            linewidth=0.0,
            label=f"{item['name']} 范围",
        )
        axis.plot(
            iterations[mask],
            item["mean"][mask],
            linewidth=1.9,
            color=color,
            label=f"{item['name']} 平均值",
        )
        main_y_values.extend(item["mean"][mask])
        main_y_values.extend(item["min"][mask])

    main_y_values = np.array(main_y_values, dtype=float)
    main_low = float(np.min(main_y_values))
    main_high = float(np.max(main_y_values))
    margin = max((main_high - main_low) * 0.08, 1.0)

    axis.set_xlim(main_start_iteration, max(float(rows[-1]["iteration"]) for rows in convergence_sets.values()))
    axis.set_ylim(main_low - margin, main_high + margin)
    axis.set_xlabel("迭代次数 / 次")
    axis.set_ylabel("最优目标和")
    set_common_style(axis)
    axis.legend(loc="upper right", ncol=2, fontsize=LEGEND_FONT_SIZE)

    inset = axis.inset_axes([0.30, 0.16, 0.50, 0.42])
    for item in series:
        iterations = item["iterations"]
        mask = iterations <= main_start_iteration
        color = item["color"]
        inset.fill_between(
            iterations[mask],
            item["min"][mask],
            item["max"][mask],
            color=color,
            alpha=0.08,
            linewidth=0.0,
        )
        inset.plot(iterations[mask], item["mean"][mask], color=color, linewidth=1.2)
    inset.set_yscale("log")
    inset.set_xlabel("前期迭代", fontsize=9)
    inset.set_ylabel("目标和", fontsize=9)
    inset.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    inset.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def has_result_set(input_dir):
    return (
        (input_dir / "compromise_power_curves.csv").exists()
        and (input_dir / "compromise_soc_curve.csv").exists()
        and (input_dir / "multi_objective_average_convergence_history.csv").exists()
    )


def plot_result_set(input_dir, output_dir, algorithm_name=None):
    output_dir.mkdir(parents=True, exist_ok=True)

    power_rows = read_csv_rows(input_dir / "compromise_power_curves.csv")

    plot_power_dispatch(power_rows, output_dir / "compromise_power_dispatch.png", algorithm_name=algorithm_name)
    plot_cost_breakdown(power_rows, output_dir / "compromise_cost_breakdown.png", algorithm_name=algorithm_name)


def clear_existing_png_files(output_dir):
    if not output_dir.exists():
        return
    for path in output_dir.rglob("*.png"):
        path.unlink()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clear_existing_png_files(OUTPUT_DIR)

    plotted_dirs = []
    metric_sets = {}
    convergence_sets = {}
    pareto_sets = {}
    reference_rows = []

    algorithm_dirs = [
        path
        for path in sorted(RESULTS_DIR.iterdir())
        if path.is_dir() and has_result_set(path)
    ]
    algorithm_dirs = sorted(algorithm_dirs, key=lambda path: algorithm_sort_key(read_algorithm_name(path)))

    for input_dir in algorithm_dirs:
        algorithm_name = read_algorithm_name(input_dir)
        if has_result_set(input_dir):
            output_dir = OUTPUT_DIR / input_dir.name
            plot_result_set(input_dir, output_dir, algorithm_name=algorithm_name)
            plotted_dirs.append(input_dir)
        metrics_path = input_dir / "multi_objective_metrics.csv"
        if metrics_path.exists():
            metric_sets[algorithm_name] = read_dict_rows(metrics_path)
        convergence_path = input_dir / "multi_objective_average_convergence_history.csv"
        if convergence_path.exists():
            convergence_sets[algorithm_name] = read_dict_rows(convergence_path)
        pareto_path = input_dir / "multi_objective_pareto.csv"
        if pareto_path.exists():
            pareto_sets[algorithm_name] = read_dict_rows(pareto_path)
        reference_path = input_dir / "multi_objective_reference_pareto.csv"
        if not reference_rows and reference_path.exists():
            reference_rows = read_dict_rows(reference_path)

    if len(metric_sets) >= 2:
        plot_igd_igd_plus_comparison(metric_sets, OUTPUT_DIR / "algorithm_igd_igd_plus_comparison.png")
    if len(pareto_sets) >= 2:
        plot_algorithm_pareto_comparison(pareto_sets, reference_rows, OUTPUT_DIR / "algorithm_pareto_comparison.png")
    if len(convergence_sets) >= 2:
        plot_algorithm_convergence_comparison(
            convergence_sets,
            OUTPUT_DIR / "algorithm_average_convergence_comparison.png",
        )

    if not plotted_dirs:
        raise FileNotFoundError(f"No complete result CSV set found in {RESULTS_DIR}.")

    print("=" * 80)
    print("微电网结果图已保存")
    print(f"输入结果目录: {RESULTS_DIR}")
    print(f"输出图片目录: {OUTPUT_DIR}")
    print("已绘图目录:")
    for plotted_dir in plotted_dirs:
        print(f"- {plotted_dir}")


if __name__ == "__main__":
    main()
