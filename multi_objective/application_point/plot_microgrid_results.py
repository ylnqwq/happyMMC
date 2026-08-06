# -*- coding: utf-8 -*-
"""Plot figures from saved microgrid dispatch result CSV files."""

from __future__ import annotations

import csv
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
LEGEND_FONT_SIZE = 18


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as file:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(file)]


def set_common_style(axis):
    axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_power_dispatch(rows, path):
    hours = np.array([row["hour"] for row in rows], dtype=float)
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
        axis.bar(hours, values, bottom=bottom, width=0.72, label=label, color=color, edgecolor="white", linewidth=0.35)
        bottom += values

    bottom = np.zeros_like(hours)
    for label, values, color in negative_series:
        axis.bar(hours, values, bottom=bottom, width=0.72, label=label, color=color, edgecolor="white", linewidth=0.35)
        bottom += values

    axis.plot(hours, load, color="#222222", marker="o", markersize=4, linewidth=1.8, label="负荷")
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set_xlim(0.3, 24.7)
    axis.set_xticks(np.arange(1, 25, 1))
    axis.set_xlabel("时段 / h")
    axis.set_ylabel("功率 / kW")
    axis.set_title("折中方案下各能源出力曲线")
    set_common_style(axis)
    axis.legend(
        ncol=4,
        fontsize=LEGEND_FONT_SIZE,
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_soc_curve(rows, path):
    hours = np.arange(0, len(rows) + 1, dtype=float)
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
    axis.set_title("储能荷电状态曲线")
    set_common_style(axis)
    axis.legend(
        fontsize=LEGEND_FONT_SIZE,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_average_convergence(rows, path):
    iterations = np.array([row["iteration"] for row in rows], dtype=float)
    mean_values = np.array([row["mean_best_sum"] for row in rows], dtype=float)
    min_values = np.array([row["min_best_sum"] for row in rows], dtype=float)
    max_values = np.array([row["max_best_sum"] for row in rows], dtype=float)

    fig, axis = plt.subplots(figsize=(10, 6))
    axis.fill_between(iterations, min_values, max_values, color="#b8c7d9", alpha=0.35, label="最小-最大范围")
    axis.plot(iterations, mean_values, color="#2f4b7c", linewidth=1.9, label="平均最优目标和")
    axis.set_xlabel("迭代次数 / 次")
    axis.set_ylabel("最优目标和")
    axis.set_title("MOIABC 平均收敛曲线")
    set_common_style(axis)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=LEGEND_FONT_SIZE,
    )

    inset = axis.inset_axes([0.58, 0.38, 0.34, 0.35])
    tail_mask = iterations >= max(0, iterations[-1] - 100)
    inset.plot(iterations[tail_mask], mean_values[tail_mask], color="#2f4b7c", linewidth=1.2)
    inset.fill_between(iterations[tail_mask], min_values[tail_mask], max_values[tail_mask], color="#b8c7d9", alpha=0.35)
    inset.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    inset.set_title("后 100 次迭代", fontsize=10)
    inset.tick_params(labelsize=8)

    fig.tight_layout(rect=[0, 0.13, 1, 1])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_cost_breakdown(rows, path):
    diesel_cost = sum(row["diesel_kw"] for row in rows) * PARAMS.diesel_unit_cost
    battery_cost = sum(abs(row["battery_kw"]) for row in rows) * PARAMS.battery_om_cost
    buy_cost = sum(row["grid_buy_kw"] * BUY_PRICE[index] for index, row in enumerate(rows))
    sell_revenue = sum(row["grid_sell_kw"] * SELL_PRICE[index] for index, row in enumerate(rows))
    net_cost = diesel_cost + battery_cost + buy_cost - sell_revenue

    labels = ["微型燃气轮机", "储能", "主网购电", "售电收益", "净成本"]
    values = [diesel_cost, battery_cost, buy_cost, -sell_revenue, net_cost]
    colors = ["#8f6f4e", "#b07aa1", "#59a14f", "#76b7b2", "#e15759"]

    fig, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set_ylabel("成本 / 元")
    axis.set_title("折中方案经济成本构成")
    set_common_style(axis)
    for bar, value in zip(bars, values):
        va = "bottom" if value >= 0 else "top"
        offset = 3 if value >= 0 else -3
        axis.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width() / 2, value), xytext=(0, offset),
                      textcoords="offset points", ha="center", va=va, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    power_rows = read_csv_rows(RESULTS_DIR / "compromise_power_curves.csv")
    soc_rows = read_csv_rows(RESULTS_DIR / "compromise_soc_curve.csv")
    convergence_rows = read_csv_rows(RESULTS_DIR / "multi_objective_average_convergence_history.csv")

    plot_power_dispatch(power_rows, OUTPUT_DIR / "compromise_power_dispatch.png")
    plot_soc_curve(soc_rows, OUTPUT_DIR / "compromise_soc_curve.png")
    plot_average_convergence(convergence_rows, OUTPUT_DIR / "multi_objective_average_convergence.png")
    plot_cost_breakdown(power_rows, OUTPUT_DIR / "compromise_cost_breakdown.png")

    print("=" * 80)
    print("微电网结果图已保存")
    print(f"输入结果目录: {RESULTS_DIR}")
    print(f"输出图片目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
