# -*- coding: utf-8 -*-

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "moiabc_sensitivity_results"

METRIC_SPECS = {
    "hypervolume": {
        "mean": "mean_hypervolume",
        "std": "std_hypervolume",
        "higher_is_better": True,
        "label": "Hypervolume",
        "suffix": "hv",
    },
    "spacing": {
        "mean": "mean_spacing",
        "std": "std_spacing",
        "higher_is_better": False,
        "label": "Spacing",
        "suffix": "spacing",
    },
    "best_sum": {
        "mean": "mean_best_sum",
        "std": "std_best_sum",
        "higher_is_better": False,
        "label": "Best-sum",
        "suffix": "best_sum",
    },
    "mean_error": {
        "mean": "mean_error",
        "std": "std_error",
        "higher_is_better": False,
        "label": "Mean error",
        "suffix": "mean_error",
    },
}

THREE_OBJECTIVE_BENCHMARK_IDS = {
    "MMF14",
    "MMF15",
    "MMF14_A",
    "MMF15_A",
    "MMF15_L",
    "MMF15_A_L",
    "MMF16_L1",
    "MMF16_L2",
    "MMF16_L3",
}


def configure_fonts():
    font_paths = [
        Path("C:/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/timesbd.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    for font_path in font_paths:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["font.serif"] = ["Times New Roman", "SimSun", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw paper-style MOIABC parameter sensitivity tables."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing moiabc_sensitivity_*.csv files.",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_SPECS),
        default=None,
        help="Metric used in each table cell. Default: draw all available metrics.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top parameter combinations included in the table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: same as input-dir.",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Do not draw the table title.",
    )
    parser.add_argument(
        "--no-note",
        action="store_true",
        help="Do not draw the table note.",
    )
    parser.add_argument(
        "--bar",
        action="store_true",
        help="Draw one top-k average-rank bar chart instead of a table.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["average_rank", "best_count"],
        default="average_rank",
        help="Ranking key for top-k parameter combinations.",
    )
    return parser.parse_args()


def find_csv(input_dir, names):
    for name in names:
        path = input_dir / name
        if path.exists():
            return path
    joined = ", ".join(names)
    raise FileNotFoundError(f"Cannot find any of: {joined} in {input_dir}")


def read_csv_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def to_float(value):
    if value is None or value == "":
        return float("nan")
    return float(value)


def normalize_rows(rows):
    normalized = []
    for row in rows:
        item = dict(row)
        item["elite_rate"] = to_float(item["elite_rate"])
        item["elimination_rate"] = to_float(item["elimination_rate"])
        for key, value in row.items():
            if key in {"benchmark_id", "function"}:
                continue
            if key not in {"elite_rate", "elimination_rate"}:
                try:
                    item[key] = to_float(value)
                except (TypeError, ValueError):
                    item[key] = value
        normalized.append(item)
    return normalized


def keep_two_objective_rows(rows):
    return [row for row in rows if row.get("benchmark_id") not in THREE_OBJECTIVE_BENCHMARK_IDS]


def available_metrics(summary_rows):
    if not summary_rows:
        return []
    keys = set(summary_rows[0])
    return [
        metric
        for metric, spec in METRIC_SPECS.items()
        if spec["mean"] in keys and spec["std"] in keys
    ]


def benchmark_sort_key(benchmark_id):
    zdt_match = re.fullmatch(r"ZDT(\d+)", benchmark_id)
    if zdt_match:
        return 0, int(zdt_match.group(1)), "", ""

    mmf_match = re.fullmatch(r"MMF(\d+)(.*)", benchmark_id)
    if mmf_match:
        return 1, int(mmf_match.group(1)), mmf_match.group(2), ""

    cec_match = re.search(r"CEC(\d+)_F(\d+)", benchmark_id)
    if cec_match:
        return 2, int(cec_match.group(1)), int(cec_match.group(2)), ""

    function_match = re.search(r"F(\d+)", benchmark_id)
    if function_match:
        return 3, int(function_match.group(1)), "", benchmark_id

    return 9, 0, "", benchmark_id


def function_label(benchmark_id):
    cec_match = re.search(r"CEC\d+_F(\d+)", benchmark_id)
    if cec_match:
        return f"F{int(cec_match.group(1)):02d}"
    return benchmark_id


def format_rate(value):
    return f"{value:.2f}".rstrip("0").rstrip(".")


def format_scientific(value):
    if not np.isfinite(value):
        return "nan"
    return f"{value:.2E}".replace("E+0", "E+").replace("E-0", "E-")


def relative_to_best(value, best_value, higher_is_better):
    if best_value is None or not np.isfinite(value) or not np.isfinite(best_value):
        return float("nan")
    relative_value = best_value - value if higher_is_better else value - best_value
    if math.isclose(relative_value, 0.0, rel_tol=1e-12, abs_tol=1e-12):
        return 0.0
    return max(0.0, relative_value)


def rank_key(row):
    return (
        row.get("average_rank", float("inf")),
        -row.get("best_count", 0.0),
        row["elite_rate"],
        row["elimination_rate"],
    )


def select_combinations(summary_rows, rank_rows, top_k):
    if rank_rows:
        ranked = sorted(rank_rows, key=rank_key)
        return [(row["elite_rate"], row["elimination_rate"]) for row in ranked[:top_k]]

    seen = sorted(
        {(row["elite_rate"], row["elimination_rate"]) for row in summary_rows},
        key=lambda item: (item[0], item[1]),
    )
    return seen[:top_k]


def write_table_csv(summary_rows, benchmark_ids, combinations, metric, output_path):
    spec = METRIC_SPECS[metric]
    index = {
        (row["benchmark_id"], row["elite_rate"], row["elimination_rate"]): row
        for row in summary_rows
    }
    headers = [f"e={format_rate(e)},d={format_rate(d)}" for e, d in combinations]
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["benchmark_id", "function"] + headers)
        writer.writeheader()
        for benchmark_id in benchmark_ids:
            output_row = {"benchmark_id": benchmark_id, "function": function_label(benchmark_id)}
            best_value = get_row_best_value(
                index,
                benchmark_id,
                combinations,
                spec["mean"],
                spec["higher_is_better"],
            )
            for elite_rate, elimination_rate in combinations:
                header = f"e={format_rate(elite_rate)},d={format_rate(elimination_rate)}"
                row = index.get((benchmark_id, elite_rate, elimination_rate))
                if row is None:
                    output_row[header] = ""
                else:
                    mean = row[spec["mean"]]
                    std = row[spec["std"]]
                    relative_mean = relative_to_best(mean, best_value, spec["higher_is_better"])
                    output_row[header] = f"{format_scientific(relative_mean)}±{format_scientific(std)}"
            writer.writerow(output_row)


def draw_table(summary_rows, benchmark_ids, combinations, metric, output_path, show_title=True, show_note=True):
    spec = METRIC_SPECS[metric]
    index = {
        (row["benchmark_id"], row["elite_rate"], row["elimination_rate"]): row
        for row in summary_rows
    }

    column_headers = ["函数"] + [f"e={format_rate(e)}\nd={format_rate(d)}" for e, d in combinations]
    column_widths = [1.25] + [2.15] * len(combinations)
    total_width = sum(column_widths)
    benchmark_count = len(benchmark_ids)
    title_space = 0.62 if show_title else 0.05
    note_space = 0.48 if show_note else 0.08
    fig_width = max(10.5, total_width * 0.82)
    fig_height = max(4.8, 1.2 + title_space + note_space + benchmark_count * 0.34)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    ax.set_axis_off()
    ax.set_xlim(0, total_width)
    ax.set_ylim(-note_space, benchmark_count + 1.65 + title_space)

    font_family = ["Times New Roman", "SimSun"]
    x_positions = np.cumsum([0] + column_widths)

    if show_title:
        title = f"MOIABC 参数敏感性分析结果（{spec['label']}）"
        ax.text(
            total_width / 2,
            benchmark_count + 1.58,
            title,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            family=font_family,
        )

    top_y = benchmark_count + 1.22
    header_y = benchmark_count + 0.72
    bottom_y = -0.08
    ax.hlines(
        [top_y, header_y - 0.28, bottom_y],
        0,
        total_width,
        colors="black",
        linewidths=[1.35, 0.75, 1.35],
    )

    for column_index, header in enumerate(column_headers):
        x = (x_positions[column_index] + x_positions[column_index + 1]) / 2
        ax.text(
            x,
            header_y,
            header,
            ha="center",
            va="center",
            fontsize=10.3,
            family=font_family,
            linespacing=0.95,
        )

    mean_key = spec["mean"]
    std_key = spec["std"]
    higher_is_better = spec["higher_is_better"]
    for row_index, benchmark_id in enumerate(benchmark_ids):
        y = benchmark_count - row_index + 0.2
        ax.text(
            (x_positions[0] + x_positions[1]) / 2,
            y,
            function_label(benchmark_id),
            ha="center",
            va="center",
            fontsize=9.6,
            family=font_family,
        )

        values = [
            index[(benchmark_id, elite_rate, elimination_rate)][mean_key]
            for elite_rate, elimination_rate in combinations
            if (benchmark_id, elite_rate, elimination_rate) in index
            and np.isfinite(index[(benchmark_id, elite_rate, elimination_rate)][mean_key])
        ]
        if values:
            best_value = max(values) if higher_is_better else min(values)
        else:
            best_value = None

        for column_index, (elite_rate, elimination_rate) in enumerate(combinations, start=1):
            row = index.get((benchmark_id, elite_rate, elimination_rate))
            if row is None or not np.isfinite(row[mean_key]):
                text = "-"
                fontweight = "normal"
            else:
                mean = row[mean_key]
                std = row[std_key]
                relative_mean = relative_to_best(mean, best_value, higher_is_better)
                text = f"{format_scientific(relative_mean)}±{format_scientific(std)}"
                fontweight = "bold" if is_best_cell(mean, best_value) else "normal"
            x = (x_positions[column_index] + x_positions[column_index + 1]) / 2
            ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                fontsize=8.7,
                fontweight=fontweight,
                family=font_family,
            )

    if show_note:
        direction = "越大越优" if higher_is_better else "越小越优"
        note = (
            "注：e 表示 elite_rate，d 表示 elimination_rate；"
            f"数值为相对本行最优均值的差值±标准差（最优为 0，原指标{direction}）；"
            "每行加粗表示该测试函数上的最优参数组合。"
        )
        ax.text(0, -0.42, note, ha="left", va="center", fontsize=8.2, family=font_family)

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def combination_header(elite_rate, elimination_rate):
    return f"e={format_rate(elite_rate)}, d={format_rate(elimination_rate)}"


def get_row_best_value(index, benchmark_id, combinations, mean_key, higher_is_better):
    values = [
        index[(benchmark_id, elite_rate, elimination_rate)][mean_key]
        for elite_rate, elimination_rate in combinations
        if (benchmark_id, elite_rate, elimination_rate) in index
        and np.isfinite(index[(benchmark_id, elite_rate, elimination_rate)][mean_key])
    ]
    if not values:
        return None
    return max(values) if higher_is_better else min(values)


def is_best_cell(value, best_value):
    return best_value is not None and math.isclose(
        value,
        best_value,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def count_best_cells(index, benchmark_ids, combinations, mean_key, higher_is_better):
    counts = [0] * len(combinations)
    for benchmark_id in benchmark_ids:
        best_value = get_row_best_value(index, benchmark_id, combinations, mean_key, higher_is_better)
        for column_index, (elite_rate, elimination_rate) in enumerate(combinations):
            row = index.get((benchmark_id, elite_rate, elimination_rate))
            if row is None or not np.isfinite(row[mean_key]):
                continue
            if is_best_cell(row[mean_key], best_value):
                counts[column_index] += 1
    return counts


def sort_combinations_by_table_count(summary_rows, benchmark_ids, combinations, metric):
    spec = METRIC_SPECS[metric]
    index = {
        (row["benchmark_id"], row["elite_rate"], row["elimination_rate"]): row
        for row in summary_rows
    }
    counts = count_best_cells(
        index,
        benchmark_ids,
        combinations,
        spec["mean"],
        spec["higher_is_better"],
    )
    paired = list(zip(combinations, counts))
    paired.sort(key=lambda item: (-item[1], item[0][0], item[0][1]))
    return [item[0] for item in paired], [item[1] for item in paired]


def rank_combinations_for_metric(summary_rows, metric, sort_by="average_rank"):
    spec = METRIC_SPECS[metric]
    mean_key = spec["mean"]
    higher_is_better = spec["higher_is_better"]
    by_benchmark = {}
    for row in summary_rows:
        if np.isfinite(row.get(mean_key, float("nan"))):
            by_benchmark.setdefault(row["benchmark_id"], []).append(row)

    rank_sums = {}
    best_counts = {}
    metric_sums = {}
    for rows in by_benchmark.values():
        ordered = sorted(rows, key=lambda item: item[mean_key], reverse=higher_is_better)
        for rank, row in enumerate(ordered, start=1):
            key = (row["elite_rate"], row["elimination_rate"])
            rank_sums[key] = rank_sums.get(key, 0.0) + rank
            metric_sums[key] = metric_sums.get(key, 0.0) + row[mean_key]
            best_counts.setdefault(key, 0)
        if ordered:
            best = ordered[0]
            best_key = (best["elite_rate"], best["elimination_rate"])
            best_counts[best_key] = best_counts.get(best_key, 0) + 1

    benchmark_count = len(by_benchmark)
    rows = []
    for elite_rate, elimination_rate in sorted(rank_sums):
        key = (elite_rate, elimination_rate)
        rows.append(
            {
                "elite_rate": elite_rate,
                "elimination_rate": elimination_rate,
                "average_rank": rank_sums[key] / benchmark_count,
                "best_count": best_counts.get(key, 0),
                "benchmark_count": benchmark_count,
                "average_metric": metric_sums[key] / benchmark_count,
            }
        )
    if sort_by == "best_count":
        return sorted(rows, key=lambda item: (-item["best_count"], item["average_rank"], item["average_metric"]))
    return sorted(rows, key=lambda item: (item["average_rank"], -item["best_count"], item["average_metric"]))


def rank_combinations_across_metrics(summary_rows, metrics):
    rank_sums = {}
    best_counts = {}
    benchmark_metric_count = 0

    for metric in metrics:
        spec = METRIC_SPECS[metric]
        mean_key = spec["mean"]
        higher_is_better = spec["higher_is_better"]
        by_benchmark = {}
        for row in summary_rows:
            if np.isfinite(row.get(mean_key, float("nan"))):
                by_benchmark.setdefault(row["benchmark_id"], []).append(row)

        for rows in by_benchmark.values():
            benchmark_metric_count += 1
            ordered = sorted(rows, key=lambda item: item[mean_key], reverse=higher_is_better)
            for rank, row in enumerate(ordered, start=1):
                key = (row["elite_rate"], row["elimination_rate"])
                rank_sums[key] = rank_sums.get(key, 0.0) + rank
                best_counts.setdefault(key, 0)
            if ordered:
                best_key = (ordered[0]["elite_rate"], ordered[0]["elimination_rate"])
                best_counts[best_key] = best_counts.get(best_key, 0) + 1

    if benchmark_metric_count == 0:
        return []
    rows = [
        {
            "elite_rate": elite_rate,
            "elimination_rate": elimination_rate,
            "average_rank": rank_sum / benchmark_metric_count,
            "best_count": best_counts.get((elite_rate, elimination_rate), 0),
        }
        for (elite_rate, elimination_rate), rank_sum in rank_sums.items()
    ]
    return sorted(rows, key=lambda item: (item["average_rank"], -item["best_count"]))


def draw_top_bar_chart(summary_rows, metric, top_k, output_path, show_title=True, sort_by="average_rank"):
    spec = METRIC_SPECS[metric]
    top_rows = rank_combinations_for_metric(summary_rows, metric)[:top_k]
    if not top_rows:
        raise ValueError(f"No rows available for metric {metric!r}.")

    combinations = [(row["elite_rate"], row["elimination_rate"]) for row in top_rows]
    benchmark_ids = sorted({row["benchmark_id"] for row in summary_rows}, key=benchmark_sort_key)
    if sort_by == "best_count":
        combinations, values = sort_combinations_by_table_count(
            summary_rows,
            benchmark_ids,
            combinations,
            metric,
        )
    else:
        values = [row["average_rank"] for row in top_rows]

    labels = [f"e={elite_rate:.2f}\nd={elimination_rate:.2f}" for elite_rate, elimination_rate in combinations]
    colors = ["#f1b183", "#7fa6d9", "#a7dce0", "#4e8fc7", "#b7cee8"]

    font_family = "Microsoft YaHei"
    fig, ax = plt.subplots(figsize=(5.2, 3.7), dpi=300)
    bars = ax.bar(labels, values, color=colors[: len(top_rows)], edgecolor="#606060", linewidth=0.7)

    y_label = "最优次数" if sort_by == "best_count" else "平均排名"
    ax.set_ylabel(y_label, fontsize=10.5, family=font_family)
    ax.set_xlabel("参数组合", fontsize=10.5, family=font_family)
    if show_title:
        ax.set_title(
            f"MOIABC 参数敏感性 Top {len(top_rows)}（{spec['label']}）",
            fontsize=11.5,
            fontweight="bold",
            family=font_family,
        )

    y_limit = max(values) * 1.18 if max(values) > 0 else 1.0
    ax.set_ylim(0, y_limit)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        [bars[0]],
        [f"{spec['label']} 最优次数（越大越优）" if sort_by == "best_count" else f"{spec['label']} 平均排名（越小越优）"],
        loc="upper left",
        frameon=False,
        fontsize=9.0,
        prop={"family": font_family},
    )

    ax.tick_params(axis="x", labelsize=9.2)
    ax.tick_params(axis="y", labelsize=9.2)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_limit * 0.025,
            f"{int(value)}" if sort_by == "best_count" else f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.2,
            family=font_family,
        )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_table_csv_paper(summary_rows, benchmark_ids, combinations, metric, output_path):
    spec = METRIC_SPECS[metric]
    mean_key = spec["mean"]
    std_key = spec["std"]
    index = {
        (row["benchmark_id"], row["elite_rate"], row["elimination_rate"]): row
        for row in summary_rows
    }
    headers = [combination_header(e, d) for e, d in combinations]
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["benchmark_id", "function"] + headers)
        writer.writeheader()
        for benchmark_id in benchmark_ids:
            output_row = {"benchmark_id": benchmark_id, "function": function_label(benchmark_id)}
            best_value = get_row_best_value(
                index,
                benchmark_id,
                combinations,
                mean_key,
                spec["higher_is_better"],
            )
            for elite_rate, elimination_rate in combinations:
                header = combination_header(elite_rate, elimination_rate)
                row = index.get((benchmark_id, elite_rate, elimination_rate))
                if row is None:
                    output_row[header] = ""
                else:
                    relative_mean = relative_to_best(row[mean_key], best_value, spec["higher_is_better"])
                    output_row[header] = (
                        f"{format_scientific(relative_mean)}±{format_scientific(row[std_key])}"
                    )
            writer.writerow(output_row)

        count_row = {"benchmark_id": "count", "function": "个数"}
        counts = count_best_cells(index, benchmark_ids, combinations, mean_key, spec["higher_is_better"])
        for header, count in zip(headers, counts):
            count_row[header] = count
        writer.writerow(count_row)


def draw_table_paper(summary_rows, benchmark_ids, combinations, metric, output_path, show_title=True, show_note=True):
    spec = METRIC_SPECS[metric]
    index = {
        (row["benchmark_id"], row["elite_rate"], row["elimination_rate"]): row
        for row in summary_rows
    }
    mean_key = spec["mean"]
    std_key = spec["std"]
    higher_is_better = spec["higher_is_better"]

    column_headers = ["函数"] + [combination_header(e, d) for e, d in combinations]
    column_widths = [1.25] + [2.25] * len(combinations)
    total_width = sum(column_widths)
    benchmark_count = len(benchmark_ids)
    row_count = benchmark_count + 1
    title_space = 0.62 if show_title else 0.05
    note_space = 0.48 if show_note else 0.05
    fig_width = max(10.5, total_width * 0.86)
    fig_height = max(4.8, 1.15 + title_space + note_space + row_count * 0.34)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    ax.set_axis_off()
    ax.set_xlim(0, total_width)
    ax.set_ylim(-note_space, row_count + 1.55 + title_space)

    font_family = ["Times New Roman", "SimSun"]
    x_positions = np.cumsum([0] + column_widths)

    if show_title:
        title = f"表  MOIABC 不同参数组合的敏感度分析结果（{spec['label']}）"
        ax.text(
            total_width / 2,
            row_count + 1.48,
            title,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            family=font_family,
        )

    top_y = row_count + 1.12
    header_y = row_count + 0.65
    bottom_y = -0.02
    ax.hlines(
        [top_y, header_y - 0.28, bottom_y],
        0,
        total_width,
        colors="black",
        linewidths=[1.35, 0.75, 1.35],
    )

    for column_index, header in enumerate(column_headers):
        x = (x_positions[column_index] + x_positions[column_index + 1]) / 2
        ax.text(
            x,
            header_y,
            header,
            ha="center",
            va="center",
            fontsize=10.3,
            family=font_family,
        )

    for row_index, benchmark_id in enumerate(benchmark_ids):
        y = row_count - row_index - 0.05
        ax.text(
            (x_positions[0] + x_positions[1]) / 2,
            y,
            function_label(benchmark_id),
            ha="center",
            va="center",
            fontsize=9.6,
            family=font_family,
        )

        best_value = get_row_best_value(index, benchmark_id, combinations, mean_key, higher_is_better)
        for column_index, (elite_rate, elimination_rate) in enumerate(combinations, start=1):
            row = index.get((benchmark_id, elite_rate, elimination_rate))
            if row is None or not np.isfinite(row[mean_key]):
                text = "-"
                fontweight = "normal"
            else:
                mean = row[mean_key]
                relative_mean = relative_to_best(mean, best_value, higher_is_better)
                text = f"{format_scientific(relative_mean)}±{format_scientific(row[std_key])}"
                fontweight = "bold" if is_best_cell(mean, best_value) else "normal"
            x = (x_positions[column_index] + x_positions[column_index + 1]) / 2
            ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                fontsize=8.7,
                fontweight=fontweight,
                family=font_family,
            )

    count_y = 0.25
    counts = count_best_cells(index, benchmark_ids, combinations, mean_key, higher_is_better)
    ax.text(
        (x_positions[0] + x_positions[1]) / 2,
        count_y,
        "个数",
        ha="center",
        va="center",
        fontsize=9.6,
        family=font_family,
    )
    for column_index, count in enumerate(counts, start=1):
        x = (x_positions[column_index] + x_positions[column_index + 1]) / 2
        ax.text(
            x,
            count_y,
            str(count),
            ha="center",
            va="center",
            fontsize=9.6,
            family=font_family,
        )

    if show_note:
        direction = "越大越优" if higher_is_better else "越小越优"
        note = (
            "注：e 表示 elite_rate，d 表示 elimination_rate；"
            f"数值为相对本行最优均值的差值±标准差（最优为 0，原指标{direction}）；"
            "每行加粗表示该测试函数上的最优参数组合。"
        )
        ax.text(0, -0.42, note, ha="left", va="center", fontsize=8.2, family=font_family)

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main():
    configure_fonts()
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = find_csv(
        input_dir,
        [
            "moiabc_sensitivity_summary_by_function.csv",
            "sensitivity_summary_by_function.csv",
        ],
    )
    summary_rows = keep_two_objective_rows(normalize_rows(read_csv_rows(summary_path)))

    try:
        rank_path = find_csv(
            input_dir,
            [
                "moiabc_sensitivity_average_rank.csv",
                "sensitivity_average_rank.csv",
            ],
        )
        rank_rows = normalize_rows(read_csv_rows(rank_path))
    except FileNotFoundError:
        rank_rows = []

    metrics = available_metrics(summary_rows)
    if args.metric:
        if args.metric not in metrics:
            available = ", ".join(metrics) or "none"
            raise ValueError(f"Metric {args.metric!r} is not available. Available metrics: {available}")
        metrics = [args.metric]

    benchmark_ids = sorted({row["benchmark_id"] for row in summary_rows}, key=benchmark_sort_key)
    if len(metrics) == 1:
        ranked_rows = rank_combinations_for_metric(summary_rows, metrics[0])[: args.top_k]
        combinations = [(row["elite_rate"], row["elimination_rate"]) for row in ranked_rows]
        if args.sort_by == "best_count":
            combinations, _ = sort_combinations_by_table_count(
                summary_rows,
                benchmark_ids,
                combinations,
                metrics[0],
            )
    else:
        ranked_rows = rank_combinations_across_metrics(summary_rows, metrics)[: args.top_k]
        if ranked_rows:
            combinations = [(row["elite_rate"], row["elimination_rate"]) for row in ranked_rows]
        else:
            combinations = select_combinations(summary_rows, rank_rows, args.top_k)

    if args.bar:
        if len(metrics) != 1:
            raise ValueError("Please pass --metric when using --bar so only one figure is generated.")
        metric = metrics[0]
        suffix = METRIC_SPECS[metric]["suffix"]
        output_png = output_dir / f"moiabc_sensitivity_top{args.top_k}_{suffix}_bar.png"
        draw_top_bar_chart(
            summary_rows,
            metric,
            args.top_k,
            output_png,
            show_title=not args.no_title,
            sort_by=args.sort_by,
        )
        print(output_png)
        return

    outputs = []
    for metric in metrics:
        suffix = METRIC_SPECS[metric]["suffix"]
        output_png = output_dir / f"moiabc_sensitivity_top{len(combinations)}_{suffix}_table.png"
        output_csv = output_png.with_suffix(".csv")
        draw_table_paper(
            summary_rows,
            benchmark_ids,
            combinations,
            metric,
            output_png,
            show_title=not args.no_title,
            show_note=not args.no_note,
        )
        write_table_csv_paper(summary_rows, benchmark_ids, combinations, metric, output_csv)
        outputs.extend([output_png, output_csv])

    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
