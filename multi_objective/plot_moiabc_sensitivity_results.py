# -*- coding: utf-8 -*-

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "moiabc_elite_elimination_sensitivity_results"
DEFAULT_ABLATION_INPUT_DIR = Path(__file__).resolve().parent / "moiabc_ablation_results"

ABLATION_ALGORITHM_ORDER = [
    "MOIABC",
    "MOIABC-no-good-point-init",
    "MOIABC-no-tournament-selection",
    "MOIABC-no-elite-enhancement",
    "MOIABC-no-worst-elimination",
    "MOABC-equivalent",
]

ABLATION_ALGORITHM_LABELS = {
    "MOIABC": "MOIABC",
    "MOIABC-no-good-point-init": "MOIABC-NG",
    "MOIABC-no-tournament-selection": "MOIABC-NT",
    "MOIABC-no-elite-enhancement": "MOIABC-NE",
    "MOIABC-no-worst-elimination": "MOIABC-NW",
    "MOABC-equivalent": "MOABC",
}

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

EXCLUDED_DUPLICATE_BENCHMARK_IDS = set()


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
        "--mode",
        choices=["sensitivity", "ablation"],
        default="sensitivity",
        help="Draw parameter sensitivity figures or MOIABC ablation figures.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing MOIABC sensitivity CSV files.",
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
        "--friedman",
        action="store_true",
        help="Draw rank bar chart(s) instead of tables.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["average_rank", "best_count"],
        default="best_count",
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
        for key in sensitivity_parameter_columns([row]):
            item[key] = to_float(item[key])
        for key, value in row.items():
            if key in {"benchmark_id", "function"}:
                continue
            if key not in {"elite_rate", "elimination_rate", "archive_rate"}:
                try:
                    item[key] = to_float(value)
                except (TypeError, ValueError):
                    item[key] = value
        normalized.append(item)
    return normalized


def exclude_duplicate_rows(rows):
    return [row for row in rows if row.get("benchmark_id") not in EXCLUDED_DUPLICATE_BENCHMARK_IDS]


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

    uf_match = re.fullmatch(r"UF(\d+)", benchmark_id)
    if uf_match:
        return 1, int(uf_match.group(1)), "", ""

    mmf_match = re.fullmatch(r"MMF(\d+)(.*)", benchmark_id)
    if mmf_match:
        return 2, int(mmf_match.group(1)), mmf_match.group(2), ""

    cec_match = re.search(r"CEC(\d+)_F(\d+)", benchmark_id)
    if cec_match:
        return 3, int(cec_match.group(1)), int(cec_match.group(2)), ""

    function_match = re.search(r"F(\d+)", benchmark_id)
    if function_match:
        return 4, int(function_match.group(1)), "", benchmark_id

    return 9, 0, "", benchmark_id


def function_label(benchmark_id):
    cec_match = re.search(r"CEC\d+_F(\d+)", benchmark_id)
    if cec_match:
        return f"F{int(cec_match.group(1)):02d}"
    return benchmark_id


def format_rate(value):
    return f"{value:.2f}".rstrip("0").rstrip(".")


def sensitivity_parameter_columns(rows):
    if not rows:
        return ("elite_rate", "elimination_rate")
    keys = set(rows[0])
    if "archive_rate" in keys:
        return ("archive_rate",)
    return ("elite_rate", "elimination_rate")


def parameter_key(row, columns=None):
    columns = columns or sensitivity_parameter_columns([row])
    return tuple(row[column] for column in columns)


def parameter_index_key(benchmark_id, key):
    return (benchmark_id, *key)


def parameter_label(key, multiline=False):
    separator = "\n" if multiline else ", "
    if len(key) == 1:
        return f"ra={format_rate(key[0])}"
    if len(key) == 2:
        return separator.join([f"e={format_rate(key[0])}", f"d={format_rate(key[1])}"])
    return separator.join(format_rate(value) for value in key)


def parameter_row_fields(key, columns):
    return {column: value for column, value in zip(columns, key)}


def parameter_note(columns, higher_is_better):
    direction = "越大越优" if higher_is_better else "越小越优"
    if columns == ("archive_rate",):
        prefix = "注：ra 表示外部档案引导率；"
    else:
        prefix = "注：e 表示 elite_rate，d 表示 elimination_rate；"
    return (
        prefix
        + f"数值为相对本行最优均值的差值±标准差（最优为 0，原指标{direction}）；"
        + "每行加粗表示该测试函数上的最优参数。"
    )


def sensitivity_output_prefix(summary_rows):
    if sensitivity_parameter_columns(summary_rows) == ("archive_rate",):
        return "moiabc_archive_rate_sensitivity"
    return "moiabc_sensitivity"


def sensitivity_title_label(summary_rows):
    if sensitivity_parameter_columns(summary_rows) == ("archive_rate",):
        return "MOIABC ra 敏感性分析"
    return "MOIABC 参数敏感性分析"


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
    columns = sensitivity_parameter_columns([row])
    return (
        row.get("average_rank", float("inf")),
        -row.get("best_count", 0.0),
        *parameter_key(row, columns),
    )


def select_combinations(summary_rows, rank_rows, top_k):
    columns = sensitivity_parameter_columns(summary_rows or rank_rows)
    if rank_rows:
        ranked = sorted(rank_rows, key=rank_key)
        return [parameter_key(row, columns) for row in ranked[:top_k]]

    seen = sorted(
        {parameter_key(row, columns) for row in summary_rows},
        key=lambda item: item,
    )
    return seen[:top_k]


def write_table_csv(summary_rows, benchmark_ids, combinations, metric, output_path):
    spec = METRIC_SPECS[metric]
    columns = sensitivity_parameter_columns(summary_rows)
    index = {
        parameter_index_key(row["benchmark_id"], parameter_key(row, columns)): row
        for row in summary_rows
    }
    headers = [parameter_label(key) for key in combinations]
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
            for key in combinations:
                header = parameter_label(key)
                row = index.get(parameter_index_key(benchmark_id, key))
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
    columns = sensitivity_parameter_columns(summary_rows)
    index = {
        parameter_index_key(row["benchmark_id"], parameter_key(row, columns)): row
        for row in summary_rows
    }

    column_headers = ["函数"] + [parameter_label(key, multiline=True) for key in combinations]
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
        title = f"{sensitivity_title_label(summary_rows)}结果（{spec['label']}）"
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
            index[parameter_index_key(benchmark_id, key)][mean_key]
            for key in combinations
            if parameter_index_key(benchmark_id, key) in index
            and np.isfinite(index[parameter_index_key(benchmark_id, key)][mean_key])
        ]
        if values:
            best_value = max(values) if higher_is_better else min(values)
        else:
            best_value = None

        for column_index, key in enumerate(combinations, start=1):
            row = index.get(parameter_index_key(benchmark_id, key))
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
        note = parameter_note(columns, higher_is_better)
        ax.text(0, -0.42, note, ha="left", va="center", fontsize=8.2, family=font_family)

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def combination_header(*rates):
    if len(rates) == 1 and isinstance(rates[0], tuple):
        rates = rates[0]
    return parameter_label(tuple(rates))


def get_row_best_value(index, benchmark_id, combinations, mean_key, higher_is_better):
    values = [
        index[parameter_index_key(benchmark_id, key)][mean_key]
        for key in combinations
        if parameter_index_key(benchmark_id, key) in index
        and np.isfinite(index[parameter_index_key(benchmark_id, key)][mean_key])
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
        for column_index, key in enumerate(combinations):
            row = index.get(parameter_index_key(benchmark_id, key))
            if row is None or not np.isfinite(row[mean_key]):
                continue
            if is_best_cell(row[mean_key], best_value):
                counts[column_index] += 1
    return counts


def sort_combinations_by_table_count(summary_rows, benchmark_ids, combinations, metric):
    spec = METRIC_SPECS[metric]
    columns = sensitivity_parameter_columns(summary_rows)
    index = {
        parameter_index_key(row["benchmark_id"], parameter_key(row, columns)): row
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
    paired.sort(key=lambda item: (-item[1], item[0]))
    return [item[0] for item in paired], [item[1] for item in paired]


def rank_combinations_for_metric(summary_rows, metric, sort_by="average_rank"):
    spec = METRIC_SPECS[metric]
    mean_key = spec["mean"]
    higher_is_better = spec["higher_is_better"]
    columns = sensitivity_parameter_columns(summary_rows)
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
            key = parameter_key(row, columns)
            rank_sums[key] = rank_sums.get(key, 0.0) + rank
            metric_sums[key] = metric_sums.get(key, 0.0) + row[mean_key]
            best_counts.setdefault(key, 0)
        if ordered:
            best = ordered[0]
            best_key = parameter_key(best, columns)
            best_counts[best_key] = best_counts.get(best_key, 0) + 1

    benchmark_count = len(by_benchmark)
    rows = []
    for key in sorted(rank_sums):
        rows.append(
            {
                **parameter_row_fields(key, columns),
                "average_rank": rank_sums[key] / benchmark_count,
                "best_count": best_counts.get(key, 0),
                "benchmark_count": benchmark_count,
                "average_metric": metric_sums[key] / benchmark_count,
            }
        )
    if sort_by == "best_count":
        return sorted(rows, key=lambda item: (-item["best_count"], item["average_rank"], item["average_metric"]))
    return sorted(rows, key=lambda item: (item["average_rank"], -item["best_count"], item["average_metric"]))


def rank_combinations_across_metrics(summary_rows, metrics, sort_by="average_rank"):
    columns = sensitivity_parameter_columns(summary_rows)
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
                key = parameter_key(row, columns)
                rank_sums[key] = rank_sums.get(key, 0.0) + rank
                best_counts.setdefault(key, 0)
            if ordered:
                best_key = parameter_key(ordered[0], columns)
                best_counts[best_key] = best_counts.get(best_key, 0) + 1

    if benchmark_metric_count == 0:
        return []
    rows = [
        {
            **parameter_row_fields(key, columns),
            "average_rank": rank_sum / benchmark_metric_count,
            "best_count": best_counts.get(key, 0),
        }
        for key, rank_sum in rank_sums.items()
    ]
    if sort_by == "best_count":
        return sorted(rows, key=lambda item: (-item["best_count"], item["average_rank"]))
    return sorted(rows, key=lambda item: (item["average_rank"], -item["best_count"]))


def draw_top_bar_chart(summary_rows, metric, top_k, output_path, show_title=True, sort_by="average_rank"):
    spec = METRIC_SPECS[metric]
    columns = sensitivity_parameter_columns(summary_rows)
    top_rows = rank_combinations_for_metric(summary_rows, metric, sort_by=sort_by)[:top_k]
    if not top_rows:
        raise ValueError(f"No rows available for metric {metric!r}.")

    combinations = [parameter_key(row, columns) for row in top_rows]
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

    labels = [parameter_label(key, multiline=True) for key in combinations]
    colors = ["#f1b183", "#7fa6d9", "#a7dce0", "#4e8fc7", "#b7cee8"]

    font_family = "Microsoft YaHei"
    fig, ax = plt.subplots(figsize=(5.2, 3.7), dpi=300)
    bars = ax.bar(labels, values, color=colors[: len(top_rows)], edgecolor="#606060", linewidth=0.7)

    if sort_by == "best_count":
        y_label = "最优次数"
    else:
        y_label = "平均排名"
    ax.set_ylabel(y_label, fontsize=10.5, family=font_family)
    ax.set_xlabel("参数组合", fontsize=10.5, family=font_family)
    if show_title:
        ax.set_title(
            f"{sensitivity_title_label(summary_rows)} Top {len(top_rows)}（{spec['label']}）",
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


def normalize_ablation_rows(rows):
    normalized = []
    numeric_keys = {
        "run_times",
        "mean_archive_size",
        "mean_best_sum",
        "std_best_sum",
        "best_sum",
        "mean_sum",
        "mean_spacing",
        "std_spacing",
        "mean_hypervolume",
        "std_hypervolume",
        "best_hypervolume",
        "mean_time",
    }
    for row in rows:
        item = dict(row)
        for key in numeric_keys:
            if key in item:
                item[key] = to_float(item[key])
        normalized.append(item)
    return normalized


def ablation_algorithms(rows):
    present = {row["variant"] for row in rows}
    ordered = [algorithm for algorithm in ABLATION_ALGORITHM_ORDER if algorithm in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def ablation_index(rows):
    return {(row["benchmark_id"], row["variant"]): row for row in rows}


def ablation_best_value(index, benchmark_id, algorithms, mean_key, higher_is_better):
    values = [
        index[(benchmark_id, algorithm)][mean_key]
        for algorithm in algorithms
        if (benchmark_id, algorithm) in index
        and np.isfinite(index[(benchmark_id, algorithm)][mean_key])
    ]
    if not values:
        return None
    return max(values) if higher_is_better else min(values)


def draw_ablation_table(
    summary_rows,
    benchmark_ids,
    algorithms,
    metric,
    output_path,
    show_title=True,
    title_prefix="MOIABC 消融实验结果",
    show_note=False,
    cell_suffixes=None,
    note_suffix="",
):
    spec = METRIC_SPECS[metric]
    mean_key = spec["mean"]
    std_key = spec["std"]
    higher_is_better = spec["higher_is_better"]
    index = ablation_index(summary_rows)

    column_headers = ["函数"] + [ABLATION_ALGORITHM_LABELS.get(algorithm, algorithm) for algorithm in algorithms]
    column_widths = [1.1] + [1.82] * len(algorithms)
    total_width = sum(column_widths)
    benchmark_count = len(benchmark_ids)
    row_count = benchmark_count + 1
    title_space = 0.62 if show_title else 0.05
    note_space = 0.42 if show_note else 0.0
    fig_width = max(12.0, total_width * 0.82)
    fig_height = max(7.0, 1.05 + title_space + note_space + row_count * 0.32)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    ax.set_axis_off()
    ax.set_xlim(0, total_width)
    ax.set_ylim(-note_space, row_count + 1.55 + title_space)

    font_family = ["Times New Roman", "SimSun"]
    x_positions = np.cumsum([0] + column_widths)

    if show_title:
        ax.text(
            total_width / 2,
            row_count + 1.46,
            f"{title_prefix}（{spec['label']}）",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            family=font_family,
        )

    top_y = row_count + 1.08
    header_y = row_count + 0.62
    bottom_y = 0.02
    ax.hlines(
        [top_y, header_y - 0.27, bottom_y],
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
            fontsize=9.2,
            family=font_family,
        )

    best_counts = [0] * len(algorithms)
    for row_index, benchmark_id in enumerate(benchmark_ids):
        y = row_count - row_index - 0.05
        ax.text(
            (x_positions[0] + x_positions[1]) / 2,
            y,
            function_label(benchmark_id),
            ha="center",
            va="center",
            fontsize=8.7,
            family=font_family,
        )

        best_value = ablation_best_value(index, benchmark_id, algorithms, mean_key, higher_is_better)
        for column_index, algorithm in enumerate(algorithms, start=1):
            row = index.get((benchmark_id, algorithm))
            if row is None or not np.isfinite(row[mean_key]):
                text = "-"
                fontweight = "normal"
            else:
                mean = row[mean_key]
                std = row[std_key]
                relative_mean = relative_to_best(mean, best_value, higher_is_better)
                suffix = (cell_suffixes or {}).get((benchmark_id, algorithm, metric), "")
                text = f"{format_scientific(relative_mean)}±{format_scientific(std)}{suffix}"
                is_best = is_best_cell(mean, best_value)
                fontweight = "bold" if is_best else "normal"
                if is_best:
                    best_counts[column_index - 1] += 1
            x = (x_positions[column_index] + x_positions[column_index + 1]) / 2
            ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                fontsize=7.0,
                fontweight=fontweight,
                family=font_family,
            )

    count_y = 0.27
    ax.text(
        (x_positions[0] + x_positions[1]) / 2,
        count_y,
        "个数",
        ha="center",
        va="center",
        fontsize=8.8,
        family=font_family,
    )
    for column_index, count in enumerate(best_counts, start=1):
        x = (x_positions[column_index] + x_positions[column_index + 1]) / 2
        ax.text(x, count_y, str(count), ha="center", va="center", fontsize=8.8, family=font_family)

    if show_note:
        direction = "越大越优" if higher_is_better else "越小越优"
        note = (
            f"注：数值为相对本行最优均值的差值±标准差（最优为 0，原指标{direction}）；"
            "每行加粗表示该测试函数上的最优算法。"
            + note_suffix
        )
        ax.text(0, -0.28, note, ha="left", va="center", fontsize=8.2, family=font_family)

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def write_ablation_table_csv(summary_rows, benchmark_ids, algorithms, metric, output_path, cell_suffixes=None):
    spec = METRIC_SPECS[metric]
    mean_key = spec["mean"]
    std_key = spec["std"]
    higher_is_better = spec["higher_is_better"]
    index = ablation_index(summary_rows)
    headers = [ABLATION_ALGORITHM_LABELS.get(algorithm, algorithm) for algorithm in algorithms]
    best_counts = [0] * len(algorithms)

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["benchmark_id", "function"] + headers)
        writer.writeheader()
        for benchmark_id in benchmark_ids:
            output_row = {"benchmark_id": benchmark_id, "function": function_label(benchmark_id)}
            best_value = ablation_best_value(index, benchmark_id, algorithms, mean_key, higher_is_better)
            for algorithm_index, algorithm in enumerate(algorithms):
                header = headers[algorithm_index]
                row = index.get((benchmark_id, algorithm))
                if row is None:
                    output_row[header] = ""
                    continue
                relative_mean = relative_to_best(row[mean_key], best_value, higher_is_better)
                suffix = (cell_suffixes or {}).get((benchmark_id, algorithm, metric), "")
                output_row[header] = f"{format_scientific(relative_mean)}±{format_scientific(row[std_key])}{suffix}"
                if is_best_cell(row[mean_key], best_value):
                    best_counts[algorithm_index] += 1
            writer.writerow(output_row)

        count_row = {"benchmark_id": "count", "function": "个数"}
        for header, count in zip(headers, best_counts):
            count_row[header] = count
        writer.writerow(count_row)


def tied_ranks(values, higher_is_better):
    values = np.asarray(values, dtype=float)
    ranked_values = -values if higher_is_better else values
    order = np.argsort(ranked_values)
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and np.isclose(ranked_values[order[end]], ranked_values[order[index]]):
            end += 1
        average_rank = (index + 1 + end) / 2.0
        ranks[order[index:end]] = average_rank
        index = end
    return ranks


def chi_square_sf(value, degrees_of_freedom):
    if value < 0:
        return float("nan")
    if degrees_of_freedom <= 0:
        return float("nan")
    if degrees_of_freedom % 2 == 0:
        half_value = value / 2.0
        terms = sum(half_value**index / math.factorial(index) for index in range(degrees_of_freedom // 2))
        return float(math.exp(-half_value) * terms)

    # Wilson-Hilferty normal approximation for odd degrees of freedom.
    z_value = ((value / degrees_of_freedom) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * degrees_of_freedom)))
    z_value /= math.sqrt(2.0 / (9.0 * degrees_of_freedom))
    return float(0.5 * math.erfc(z_value / math.sqrt(2.0)))


def friedman_rank_rows(summary_rows, benchmark_ids, algorithms, metric):
    spec = METRIC_SPECS[metric]
    mean_key = spec["mean"]
    higher_is_better = spec["higher_is_better"]
    index = ablation_index(summary_rows)
    rank_sums = dict.fromkeys(algorithms, 0.0)
    best_counts = dict.fromkeys(algorithms, 0)
    complete_blocks = []

    for benchmark_id in benchmark_ids:
        if any((benchmark_id, algorithm) not in index for algorithm in algorithms):
            continue
        values = np.asarray([index[(benchmark_id, algorithm)][mean_key] for algorithm in algorithms], dtype=float)
        if not np.all(np.isfinite(values)):
            continue
        ranks = tied_ranks(values, higher_is_better)
        complete_blocks.append(values)
        for algorithm, rank in zip(algorithms, ranks):
            rank_sums[algorithm] += float(rank)
            if math.isclose(rank, 1.0, rel_tol=1e-12, abs_tol=1e-12):
                best_counts[algorithm] += 1

    block_count = len(complete_blocks)
    if block_count == 0:
        raise ValueError(f"No complete ablation blocks for metric {metric!r}.")

    average_ranks = {algorithm: rank_sums[algorithm] / block_count for algorithm in algorithms}
    k = len(algorithms)
    friedman_statistic = (
        12.0 * block_count / (k * (k + 1.0)) * sum(value * value for value in average_ranks.values())
        - 3.0 * block_count * (k + 1.0)
    )
    p_value = chi_square_sf(friedman_statistic, k - 1)
    try:
        from scipy.stats import chi2

        p_value = float(chi2.sf(friedman_statistic, k - 1))
    except Exception:
        pass

    rows = [
        {
            "metric": metric,
            "algorithm": algorithm,
            "label": ABLATION_ALGORITHM_LABELS.get(algorithm, algorithm),
            "average_rank": average_ranks[algorithm],
            "best_count": best_counts[algorithm],
            "benchmark_count": block_count,
            "friedman_statistic": friedman_statistic,
            "p_value": p_value,
        }
        for algorithm in algorithms
    ]
    return sorted(rows, key=lambda item: (item["average_rank"], -item["best_count"], item["label"]))


def write_friedman_rank_csv(rows, output_path):
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "metric",
                "algorithm",
                "label",
                "average_rank",
                "best_count",
                "benchmark_count",
                "average_metric",
                "friedman_statistic",
                "p_value",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def draw_friedman_rank_bar(
    rows,
    output_path,
    show_title=False,
    x_label="算法变体",
    value_key="average_rank",
    y_label="排名",
    value_format="{:.2f}",
):
    colors = ["#f1b183", "#7fa6d9", "#a7dce0", "#4e8fc7", "#b7cee8", "#9fc490", "#d7b9d5"]
    labels = [row["label"] for row in rows]
    values = [row[value_key] for row in rows]
    metric_label = METRIC_SPECS[rows[0]["metric"]]["label"]

    font_family = ["Times New Roman", "SimSun"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)
    bars = ax.bar(labels, values, color=colors[: len(rows)], edgecolor="#606060", linewidth=0.7)
    ax.set_ylabel(y_label, fontsize=11, family=font_family)
    ax.set_xlabel(x_label, fontsize=11, family=font_family)
    if show_title:
        p_value = rows[0]["p_value"]
        p_text = f", p={p_value:.3g}" if np.isfinite(p_value) else ""
        title_value = "平均排名" if value_key == "average_rank" else "最优次数"
        ax.set_title(f"{metric_label} Friedman {title_value}{p_text}", fontsize=12, fontweight="bold", family=font_family)
    ax.set_ylim(0, max(values) * 1.22)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8.5, rotation=0)
    ax.tick_params(axis="y", labelsize=9.2)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.025,
            value_format.format(value),
            ha="center",
            va="bottom",
            fontsize=9.0,
            family=font_family,
        )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def sensitivity_rank_rows(summary_rows, metric, sort_by="best_count"):
    columns = sensitivity_parameter_columns(summary_rows)
    ranked_rows = rank_combinations_for_metric(summary_rows, metric, sort_by=sort_by)
    rows = []
    for row in ranked_rows:
        key = parameter_key(row, columns)
        rows.append(
            {
                "metric": metric,
                "algorithm": "|".join(format_rate(value) for value in key),
                "label": parameter_label(key),
                "average_rank": row["average_rank"],
                "best_count": row["best_count"],
                "benchmark_count": row["benchmark_count"],
                "average_metric": row["average_metric"],
                "friedman_statistic": float("nan"),
                "p_value": float("nan"),
            }
        )
    return rows


def draw_sensitivity_rank_outputs(args):
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = find_csv(
        input_dir,
        [
            "moiabc_archive_rate_sensitivity_summary_by_function.csv",
            "moiabc_sensitivity_summary_by_function.csv",
            "sensitivity_summary_by_function.csv",
        ],
    )
    summary_rows = exclude_duplicate_rows(normalize_rows(read_csv_rows(summary_path)))
    prefix = sensitivity_output_prefix(summary_rows)
    metrics = available_metrics(summary_rows)
    if args.metric:
        if args.metric not in metrics:
            available = ", ".join(metrics) or "none"
            raise ValueError(f"Metric {args.metric!r} is not available. Available metrics: {available}")
        metrics = [args.metric]

    outputs = []
    all_rank_rows = []
    for metric in metrics:
        suffix = METRIC_SPECS[metric]["suffix"]
        rank_rows = sensitivity_rank_rows(summary_rows, metric, sort_by="average_rank")[: args.top_k]
        output_png = output_dir / f"{prefix}_friedman_{suffix}_rank.png"
        output_csv = output_png.with_suffix(".csv")
        x_label = "外部档案引导率" if sensitivity_parameter_columns(summary_rows) == ("archive_rate",) else "参数组合"
        draw_friedman_rank_bar(rank_rows, output_png, show_title=False, x_label=x_label)
        write_friedman_rank_csv(rank_rows, output_csv)
        outputs.extend([output_png, output_csv])
        all_rank_rows.extend(rank_rows)

    output_csv = output_dir / f"{prefix}_friedman_all_ranks.csv"
    write_friedman_rank_csv(all_rank_rows, output_csv)
    outputs.append(output_csv)

    for output in outputs:
        print(output)


def read_filtered_wilcoxon_rows(input_dir):
    path = input_dir / "wilcoxon_ablation_vs_moiabc_results.csv"
    rows = read_csv_rows(path)
    return [row for row in rows if row["benchmark_id"] not in EXCLUDED_DUPLICATE_BENCHMARK_IDS]


def wilcoxon_sign(row):
    p_value = to_float(row["p_two_sided"])
    mean_difference = to_float(row["mean_difference"])
    if p_value >= 0.05 or math.isclose(mean_difference, 0.0, rel_tol=1e-12, abs_tol=1e-12):
        return "="
    return "+" if mean_difference > 0.0 else "-"


def ablation_cell_suffixes(input_dir, reference_algorithm="MOIABC"):
    try:
        rows = read_filtered_wilcoxon_rows(input_dir)
    except FileNotFoundError:
        return {}
    suffixes = {}
    for row in rows:
        if row["improved_algorithm"] != reference_algorithm:
            continue
        suffixes[(row["benchmark_id"], row["base_algorithm"], row["metric"])] = wilcoxon_sign(row)
    return suffixes


def ablation_effect_rows(wilcoxon_rows, algorithms):
    rows = []
    variants = [algorithm for algorithm in algorithms if algorithm != "MOIABC"]
    for variant in variants:
        for metric in ["hypervolume", "spacing", "best_sum"]:
            metric_rows = [
                row
                for row in wilcoxon_rows
                if row["base_algorithm"] == variant and row["metric"] == metric
            ]
            if not metric_rows:
                continue
            positive_rows = [row for row in metric_rows if float(row["mean_difference"]) > 0.0]
            significant_rows = [
                row
                for row in metric_rows
                if float(row["mean_difference"]) > 0.0 and float(row["p_improved"]) < 0.05
            ]
            opposite_rows = [
                row
                for row in metric_rows
                if float(row["mean_difference"]) < 0.0 and float(row["p_two_sided"]) < 0.05
            ]
            rows.append(
                {
                    "variant": variant,
                    "label": ABLATION_ALGORITHM_LABELS.get(variant, variant),
                    "metric": metric,
                    "benchmark_count": len(metric_rows),
                    "worse_count": len(positive_rows),
                    "significant_worse_count": len(significant_rows),
                    "opposite_significant_count": len(opposite_rows),
                    "mean_difference": float(np.mean([float(row["mean_difference"]) for row in metric_rows])),
                }
            )
    return rows


def write_ablation_effect_csv(rows, output_path):
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "variant",
                "label",
                "metric",
                "benchmark_count",
                "worse_count",
                "significant_worse_count",
                "opposite_significant_count",
                "mean_difference",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def try_write_csv(write_func, *args):
    try:
        write_func(*args)
        return True
    except PermissionError as error:
        print(f"Skip locked CSV: {error.filename}")
        return False


def draw_ablation_effect_table(rows, output_path, show_title=True):
    metrics = ["hypervolume", "spacing", "best_sum"]
    variants = []
    for row in rows:
        if row["variant"] not in variants:
            variants.append(row["variant"])
    index = {(row["metric"], row["variant"]): row for row in rows}

    column_headers = ["指标"] + [ABLATION_ALGORITHM_LABELS.get(variant, variant) for variant in variants]
    column_widths = [1.35] + [2.15] * len(variants)
    total_width = sum(column_widths)
    row_count = len(metrics)
    title_space = 0.58 if show_title else 0.05
    fig_width = max(10.8, total_width * 0.82)
    fig_height = 2.7 + title_space

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    ax.set_axis_off()
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, row_count + 1.45 + title_space)

    font_family = ["Times New Roman", "SimSun"]
    x_positions = np.cumsum([0] + column_widths)

    if show_title:
        ax.text(
            total_width / 2,
            row_count + 1.36,
            "MOIABC 消融模块退化统计",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            family=font_family,
        )

    top_y = row_count + 1.02
    header_y = row_count + 0.58
    bottom_y = 0.05
    ax.hlines(
        [top_y, header_y - 0.25, bottom_y],
        0,
        total_width,
        colors="black",
        linewidths=[1.35, 0.75, 1.35],
    )

    for column_index, header in enumerate(column_headers):
        x = (x_positions[column_index] + x_positions[column_index + 1]) / 2
        ax.text(x, header_y, header, ha="center", va="center", fontsize=9.2, family=font_family)

    for row_index, metric in enumerate(metrics):
        y = row_count - row_index - 0.02
        ax.text(
            (x_positions[0] + x_positions[1]) / 2,
            y,
            METRIC_SPECS[metric]["label"],
            ha="center",
            va="center",
            fontsize=9.0,
            family=font_family,
        )
        for column_index, variant in enumerate(variants, start=1):
            row = index[(metric, variant)]
            text = (
                f"{row['worse_count']}/{row['benchmark_count']}\n"
                f"显著 {row['significant_worse_count']}，反向 {row['opposite_significant_count']}"
            )
            fontweight = "bold" if row["significant_worse_count"] > row["opposite_significant_count"] else "normal"
            x = (x_positions[column_index] + x_positions[column_index + 1]) / 2
            ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                fontsize=8.0,
                fontweight=fontweight,
                family=font_family,
                linespacing=0.95,
            )

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def draw_ablation_outputs(args):
    input_dir = args.input_dir if args.input_dir != DEFAULT_INPUT_DIR else DEFAULT_ABLATION_INPUT_DIR
    output_dir = args.output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = find_csv(input_dir, ["moiabc_ablation_summary_by_function.csv"])
    summary_rows = exclude_duplicate_rows(normalize_ablation_rows(read_csv_rows(summary_path)))
    metrics = available_metrics(summary_rows)
    if args.metric:
        if args.metric not in metrics:
            available = ", ".join(metrics) or "none"
            raise ValueError(f"Metric {args.metric!r} is not available. Available metrics: {available}")
        metrics = [args.metric]

    benchmark_ids = sorted({row["benchmark_id"] for row in summary_rows}, key=benchmark_sort_key)
    algorithms = ablation_algorithms(summary_rows)
    cell_suffixes = ablation_cell_suffixes(input_dir)

    outputs = []
    if args.friedman:
        if args.mode == "sensitivity":
            draw_sensitivity_rank_outputs(args)
            return
        all_rank_rows = []
        for metric in metrics:
            suffix = METRIC_SPECS[metric]["suffix"]
            rank_rows = friedman_rank_rows(summary_rows, benchmark_ids, algorithms, metric)
            output_png = output_dir / f"moiabc_ablation_friedman_{suffix}_rank.png"
            output_csv = output_png.with_suffix(".csv")
            draw_friedman_rank_bar(rank_rows, output_png, show_title=not args.no_title)
            write_friedman_rank_csv(rank_rows, output_csv)
            outputs.extend([output_png, output_csv])
            all_rank_rows.extend(rank_rows)
        output_csv = output_dir / "moiabc_ablation_friedman_all_ranks.csv"
        write_friedman_rank_csv(all_rank_rows, output_csv)
        outputs.append(output_csv)
    else:
        for metric in metrics:
            suffix = METRIC_SPECS[metric]["suffix"]
            output_png = output_dir / f"moiabc_ablation_{suffix}_table.png"
            output_csv = output_png.with_suffix(".csv")
            draw_ablation_table(
                summary_rows,
                benchmark_ids,
                algorithms,
                metric,
                output_png,
                show_title=not args.no_title,
                show_note=not args.no_note,
                cell_suffixes=cell_suffixes,
                note_suffix="符号 +、=、- 分别表示 MOIABC 的性能优于、相近于或劣于对应消融变体。",
            )
            try_write_csv(write_ablation_table_csv, summary_rows, benchmark_ids, algorithms, metric, output_csv, cell_suffixes)
            outputs.extend([output_png, output_csv])

        wilcoxon_rows = read_filtered_wilcoxon_rows(input_dir)
        effect_rows = ablation_effect_rows(wilcoxon_rows, algorithms)
        output_png = output_dir / "moiabc_ablation_effect_summary_table.png"
        output_csv = output_png.with_suffix(".csv")
        draw_ablation_effect_table(effect_rows, output_png, show_title=not args.no_title)
        try_write_csv(write_ablation_effect_csv, effect_rows, output_csv)
        outputs.extend([output_png, output_csv])

    for output in outputs:
        print(output)


def write_table_csv_paper(summary_rows, benchmark_ids, combinations, metric, output_path):
    spec = METRIC_SPECS[metric]
    mean_key = spec["mean"]
    std_key = spec["std"]
    columns = sensitivity_parameter_columns(summary_rows)
    index = {
        parameter_index_key(row["benchmark_id"], parameter_key(row, columns)): row
        for row in summary_rows
    }
    headers = [combination_header(key) for key in combinations]
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
            for key in combinations:
                header = combination_header(key)
                row = index.get(parameter_index_key(benchmark_id, key))
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
    columns = sensitivity_parameter_columns(summary_rows)
    index = {
        parameter_index_key(row["benchmark_id"], parameter_key(row, columns)): row
        for row in summary_rows
    }
    mean_key = spec["mean"]
    std_key = spec["std"]
    higher_is_better = spec["higher_is_better"]

    column_headers = ["函数"] + [combination_header(key) for key in combinations]
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
        if columns == ("archive_rate",):
            title = f"表X MOIABC 不同 ra 的敏感性分析结果（{spec['label']}）"
        else:
            title = f"表X MOIABC 不同参数组合的敏感性分析结果（{spec['label']}）"
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
        for column_index, key in enumerate(combinations, start=1):
            row = index.get(parameter_index_key(benchmark_id, key))
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
        note = parameter_note(columns, higher_is_better)
        ax.text(0, -0.42, note, ha="left", va="center", fontsize=8.2, family=font_family)

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main():
    configure_fonts()
    args = parse_args()
    if args.mode == "ablation":
        draw_ablation_outputs(args)
        return
    if args.friedman:
        draw_sensitivity_rank_outputs(args)
        return

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = find_csv(
        input_dir,
        [
            "moiabc_archive_rate_sensitivity_summary_by_function.csv",
            "moiabc_sensitivity_summary_by_function.csv",
            "sensitivity_summary_by_function.csv",
        ],
    )
    summary_rows = exclude_duplicate_rows(normalize_rows(read_csv_rows(summary_path)))
    columns = sensitivity_parameter_columns(summary_rows)
    prefix = sensitivity_output_prefix(summary_rows)

    try:
        rank_path = find_csv(
            input_dir,
            [
                "moiabc_archive_rate_sensitivity_average_rank.csv",
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
        ranked_rows = rank_combinations_for_metric(
            summary_rows,
            metrics[0],
            sort_by=args.sort_by,
        )[: args.top_k]
        combinations = [parameter_key(row, columns) for row in ranked_rows]
    else:
        ranked_rows = rank_combinations_across_metrics(
            summary_rows,
            metrics,
            sort_by=args.sort_by,
        )[: args.top_k]
        if ranked_rows:
            combinations = [parameter_key(row, columns) for row in ranked_rows]
        else:
            combinations = select_combinations(summary_rows, rank_rows, args.top_k)

    if args.bar:
        if len(metrics) != 1:
            raise ValueError("Please pass --metric when using --bar so only one figure is generated.")
        metric = metrics[0]
        suffix = METRIC_SPECS[metric]["suffix"]
        output_png = output_dir / f"{prefix}_top{args.top_k}_{suffix}_bar.png"
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
        if len(metrics) > 1 and args.sort_by == "best_count":
            metric_ranked_rows = rank_combinations_for_metric(
                summary_rows,
                metric,
                sort_by=args.sort_by,
            )[: args.top_k]
            metric_combinations = [
                parameter_key(row, columns) for row in metric_ranked_rows
            ]
        else:
            metric_combinations = combinations
        if args.sort_by == "best_count":
            metric_combinations, _ = sort_combinations_by_table_count(
                summary_rows,
                benchmark_ids,
                metric_combinations,
                metric,
            )
        suffix = METRIC_SPECS[metric]["suffix"]
        output_png = output_dir / f"{prefix}_top{len(metric_combinations)}_{suffix}_table.png"
        output_csv = output_png.with_suffix(".csv")
        draw_table_paper(
            summary_rows,
            benchmark_ids,
            metric_combinations,
            metric,
            output_png,
            show_title=not args.no_title,
            show_note=not args.no_note,
        )
        write_table_csv_paper(summary_rows, benchmark_ids, metric_combinations, metric, output_csv)
        outputs.extend([output_png, output_csv])

    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

