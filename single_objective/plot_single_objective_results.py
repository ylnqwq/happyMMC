# -*- coding: utf-8 -*-

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


DEFAULT_RESULT_DIR = Path("结果") / "2026.6.15单目标-前0.15后0.15-200轮"
FALLBACK_RESULT_DIR = Path("single_objective") / "comparison_results"
DEFAULT_ALGORITHM_ORDER = ["ABC", "GA", "ACO", "IABC-MSS", "NDBP-ABC", "IABC"]


def configure_fonts():
    font_paths = [
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
    parser = argparse.ArgumentParser(description="Draw a paper-style comparison result table from *_results.csv.")
    parser.add_argument(
        "--mode",
        choices=["comparison", "sensitivity"],
        default="comparison",
        help="Draw algorithm comparison table or IABC parameter sensitivity heatmaps.",
    )
    parser.add_argument(
        "--sensitivity-style",
        choices=["top-table", "full-table", "fixed-table", "heatmap"],
        default="top-table",
        help="Output style for --mode sensitivity.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top parameter combinations used by --sensitivity-style top-table.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR if DEFAULT_RESULT_DIR.exists() else FALLBACK_RESULT_DIR,
        help="Directory containing benchmark *_results.csv files.",
    )
    parser.add_argument(
        "--suite",
        choices=["all", "CEC2017", "CEC2022"],
        default="CEC2022",
        help="Benchmark suite to include.",
    )
    parser.add_argument(
        "--metric",
        choices=["best_value", "error"],
        default="error",
        help="Metric used in each table cell. Lower is better.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Improved algorithm used for +/-/= marks. Default: IABC if present.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Table title.",
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
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Default is written under input-dir.",
    )
    return parser.parse_args()


def get_sensitivity_input_dir(input_dir):
    if (input_dir / "sensitivity_average_rank.csv").exists():
        return input_dir
    default_dir = Path("single_objective") / "sensitivity_results"
    if (default_dir / "sensitivity_average_rank.csv").exists():
        return default_dir
    raise ValueError(
        "Cannot find sensitivity_average_rank.csv. "
        "Pass --input-dir pointing to single_objective/sensitivity_results."
    )


def benchmark_sort_key(benchmark_id):
    suite_match = re.search(r"CEC(\d+)_F(\d+)", benchmark_id)
    if suite_match:
        suite_year = int(suite_match.group(1))
        function_number = int(suite_match.group(2))
        return suite_year, function_number
    function_match = re.search(r"F(\d+)", benchmark_id)
    if function_match:
        return 9999, int(function_match.group(1))
    return 9999, benchmark_id


def function_label(benchmark_id):
    match = re.search(r"F(\d+)", benchmark_id)
    if match:
        return f"F{int(match.group(1)):02d}"
    return benchmark_id


def format_scientific(value):
    if not np.isfinite(value):
        return "nan"
    if abs(value) < 0.0005:
        value = 0.0
    return f"{value:.2E}".replace("E+0", "E+").replace("E-0", "E-")


def relation_mark(mean_value, target_mean, tolerance=1e-12):
    if format_scientific(mean_value) == format_scientific(target_mean):
        return "="
    if math.isclose(mean_value, target_mean, rel_tol=1e-10, abs_tol=tolerance):
        return "="
    if mean_value > target_mean:
        return "+"
    return "-"


def displayed_equal(left, right):
    return format_scientific(left) == format_scientific(right)


def read_result_rows(input_dir, suite):
    rows = []
    for csv_path in sorted(input_dir.glob("*_results.csv")):
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = set(reader.fieldnames or [])
            if not {"benchmark_id", "algorithm", "best_value", "error"}.issubset(fieldnames):
                continue
            for row in reader:
                benchmark_id = row.get("benchmark_id", "")
                if suite != "all" and not benchmark_id.startswith(f"{suite}_"):
                    continue
                rows.append(row)
    if not rows:
        raise ValueError(f"No result rows found in {input_dir} for suite={suite}.")
    return rows


def summarize(rows, metric):
    grouped = {}
    names = {}
    for row in rows:
        key = (row["benchmark_id"], row["algorithm"])
        grouped.setdefault(key, []).append(float(row[metric]))
        names[row["benchmark_id"]] = row.get("benchmark_name", row["benchmark_id"])

    summary = {}
    for (benchmark_id, algorithm), values in grouped.items():
        array = np.asarray(values, dtype=float)
        summary.setdefault(benchmark_id, {})[algorithm] = {
            "mean": float(np.mean(array)),
            "std": float(np.std(array, ddof=1 if len(array) > 1 else 0)),
            "run_times": len(array),
        }
    return summary, names


def ordered_algorithms(summary):
    algorithms = sorted({algorithm for values in summary.values() for algorithm in values})
    preferred = [algorithm for algorithm in DEFAULT_ALGORITHM_ORDER if algorithm in algorithms]
    remaining = [algorithm for algorithm in algorithms if algorithm not in preferred]
    return preferred + remaining


def choose_target(algorithms, explicit_target):
    if explicit_target:
        if explicit_target not in algorithms:
            raise ValueError(f"Target algorithm {explicit_target!r} is not present in result files.")
        return explicit_target
    if "IABC" in algorithms:
        return "IABC"
    return algorithms[-1]


def write_summary_csv(summary, benchmark_ids, algorithms, target_algorithm, output_path):
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = ["benchmark_id", "function"] + algorithms
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for benchmark_id in benchmark_ids:
            target_mean = summary[benchmark_id][target_algorithm]["mean"]
            row = {
                "benchmark_id": benchmark_id,
                "function": function_label(benchmark_id),
            }
            for algorithm in algorithms:
                item = summary[benchmark_id].get(algorithm)
                if item is None:
                    row[algorithm] = ""
                    continue
                mark = "" if algorithm == target_algorithm else relation_mark(item["mean"], target_mean)
                row[algorithm] = f"{format_scientific(item['mean'])}±{format_scientific(item['std'])}{mark}"
            writer.writerow(row)


def draw_table(summary, benchmark_ids, algorithms, target_algorithm, output_path, title, show_title=True, show_note=True):
    column_count = len(algorithms) + 1
    benchmark_count = len(benchmark_ids)
    cell_widths = [0.56] + [1.18] * len(algorithms)
    total_width = sum(cell_widths)
    fig_width = max(6.9, total_width * 0.9)
    fig_height = max(2.75, 0.76 + benchmark_count * 0.29 + (0.4 if show_title else 0.0) + (0.25 if show_note else 0.0))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    ax.set_axis_off()
    ax.set_xlim(0, total_width)
    if show_title or show_note:
        ax.set_ylim(-0.75 if show_note else -0.16, benchmark_count + (2.0 if show_title else 1.32))
    else:
        ax.set_ylim(0.58, benchmark_count + 1.52)

    font_family = ["Times New Roman", "SimSun"]
    title_text = title or "表  单目标优化算法对比实验结果"
    title_artist = ax.text(
        total_width / 2,
        benchmark_count + 1.66,
        title_text,
        ha="center",
        va="center",
        fontsize=11.6,
        fontweight="bold",
    )
    if not show_title:
        title_artist.set_visible(False)

    top_y = benchmark_count + (1.32 if show_title else 1.03)
    header_y = benchmark_count + (0.92 if show_title else 0.68)
    bottom_y = -0.08 if show_note else 0.78
    ax.hlines([top_y, header_y - 0.22, bottom_y], 0, total_width, colors="black", linewidths=[1.1, 0.65, 1.1])

    x_positions = np.cumsum([0] + cell_widths)
    headers = ["函数"] + algorithms
    for index, header in enumerate(headers):
        x = (x_positions[index] + x_positions[index + 1]) / 2
        ax.text(x, header_y, header, ha="center", va="center", fontsize=8.5, family=font_family)

    for row_index, benchmark_id in enumerate(benchmark_ids):
        y = benchmark_count - row_index + 0.22
        ax.text(
            (x_positions[0] + x_positions[1]) / 2,
            y,
            function_label(benchmark_id),
            ha="center",
            va="center",
            fontsize=8.2,
            family=font_family,
        )

        means = {
            algorithm: summary[benchmark_id][algorithm]["mean"]
            for algorithm in algorithms
            if algorithm in summary[benchmark_id]
        }
        best_mean = min(means.values())
        target_mean = summary[benchmark_id][target_algorithm]["mean"]

        for algorithm_index, algorithm in enumerate(algorithms, start=1):
            item = summary[benchmark_id].get(algorithm)
            if item is None:
                text = "-"
                fontweight = "normal"
            else:
                mark = "" if algorithm == target_algorithm else relation_mark(item["mean"], target_mean)
                text = f"{format_scientific(item['mean'])}±{format_scientific(item['std'])}{mark}"
                is_best = math.isclose(item["mean"], best_mean, rel_tol=1e-12, abs_tol=1e-12)
                fontweight = "bold" if is_best or displayed_equal(item["mean"], best_mean) else "normal"

            x = (x_positions[algorithm_index] + x_positions[algorithm_index + 1]) / 2
            ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                fontsize=7.1,
                fontweight=fontweight,
                family=font_family,
            )

    note = f"注：数值为均值±标准差；每行加粗表示最优；+/-/= 表示该算法相对 {target_algorithm} 更差/更优/相当。"
    note_artist = ax.text(0, 0.03, note, ha="left", va="bottom", fontsize=7.0, family=font_family)
    if not show_note:
        note_artist.set_visible(False)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def read_sensitivity_rows(input_dir):
    csv_path = input_dir / "sensitivity_average_rank.csv"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}.")
    for row in rows:
        row["elite_rate"] = float(row["elite_rate"])
        row["elimination_rate"] = float(row["elimination_rate"])
        row["average_rank"] = float(row["average_rank"])
        row["best_count"] = float(row["best_count"])
        row["benchmark_count"] = float(row["benchmark_count"])
        row["average_mean_error"] = float(row["average_mean_error"])
    return rows


def read_sensitivity_function_rows(input_dir):
    csv_path = input_dir / "sensitivity_summary_by_function.csv"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}.")
    for row in rows:
        row["elite_rate"] = float(row["elite_rate"])
        row["elimination_rate"] = float(row["elimination_rate"])
        row["mean_error"] = float(row["mean_error"])
        row["std_error"] = float(row["std_error"])
    return rows


def sensitivity_matrix(rows, value_key, elite_rates, elimination_rates):
    index = {
        (row["elite_rate"], row["elimination_rate"]): row[value_key]
        for row in rows
    }
    matrix = np.zeros((len(elite_rates), len(elimination_rates)), dtype=float)
    for row_index, elite_rate in enumerate(elite_rates):
        for column_index, elimination_rate in enumerate(elimination_rates):
            matrix[row_index, column_index] = index[(elite_rate, elimination_rate)]
    return matrix


def format_rate(rate):
    return f"{rate:.2f}".rstrip("0").rstrip(".")


def draw_sensitivity_heatmaps(rows, output_path, title):
    elite_rates = sorted({row["elite_rate"] for row in rows})
    elimination_rates = sorted({row["elimination_rate"] for row in rows})
    benchmark_count = int(rows[0]["benchmark_count"])
    best_row = min(rows, key=lambda row: (row["average_rank"], -row["best_count"]))

    panels = [
        ("average_rank", "平均排名", "viridis_r", "lower"),
        ("average_mean_error", "平均误差", "YlOrRd", "lower"),
        ("best_count", "最优次数", "Blues", "higher"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7), dpi=300, constrained_layout=True)
    font_family = ["Times New Roman", "SimSun"]
    fig.suptitle(title or "IABC 参数敏感性分析结果", fontsize=14, fontweight="bold", family=font_family)

    for axis, (value_key, panel_title, cmap, direction) in zip(axes, panels):
        matrix = sensitivity_matrix(rows, value_key, elite_rates, elimination_rates)
        image = axis.imshow(matrix, cmap=cmap, origin="lower", aspect="auto")
        axis.set_title(panel_title, fontsize=11.5, family=font_family)
        axis.set_xlabel("淘汰率 elimination_rate", fontsize=10, family=font_family)
        axis.set_ylabel("精英率 elite_rate", fontsize=10, family=font_family)
        axis.set_xticks(range(len(elimination_rates)), [format_rate(rate) for rate in elimination_rates])
        axis.set_yticks(range(len(elite_rates)), [format_rate(rate) for rate in elite_rates])

        row_index = elite_rates.index(best_row["elite_rate"])
        column_index = elimination_rates.index(best_row["elimination_rate"])
        axis.scatter(column_index, row_index, s=150, facecolors="none", edgecolors="red", linewidths=1.8)

        threshold = (np.nanmax(matrix) + np.nanmin(matrix)) / 2.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if value_key == "best_count":
                    text = f"{int(value)}"
                elif value_key == "average_rank":
                    text = f"{value:.2f}"
                else:
                    text = format_scientific(value)
                text_color = "white" if value > threshold else "black"
                axis.text(j, i, text, ha="center", va="center", fontsize=7.5, color=text_color)

        colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.ax.tick_params(labelsize=8)

    note = (
        f"推荐参数：elite_rate={best_row['elite_rate']:.2f}, "
        f"elimination_rate={best_row['elimination_rate']:.2f}, "
        f"average_rank={best_row['average_rank']:.3f}, "
        f"best_count={int(best_row['best_count'])}/{benchmark_count}"
    )
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=10, family=font_family)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def write_sensitivity_table_csv(rows, benchmark_ids, parameter_name, parameter_values, fixed_name, fixed_value, output_path):
    index = {
        (row["benchmark_id"], row[parameter_name]): row
        for row in rows
        if math.isclose(row[fixed_name], fixed_value, rel_tol=1e-12, abs_tol=1e-12)
    }
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = ["benchmark_id", "function"] + [f"{parameter_name}={format_rate(value)}" for value in parameter_values]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for benchmark_id in benchmark_ids:
            output_row = {"benchmark_id": benchmark_id, "function": function_label(benchmark_id)}
            for value in parameter_values:
                row = index.get((benchmark_id, value))
                output_row[f"{parameter_name}={format_rate(value)}"] = (
                    "" if row is None else f"{format_scientific(row['mean_error'])}±{format_scientific(row['std_error'])}"
                )
            writer.writerow(output_row)


def draw_sensitivity_table(
    rows,
    benchmark_ids,
    parameter_name,
    parameter_values,
    fixed_name,
    fixed_value,
    output_path,
    title,
):
    index = {
        (row["benchmark_id"], row[parameter_name]): row
        for row in rows
        if math.isclose(row[fixed_name], fixed_value, rel_tol=1e-12, abs_tol=1e-12)
    }
    column_headers = ["函数"] + [f"{parameter_name}={format_rate(value)}" for value in parameter_values]
    column_widths = [0.75] + [2.25] * len(parameter_values)
    total_width = sum(column_widths)
    benchmark_count = len(benchmark_ids)
    fig_width = max(10.0, total_width * 0.78)
    fig_height = max(4.2, 1.5 + benchmark_count * 0.34)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    ax.set_axis_off()
    ax.set_xlim(0, total_width)
    ax.set_ylim(-0.6, benchmark_count + 2.2)

    font_family = ["Times New Roman", "SimSun"]
    title_text = title or f"表  IABC 采用不同 {parameter_name} 取值的实验结果"
    ax.text(
        total_width / 2,
        benchmark_count + 1.75,
        title_text,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        family=font_family,
    )

    top_y = benchmark_count + 1.35
    header_y = benchmark_count + 0.9
    bottom_y = -0.08
    ax.hlines([top_y, header_y - 0.24, bottom_y], 0, total_width, colors="black", linewidths=[1.2, 0.7, 1.2])

    x_positions = np.cumsum([0] + column_widths)
    for column_index, header in enumerate(column_headers):
        x = (x_positions[column_index] + x_positions[column_index + 1]) / 2
        ax.text(x, header_y, header, ha="center", va="center", fontsize=10.2, family=font_family)

    for row_index, benchmark_id in enumerate(benchmark_ids):
        y = benchmark_count - row_index + 0.25
        ax.text(
            (x_positions[0] + x_positions[1]) / 2,
            y,
            function_label(benchmark_id),
            ha="center",
            va="center",
            fontsize=9.5,
            family=font_family,
        )

        values = [
            index[(benchmark_id, parameter_value)]["mean_error"]
            for parameter_value in parameter_values
            if (benchmark_id, parameter_value) in index
        ]
        best_value = min(values) if values else None

        for column_index, parameter_value in enumerate(parameter_values, start=1):
            row = index.get((benchmark_id, parameter_value))
            if row is None:
                text = "-"
                fontweight = "normal"
            else:
                text = f"{format_scientific(row['mean_error'])}±{format_scientific(row['std_error'])}"
                fontweight = (
                    "bold"
                    if best_value is not None
                    and math.isclose(row["mean_error"], best_value, rel_tol=1e-12, abs_tol=1e-12)
                    else "normal"
                )
            x = (x_positions[column_index] + x_positions[column_index + 1]) / 2
            ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                fontsize=8.8,
                fontweight=fontweight,
                family=font_family,
            )

    note = f"注：数值为平均误差±误差标准差；每行加粗表示该测试函数上的最优参数；固定 {fixed_name}={format_rate(fixed_value)}。"
    ax.text(0, -0.42, note, ha="left", va="center", fontsize=8.2, family=font_family)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def draw_sensitivity_tables(input_dir, average_rows):
    function_rows = read_sensitivity_function_rows(input_dir)
    benchmark_ids = sorted({row["benchmark_id"] for row in function_rows}, key=benchmark_sort_key)
    elite_rates = sorted({row["elite_rate"] for row in function_rows})
    elimination_rates = sorted({row["elimination_rate"] for row in function_rows})
    best_row = min(average_rows, key=lambda row: (row["average_rank"], -row["best_count"]))

    table_specs = [
        (
            "elite_rate",
            elite_rates,
            "elimination_rate",
            best_row["elimination_rate"],
            input_dir / "iabc_sensitivity_elite_rate_table.png",
            f"表  IABC 采用不同 elite_rate 取值的实验结果",
        ),
        (
            "elimination_rate",
            elimination_rates,
            "elite_rate",
            best_row["elite_rate"],
            input_dir / "iabc_sensitivity_elimination_rate_table.png",
            f"表  IABC 采用不同 elimination_rate 取值的实验结果",
        ),
    ]

    outputs = []
    for parameter_name, parameter_values, fixed_name, fixed_value, output_png, title in table_specs:
        output_csv = output_png.with_suffix(".csv")
        draw_sensitivity_table(
            function_rows,
            benchmark_ids,
            parameter_name,
            parameter_values,
            fixed_name,
            fixed_value,
            output_png,
            title,
        )
        write_sensitivity_table_csv(
            function_rows,
            benchmark_ids,
            parameter_name,
            parameter_values,
            fixed_name,
            fixed_value,
            output_csv,
        )
        outputs.append((output_png, output_csv))
    return outputs


def write_sensitivity_full_table_csv(rows, benchmark_ids, combinations, output_path):
    index = {
        (row["benchmark_id"], row["elite_rate"], row["elimination_rate"]): row
        for row in rows
    }
    headers = [f"e={format_rate(e)},d={format_rate(d)}" for e, d in combinations]
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["benchmark_id", "function"] + headers)
        writer.writeheader()
        for benchmark_id in benchmark_ids:
            output_row = {"benchmark_id": benchmark_id, "function": function_label(benchmark_id)}
            for (elite_rate, elimination_rate), header in zip(combinations, headers):
                row = index.get((benchmark_id, elite_rate, elimination_rate))
                output_row[header] = (
                    "" if row is None else f"{format_scientific(row['mean_error'])}±{format_scientific(row['std_error'])}"
                )
            writer.writerow(output_row)


def draw_sensitivity_full_table(rows, benchmark_ids, combinations, output_path, title):
    index = {
        (row["benchmark_id"], row["elite_rate"], row["elimination_rate"]): row
        for row in rows
    }
    column_headers = ["函数"] + [f"e={format_rate(e)}\nd={format_rate(d)}" for e, d in combinations]
    column_widths = [0.75] + [1.58] * len(combinations)
    total_width = sum(column_widths)
    benchmark_count = len(benchmark_ids)
    fig_width = max(18.0, total_width * 0.58)
    fig_height = max(4.8, 1.7 + benchmark_count * 0.36)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    ax.set_axis_off()
    ax.set_xlim(0, total_width)
    ax.set_ylim(-0.65, benchmark_count + 2.35)

    font_family = ["Times New Roman", "SimSun"]
    title_text = title or "表  IABC 参数敏感性完整对比实验结果"
    ax.text(
        total_width / 2,
        benchmark_count + 1.9,
        title_text,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        family=font_family,
    )

    top_y = benchmark_count + 1.5
    header_y = benchmark_count + 1.02
    bottom_y = -0.08
    ax.hlines([top_y, header_y - 0.34, bottom_y], 0, total_width, colors="black", linewidths=[1.2, 0.7, 1.2])

    x_positions = np.cumsum([0] + column_widths)
    for column_index, header in enumerate(column_headers):
        x = (x_positions[column_index] + x_positions[column_index + 1]) / 2
        ax.text(x, header_y, header, ha="center", va="center", fontsize=6.9, family=font_family, linespacing=1.0)

    for row_index, benchmark_id in enumerate(benchmark_ids):
        y = benchmark_count - row_index + 0.18
        ax.text(
            (x_positions[0] + x_positions[1]) / 2,
            y,
            function_label(benchmark_id),
            ha="center",
            va="center",
            fontsize=8.6,
            family=font_family,
        )

        values = [
            index[(benchmark_id, elite_rate, elimination_rate)]["mean_error"]
            for elite_rate, elimination_rate in combinations
            if (benchmark_id, elite_rate, elimination_rate) in index
        ]
        best_value = min(values) if values else None

        for column_index, (elite_rate, elimination_rate) in enumerate(combinations, start=1):
            row = index.get((benchmark_id, elite_rate, elimination_rate))
            if row is None:
                text = "-"
                fontweight = "normal"
            else:
                text = f"{format_scientific(row['mean_error'])}±{format_scientific(row['std_error'])}"
                fontweight = (
                    "bold"
                    if best_value is not None
                    and math.isclose(row["mean_error"], best_value, rel_tol=1e-12, abs_tol=1e-12)
                    else "normal"
                )
            x = (x_positions[column_index] + x_positions[column_index + 1]) / 2
            ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                fontsize=5.8,
                fontweight=fontweight,
                family=font_family,
            )

    note = "注：e 表示 elite_rate，d 表示 elimination_rate；数值为平均误差±误差标准差；每行加粗表示该测试函数上的最优参数组合。"
    ax.text(0, -0.45, note, ha="left", va="center", fontsize=8.2, family=font_family)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def draw_sensitivity_full_tables(input_dir):
    function_rows = read_sensitivity_function_rows(input_dir)
    benchmark_ids = sorted({row["benchmark_id"] for row in function_rows}, key=benchmark_sort_key)
    elite_rates = sorted({row["elite_rate"] for row in function_rows})
    elimination_rates = sorted({row["elimination_rate"] for row in function_rows})
    combinations = [(elite_rate, elimination_rate) for elite_rate in elite_rates for elimination_rate in elimination_rates]

    output_png = input_dir / "iabc_sensitivity_full_table.png"
    output_csv = output_png.with_suffix(".csv")
    draw_sensitivity_full_table(
        function_rows,
        benchmark_ids,
        combinations,
        output_png,
        "表  IABC 参数敏感性完整对比实验结果",
    )
    write_sensitivity_full_table_csv(function_rows, benchmark_ids, combinations, output_csv)
    return [(output_png, output_csv)]


def draw_sensitivity_top_table(input_dir, average_rows, top_k):
    function_rows = read_sensitivity_function_rows(input_dir)
    benchmark_ids = sorted({row["benchmark_id"] for row in function_rows}, key=benchmark_sort_key)
    top_rows = sorted(average_rows, key=lambda row: (row["average_rank"], -row["best_count"]))[:top_k]
    combinations = [(row["elite_rate"], row["elimination_rate"]) for row in top_rows]

    output_png = input_dir / f"iabc_sensitivity_top{top_k}_table.png"
    output_csv = output_png.with_suffix(".csv")
    draw_sensitivity_full_table(
        function_rows,
        benchmark_ids,
        combinations,
        output_png,
        f"表  IABC 敏感性分析前 {top_k} 组参数对比结果",
    )
    write_sensitivity_full_table_csv(function_rows, benchmark_ids, combinations, output_csv)
    return [(output_png, output_csv)]


def main():
    configure_fonts()
    args = parse_args()

    if args.mode == "sensitivity":
        input_dir = get_sensitivity_input_dir(args.input_dir)
        rows = read_sensitivity_rows(input_dir)
        if args.sensitivity_style == "heatmap":
            output_png = args.output or input_dir / "iabc_sensitivity_heatmaps.png"
            output_png.parent.mkdir(parents=True, exist_ok=True)
            draw_sensitivity_heatmaps(rows, output_png, args.title)
            outputs = [(output_png, None)]
        elif args.sensitivity_style == "fixed-table":
            outputs = draw_sensitivity_tables(input_dir, rows)
        elif args.sensitivity_style == "full-table":
            outputs = draw_sensitivity_full_tables(input_dir)
        else:
            outputs = draw_sensitivity_top_table(input_dir, rows, args.top_k)
        print(f"Input directory: {input_dir}")
        print("Mode: sensitivity")
        print(f"Style: {args.sensitivity_style}")
        for output_png, output_csv in outputs:
            print(f"PNG: {output_png.resolve()}")
            if output_csv is not None:
                print(f"CSV: {output_csv.resolve()}")
        return

    rows = read_result_rows(args.input_dir, args.suite)
    summary, _ = summarize(rows, args.metric)
    benchmark_ids = sorted(summary, key=benchmark_sort_key)
    algorithms = ordered_algorithms(summary)
    target_algorithm = choose_target(algorithms, args.target)

    suite_suffix = args.suite.lower()
    metric_suffix = args.metric.lower()
    output_png = args.output or args.input_dir / f"{suite_suffix}_{metric_suffix}_comparison_table.png"
    output_csv = output_png.with_suffix(".csv")
    output_png.parent.mkdir(parents=True, exist_ok=True)

    write_summary_csv(summary, benchmark_ids, algorithms, target_algorithm, output_csv)
    draw_table(
        summary,
        benchmark_ids,
        algorithms,
        target_algorithm,
        output_png,
        args.title,
        show_title=not args.no_title,
        show_note=not args.no_note,
    )

    print(f"Input directory: {args.input_dir}")
    print(f"Suite: {args.suite}")
    print(f"Metric: {args.metric}")
    print(f"Target algorithm: {target_algorithm}")
    print(f"Benchmarks: {len(benchmark_ids)}")
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"PNG: {output_png.resolve()}")
    print(f"CSV: {output_csv.resolve()}")


if __name__ == "__main__":
    main()
