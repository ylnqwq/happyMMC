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
    if abs(value) < 0.0005:
        value = 0.0
    return f"{value:.2E}".replace("E+0", "E+").replace("E-0", "E-")


def displayed_equal(left, right):
    return format_scientific(left) == format_scientific(right)


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
            for elite_rate, elimination_rate in combinations:
                header = f"e={format_rate(elite_rate)},d={format_rate(elimination_rate)}"
                row = index.get((benchmark_id, elite_rate, elimination_rate))
                if row is None:
                    output_row[header] = ""
                else:
                    mean = row[spec["mean"]]
                    std = row[spec["std"]]
                    output_row[header] = f"{format_scientific(mean)}±{format_scientific(std)}"
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
                text = f"{format_scientific(mean)}±{format_scientific(std)}"
                fontweight = (
                    "bold"
                    if best_value is not None
                    and (math.isclose(mean, best_value, rel_tol=1e-12, abs_tol=1e-12) or displayed_equal(mean, best_value))
                    else "normal"
                )
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
        note = f"注：e 表示 elite_rate，d 表示 elimination_rate；数值为平均值±标准差；每行加粗表示该测试函数上最优参数组合（{direction}）。"
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
    summary_rows = normalize_rows(read_csv_rows(summary_path))

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
    combinations = select_combinations(summary_rows, rank_rows, args.top_k)

    outputs = []
    for metric in metrics:
        suffix = METRIC_SPECS[metric]["suffix"]
        output_png = output_dir / f"moiabc_sensitivity_top{len(combinations)}_{suffix}_table.png"
        output_csv = output_png.with_suffix(".csv")
        draw_table(
            summary_rows,
            benchmark_ids,
            combinations,
            metric,
            output_png,
            show_title=not args.no_title,
            show_note=not args.no_note,
        )
        write_table_csv(summary_rows, benchmark_ids, combinations, metric, output_csv)
        outputs.extend([output_png, output_csv])

    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
