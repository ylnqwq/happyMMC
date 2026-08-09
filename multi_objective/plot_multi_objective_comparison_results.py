# -*- coding: utf-8 -*-

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODULE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from multi_objective.plot_moiabc_sensitivity_results import (
    METRIC_SPECS,
    available_metrics,
    benchmark_sort_key,
    configure_fonts,
    draw_ablation_table,
    draw_friedman_rank_bar,
    write_ablation_table_csv,
    write_friedman_rank_csv,
)


DEFAULT_INPUT_DIR = MODULE_DIR / "mo_comparison_results_standard_algorithms"
ALGORITHM_ORDER = [
    "MO-DE",
    "MOEA/D",
    "MOPSO",
    "MOABC",
    "Zhou-IMOABC",
    "Yang-IGWO",
    "ISSA",
    "MOIABC",
]
DEFAULT_METRICS = ["hypervolume", "spacing", "best_sum", "igd", "igd_plus"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw paper-style multi-objective algorithm comparison tables."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing merged multi-objective *_results.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: same as input-dir.",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(DEFAULT_METRICS),
        default=None,
        help="Metric to draw. Default: draw Hypervolume, Spacing and Best-sum.",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Do not draw the table title.",
    )
    return parser.parse_args()


def read_csv_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def result_csv_paths(input_dir):
    return sorted(
        path
        for path in input_dir.glob("*_results.csv")
        if path.name.lower().startswith(("zdt", "uf", "mmf"))
    )


def summarize_results(input_dir):
    grouped = {}
    algorithms = []
    present_metric_names = set()
    for path in result_csv_paths(input_dir):
        for row in read_csv_rows(path):
            benchmark_id = row["benchmark_id"]
            algorithm = row["algorithm"]
            grouped.setdefault((benchmark_id, algorithm), []).append(row)
            if algorithm not in algorithms:
                algorithms.append(algorithm)
            for metric in DEFAULT_METRICS:
                if metric in row and row[metric] != "":
                    present_metric_names.add(metric)

    if not grouped:
        raise FileNotFoundError(f"No benchmark *_results.csv files found in {input_dir}")

    rows = []
    for (benchmark_id, algorithm), items in sorted(grouped.items()):
        output_row = {
            "benchmark_id": benchmark_id,
            "variant": algorithm,
        }
        for metric in DEFAULT_METRICS:
            if metric not in present_metric_names:
                continue
            values = np.array([float(item[metric]) for item in items], dtype=float)
            ddof = 1 if len(values) > 1 else 0
            spec = METRIC_SPECS[metric]
            output_row[spec["mean"]] = float(np.mean(values))
            output_row[spec["std"]] = float(np.std(values, ddof=ddof))
        rows.append(output_row)

    ordered_algorithms = [algorithm for algorithm in ALGORITHM_ORDER if algorithm in algorithms]
    ordered_algorithms.extend(sorted(set(algorithms) - set(ordered_algorithms)))
    return rows, ordered_algorithms


def write_summary_csv(rows, output_path):
    fieldnames = ["benchmark_id", "variant"]
    for metric in DEFAULT_METRICS:
        spec = METRIC_SPECS[metric]
        if rows and spec["mean"] in rows[0] and spec["std"] in rows[0]:
            fieldnames.extend([spec["mean"], spec["std"]])
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_average_rank_rows(input_dir):
    path = input_dir / "average_rank_results.csv"
    rows = []
    for row in read_csv_rows(path):
        item = {
            "metric": row["metric"],
            "algorithm": row["algorithm"],
            "label": row["algorithm"],
            "average_rank": float(row["average_rank"]),
            "best_count": int(float(row["best_count"])),
            "benchmark_count": int(float(row["benchmark_count"])),
            "average_metric": "",
            "friedman_statistic": "",
            "p_value": "",
        }
        rows.append(item)
    return rows


def draw_average_rank_outputs(input_dir, output_dir, metrics):
    rows = read_average_rank_rows(input_dir)
    outputs = []
    if "hypervolume" in metrics and "spacing" in metrics:
        outputs.extend(draw_hv_spacing_rank_output(rows, output_dir))

    individual_metrics = [
        metric
        for metric in metrics
        if metric not in {"hypervolume", "spacing"}
    ]
    for metric in individual_metrics:
        suffix = METRIC_SPECS[metric]["suffix"]
        metric_rows = [row for row in rows if row["metric"] == metric]
        metric_rows.sort(key=lambda item: (item["average_rank"], -item["best_count"], item["label"]))
        output_png = output_dir / f"mo_comparison_friedman_{suffix}_rank.png"
        output_csv = output_png.with_suffix(".csv")
        draw_friedman_rank_bar(
            metric_rows,
            output_png,
            show_title=False,
            x_label="算法",
            y_label="平均排名",
        )
        write_friedman_rank_csv(metric_rows, output_csv)
        outputs.extend([output_png, output_csv])
    return outputs


def draw_hv_spacing_rank_output(rows, output_dir):
    metric_rows = {
        metric: {row["algorithm"]: row for row in rows if row["metric"] == metric}
        for metric in ("hypervolume", "spacing")
    }
    algorithms = sorted(
        set(metric_rows["hypervolume"]) & set(metric_rows["spacing"]),
        key=lambda algorithm: (
            (
                metric_rows["hypervolume"][algorithm]["average_rank"]
                + metric_rows["spacing"][algorithm]["average_rank"]
            )
            / 2.0,
            algorithm,
        ),
    )
    if not algorithms:
        return []

    output_png = output_dir / "mo_comparison_friedman_hv_spacing_rank.png"
    output_csv = output_png.with_suffix(".csv")
    csv_rows = []
    for algorithm in algorithms:
        hv_row = metric_rows["hypervolume"][algorithm]
        spacing_row = metric_rows["spacing"][algorithm]
        csv_rows.append(
            {
                "algorithm": algorithm,
                "hv_average_rank": hv_row["average_rank"],
                "spacing_average_rank": spacing_row["average_rank"],
                "mean_average_rank": (hv_row["average_rank"] + spacing_row["average_rank"]) / 2.0,
                "hv_best_count": hv_row["best_count"],
                "spacing_best_count": spacing_row["best_count"],
                "benchmark_count": hv_row["benchmark_count"],
            }
        )

    with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    font_family = ["Times New Roman", "SimSun"]
    x = np.arange(len(algorithms), dtype=float)
    width = 0.34
    hv_values = [metric_rows["hypervolume"][algorithm]["average_rank"] for algorithm in algorithms]
    spacing_values = [metric_rows["spacing"][algorithm]["average_rank"] for algorithm in algorithms]

    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=300)
    hv_bars = ax.bar(x - width / 2, hv_values, width, label="HV", color="#7fa6d9", edgecolor="#606060", linewidth=0.7)
    spacing_bars = ax.bar(x + width / 2, spacing_values, width, label="Spacing", color="#f1b183", edgecolor="#606060", linewidth=0.7)
    ax.set_ylabel("平均排名", fontsize=11, family=font_family)
    ax.set_xlabel("算法", fontsize=11, family=font_family)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=8.5, family=font_family)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9.2)
    ax.legend(frameon=False, prop={"family": font_family}, fontsize=9.2)

    max_value = max(hv_values + spacing_values)
    ax.set_ylim(0, max_value * 1.22)
    for bars in (hv_bars, spacing_bars):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max_value * 0.025,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.2,
                family=font_family,
            )

    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return [output_png, output_csv]


def wilcoxon_sign(row):
    p_value = float(row["p_two_sided"])
    mean_difference = float(row["mean_difference"])
    if p_value >= 0.05 or np.isclose(mean_difference, 0.0):
        return "="
    return "+" if mean_difference > 0.0 else "-"


def read_wilcoxon_suffixes(input_dir, reference_algorithm="MOIABC"):
    path = input_dir / "wilcoxon_test_results.csv"
    if not path.exists():
        return {}
    suffixes = {}
    for row in read_csv_rows(path):
        if row["improved_algorithm"] != reference_algorithm:
            continue
        suffixes[(row["benchmark_id"], row["base_algorithm"], row["metric"])] = wilcoxon_sign(row)
    return suffixes


def wilcoxon_footer_values(input_dir, algorithms, metric, reference_algorithm="MOIABC"):
    path = input_dir / "wilcoxon_test_results.csv"
    if not path.exists():
        return None

    counts = {
        algorithm: {"+": 0, "=": 0, "-": 0}
        for algorithm in algorithms
        if algorithm != reference_algorithm
    }
    for row in read_csv_rows(path):
        if row["metric"] != metric or row["improved_algorithm"] != reference_algorithm:
            continue
        base_algorithm = row["base_algorithm"]
        if base_algorithm not in counts:
            continue
        counts[base_algorithm][wilcoxon_sign(row)] += 1

    values = []
    for algorithm in algorithms:
        if algorithm == reference_algorithm:
            values.append("---")
            continue
        algorithm_counts = counts.get(algorithm)
        if algorithm_counts is None:
            values.append("")
        else:
            values.append(
                f"{algorithm_counts['+']}/{algorithm_counts['=']}/{algorithm_counts['-']}"
            )
    return values


def best_count_for_algorithm(summary_rows, benchmark_ids, algorithms, metric):
    spec = METRIC_SPECS[metric]
    mean_key = spec["mean"]
    higher_is_better = spec["higher_is_better"]
    index = {
        (row["benchmark_id"], row["variant"]): row
        for row in summary_rows
    }
    counts = {algorithm: 0 for algorithm in algorithms}

    for benchmark_id in benchmark_ids:
        values = [
            index[(benchmark_id, algorithm)][mean_key]
            for algorithm in algorithms
            if (benchmark_id, algorithm) in index
        ]
        if not values:
            continue
        best_value = max(values) if higher_is_better else min(values)
        for algorithm in algorithms:
            row = index.get((benchmark_id, algorithm))
            if row is not None and np.isclose(row[mean_key], best_value, rtol=1e-12, atol=1e-12):
                counts[algorithm] += 1
    return counts


def metric_algorithm_order(summary_rows, benchmark_ids, algorithms, metric, reference_algorithm="MOIABC"):
    if metric != "best_sum":
        return algorithms
    counts = best_count_for_algorithm(summary_rows, benchmark_ids, algorithms, metric)
    original_position = {algorithm: index for index, algorithm in enumerate(algorithms)}
    ordered = sorted(
        [algorithm for algorithm in algorithms if algorithm != reference_algorithm],
        key=lambda algorithm: (-counts[algorithm], original_position[algorithm]),
    )
    if reference_algorithm in algorithms:
        return [reference_algorithm] + ordered
    return ordered


def main():
    configure_fonts()
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows, algorithms = summarize_results(input_dir)
    benchmark_ids = sorted({row["benchmark_id"] for row in summary_rows}, key=benchmark_sort_key)
    metrics = [args.metric] if args.metric else [metric for metric in DEFAULT_METRICS if metric in available_metrics(summary_rows)]
    cell_suffixes = read_wilcoxon_suffixes(input_dir)

    summary_csv = output_dir / "mo_comparison_summary_by_function.csv"
    write_summary_csv(summary_rows, summary_csv)

    outputs = [summary_csv]
    for metric in metrics:
        suffix = METRIC_SPECS[metric]["suffix"]
        output_png = output_dir / f"mo_comparison_{suffix}_table.png"
        output_csv = output_png.with_suffix(".csv")
        metric_algorithms = metric_algorithm_order(summary_rows, benchmark_ids, algorithms, metric)
        footer_label = None
        footer_values = None
        if metric == "best_sum":
            footer_label = "+/=/-"
            footer_values = wilcoxon_footer_values(input_dir, metric_algorithms, metric)
        draw_ablation_table(
            summary_rows,
            benchmark_ids,
            metric_algorithms,
            metric,
            output_png,
            show_title=False,
            title_prefix="表X 多目标算法对比实验结果",
            show_note=False,
            cell_suffixes=cell_suffixes,
            note_suffix="符号 +、=、- 分别表示 MOIABC 的性能优于、相近于或劣于对应对比算法。",
            footer_label=footer_label,
            footer_values=footer_values,
        )
        write_ablation_table_csv(
            summary_rows,
            benchmark_ids,
            metric_algorithms,
            metric,
            output_csv,
            cell_suffixes,
            footer_label=footer_label,
            footer_values=footer_values,
        )
        outputs.extend([output_png, output_csv])

    outputs.extend(draw_average_rank_outputs(input_dir, output_dir, metrics))

    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
