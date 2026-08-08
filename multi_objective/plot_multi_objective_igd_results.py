# -*- coding: utf-8 -*-

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

MODULE_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODULE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from multi_objective.mo_utils import non_dominated_mask
from multi_objective.plot_moiabc_sensitivity_results import (
    METRIC_SPECS,
    benchmark_sort_key,
    configure_fonts,
    draw_ablation_table,
    draw_friedman_rank_bar,
    friedman_rank_rows,
    write_ablation_table_csv,
    write_friedman_rank_csv,
)
from multi_objective.statistical_tests import paired_metric_summary


DEFAULT_INPUT_DIR = MODULE_DIR / "mo_comparison_results_improved_algorithms"
ALGORITHM_ORDER = [
    "MO-DE",
    "MOEA/D",
    "MOPSO",
    "MOABC",
    "Zhou-IMOABC",
    "Yang-IGWO",
    "CMMODE",
    "MOIABC",
]
METRICS = ["igd", "igd_plus"]
COLORS = {
    "MO-DE": "#f1b183",
    "MOEA/D": "#7fa6d9",
    "MOPSO": "#a7dce0",
    "MOABC": "#4e8fc7",
    "Zhou-IMOABC": "#8b6bb8",
    "Yang-IGWO": "#58a65c",
    "CMMODE": "#d39c31",
    "MOIABC": "#d66b5f",
}

METRIC_SPECS.update(
    {
        "igd": {
            "mean": "mean_igd",
            "std": "std_igd",
            "higher_is_better": False,
            "label": "IGD",
            "suffix": "igd",
        },
        "igd_plus": {
            "mean": "mean_igd_plus",
            "std": "std_igd_plus",
            "higher_is_better": False,
            "label": "IGD+",
            "suffix": "igd_plus",
        },
    }
)


def format_log_tick(value, _position):
    if value <= 0.0:
        return ""
    exponent = int(round(np.log10(value)))
    if np.isclose(value, 10.0**exponent):
        return f"1e{exponent}"
    return ""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recalculate and draw IGD/IGD+ comparison figures from saved archive points."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing *_archive_points.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: same as input-dir.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Use raw objective values instead of per-benchmark min-max normalized values.",
    )
    parser.add_argument(
        "--reference",
        default="MOIABC",
        help="Reference algorithm for Wilcoxon suffixes. Default: MOIABC.",
    )
    parser.add_argument(
        "--max-reference-points",
        type=int,
        default=2000,
        help="Maximum empirical reference-front points used per benchmark. Use 0 for all points.",
    )
    parser.add_argument(
        "--max-reference-candidates",
        type=int,
        default=12000,
        help="Maximum candidate points used before building a 3-objective empirical reference front.",
    )
    return parser.parse_args()


def read_csv_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv_rows(path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def archive_csv_paths(input_dir):
    return sorted(
        path
        for path in input_dir.glob("*_archive_points.csv")
        if path.name.lower().startswith(("zdt", "uf", "mmf"))
    )


def objective_columns(rows):
    columns = [key for key in rows[0] if key.startswith("f")]
    return sorted(
        [key for key in columns if key[1:].isdigit()],
        key=lambda item: int(item[1:]),
    )


def benchmark_id_from_archive_path(path):
    return path.name[: -len("_archive_points.csv")].upper()


def normalize_objectives(objectives, ideal, nadir):
    span = nadir - ideal
    span = np.where(np.isclose(span, 0.0), 1.0, span)
    return (objectives - ideal) / span


def group_archive_runs(rows, obj_cols):
    grouped = {}
    for row in rows:
        key = (row["algorithm"], int(float(row["run"])), row.get("seed", ""))
        grouped.setdefault(key, []).append([float(row[column]) for column in obj_cols if row.get(column, "") != ""])

    outputs = {}
    for key, values in grouped.items():
        objectives = np.asarray(values, dtype=float)
        if len(objectives) > 0:
            _, unique_indexes = np.unique(objectives, axis=0, return_index=True)
            objectives = objectives[np.sort(unique_indexes)]
            objectives = objectives[non_dominated_mask(objectives)]
        outputs[key] = objectives
    return outputs


def nondominated_front_2d(objectives):
    order = np.lexsort((objectives[:, 1], objectives[:, 0]))
    sorted_points = objectives[order]
    selected = []
    best_second = np.inf
    for point in sorted_points:
        if point[1] < best_second - 1e-12:
            selected.append(point)
            best_second = point[1]
    return np.asarray(selected, dtype=float)


def select_reference_candidates(objectives, max_candidates):
    if max_candidates <= 0 or len(objectives) <= max_candidates:
        return objectives

    selected = set(np.linspace(0, len(objectives) - 1, max_candidates, dtype=int).tolist())
    edge_count = max(20, max_candidates // (objectives.shape[1] * 20))
    for axis in range(objectives.shape[1]):
        order = np.argsort(objectives[:, axis])
        selected.update(order[:edge_count].tolist())
        selected.update(order[-edge_count:].tolist())
    indexes = np.fromiter(sorted(selected), dtype=int)
    return objectives[indexes]


def nondominated_front_3d(objectives):
    order = np.lexsort((objectives[:, 2], objectives[:, 1], objectives[:, 0]))
    front = np.empty((0, 3), dtype=float)

    for point in objectives[order]:
        if len(front) > 0:
            dominated_by_front = np.any(np.all(front <= point, axis=1) & np.any(front < point, axis=1))
            if dominated_by_front:
                continue
            dominated_front = np.all(point <= front, axis=1) & np.any(point < front, axis=1)
            if np.any(dominated_front):
                front = front[~dominated_front]
        front = np.vstack([front, point])
    return front


def nondominated_front_fast(objectives, max_candidates=12000):
    if objectives.shape[1] == 2:
        return nondominated_front_2d(objectives)
    if objectives.shape[1] == 3:
        candidates = select_reference_candidates(objectives, max_candidates)
        return nondominated_front_3d(candidates)
    return objectives[non_dominated_mask(objectives)]


def empirical_reference_front(run_objectives, max_candidates=12000):
    all_objectives = np.vstack([objectives for objectives in run_objectives if len(objectives) > 0])
    _, unique_indexes = np.unique(all_objectives, axis=0, return_index=True)
    all_objectives = all_objectives[np.sort(unique_indexes)]
    reference = nondominated_front_fast(all_objectives, max_candidates=max_candidates)
    order = np.lexsort(tuple(reference[:, index] for index in range(reference.shape[1] - 1, -1, -1)))
    return reference[order], all_objectives


def select_reference_points(reference, max_points):
    if max_points <= 0 or len(reference) <= max_points:
        return reference
    indexes = np.linspace(0, len(reference) - 1, max_points).astype(int)
    return reference[indexes]


def mean_min_distance(approximation_front, reference_front, plus=False, chunk_size=256):
    if len(approximation_front) == 0 or len(reference_front) == 0:
        return np.inf

    min_distances = []
    for start in range(0, len(reference_front), chunk_size):
        reference_chunk = reference_front[start : start + chunk_size]
        diff = approximation_front[None, :, :] - reference_chunk[:, None, :]
        if plus:
            diff = np.maximum(diff, 0.0)
        distances = np.linalg.norm(diff, axis=2)
        min_distances.append(np.min(distances, axis=1))
    return float(np.mean(np.concatenate(min_distances)))


def calculate_igd_rows(input_dir, normalize=True, max_reference_points=2000, max_reference_candidates=12000):
    rows = []
    for path in archive_csv_paths(input_dir):
        archive_rows = read_csv_rows(path)
        if not archive_rows:
            continue

        benchmark_id = benchmark_id_from_archive_path(path)
        obj_cols = objective_columns(archive_rows)
        grouped = group_archive_runs(archive_rows, obj_cols)
        reference, all_objectives = empirical_reference_front(
            list(grouped.values()),
            max_candidates=max_reference_candidates,
        )
        metric_reference = select_reference_points(reference, max_reference_points)

        if normalize:
            ideal = np.min(all_objectives, axis=0)
            nadir = np.max(all_objectives, axis=0)
            metric_reference = normalize_objectives(metric_reference, ideal, nadir)
        else:
            ideal = None
            nadir = None

        for (algorithm, run, seed), objectives in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
            metric_objectives = normalize_objectives(objectives, ideal, nadir) if normalize else objectives
            rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "algorithm": algorithm,
                    "run": run,
                    "seed": seed,
                    "archive_size": len(objectives),
                    "reference_front_size": len(reference),
                    "used_reference_front_size": len(metric_reference),
                    "igd": mean_min_distance(metric_objectives, metric_reference),
                    "igd_plus": mean_min_distance(metric_objectives, metric_reference, plus=True),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No *_archive_points.csv files found in {input_dir}")
    return rows


def algorithm_order(rows):
    present = {row["algorithm"] for row in rows}
    ordered = [algorithm for algorithm in ALGORITHM_ORDER if algorithm in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def summarize_rows(metric_rows):
    summary = []
    grouped = {}
    for row in metric_rows:
        grouped.setdefault((row["benchmark_id"], row["algorithm"]), []).append(row)

    for (benchmark_id, algorithm), items in sorted(grouped.items(), key=lambda item: (benchmark_sort_key(item[0][0]), item[0][1])):
        output = {
            "benchmark_id": benchmark_id,
            "variant": algorithm,
            "mean_igd": "",
            "std_igd": "",
            "mean_igd_plus": "",
            "std_igd_plus": "",
        }
        for metric in METRICS:
            values = np.asarray([float(item[metric]) for item in items], dtype=float)
            ddof = 1 if len(values) > 1 else 0
            output[METRIC_SPECS[metric]["mean"]] = float(np.mean(values))
            output[METRIC_SPECS[metric]["std"]] = float(np.std(values, ddof=ddof))
        summary.append(output)
    return summary


def grouped_for_statistics(metric_rows, algorithms):
    all_results = {}
    for row in sorted(metric_rows, key=lambda item: (benchmark_sort_key(item["benchmark_id"]), item["algorithm"], item["run"])):
        benchmark = row["benchmark_id"]
        algorithm = row["algorithm"]
        if algorithm not in algorithms:
            continue
        all_results.setdefault(benchmark, {name: [] for name in algorithms})
        all_results[benchmark][algorithm].append(row)
    return all_results


def write_wilcoxon_rows(path, all_results, algorithms, reference_algorithm):
    rows = []
    for benchmark_id, grouped_results in all_results.items():
        for base_algorithm in [algorithm for algorithm in algorithms if algorithm != reference_algorithm]:
            for metric in METRICS:
                summary = paired_metric_summary(
                    grouped_results,
                    base_algorithm,
                    reference_algorithm,
                    metric,
                    higher_is_better=False,
                )
                rows.append(
                    {
                        "benchmark_id": benchmark_id,
                        "metric": metric,
                        "better_direction": "lower",
                        "base_algorithm": base_algorithm,
                        "improved_algorithm": reference_algorithm,
                        "base_mean": summary["base_mean"],
                        "improved_mean": summary["improved_mean"],
                        "mean_difference": summary["mean_difference"],
                        "median_difference": summary["median_difference"],
                        "wins": summary["win_count"],
                        "ties": summary["tie_count"],
                        "losses": summary["lose_count"],
                        "wilcoxon_n": summary["n"],
                        "w_plus": summary["w_plus"],
                        "w_minus": summary["w_minus"],
                        "wilcoxon_statistic": summary["statistic"],
                        "p_two_sided": summary["p_two_sided"],
                        "p_improved": summary["p_improved"],
                        "significant_0_05": summary["p_two_sided"] < 0.05,
                    }
                )
    write_csv_rows(path, rows, list(rows[0]))
    return rows


def wilcoxon_sign(row):
    if float(row["p_two_sided"]) >= 0.05 or math.isclose(float(row["mean_difference"]), 0.0, rel_tol=1e-12, abs_tol=1e-12):
        return "="
    return "+" if float(row["mean_difference"]) > 0.0 else "-"


def cell_suffixes(wilcoxon_rows):
    return {
        (row["benchmark_id"], row["base_algorithm"], row["metric"]): wilcoxon_sign(row)
        for row in wilcoxon_rows
    }


def metric_boxplot_data(metric_rows, algorithms, metric):
    data = []
    for algorithm in algorithms:
        values = np.asarray(
            [float(row[metric]) for row in metric_rows if row["algorithm"] == algorithm],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        values = np.where(values <= 0.0, 1.0e-12, values)
        data.append(values)
    return data


def draw_metric_boxplot(metric_rows, algorithms, metric, output_path, axis=None):
    own_figure = axis is None
    if own_figure:
        fig, axis = plt.subplots(figsize=(7.6, 4.6), dpi=300)
    else:
        fig = axis.figure

    data = metric_boxplot_data(metric_rows, algorithms, metric)
    labels = [algorithm.replace("Zhou-", "Zhou-\n").replace("Yang-", "Yang-\n") for algorithm in algorithms]
    colors = [COLORS.get(algorithm, "#9a9a9a") for algorithm in algorithms]
    font_family = ["Times New Roman", "SimSun"]
    spec = METRIC_SPECS[metric]

    box = axis.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "#222222",
            "markeredgecolor": "#222222",
            "markersize": 3.2,
        },
        medianprops={"color": "#222222", "linewidth": 1.1},
        whiskerprops={"linewidth": 0.9},
        capprops={"linewidth": 0.9},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#606060",
            "markeredgecolor": "#606060",
            "markersize": 2.0,
            "alpha": 0.25,
        },
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.38)
        patch.set_edgecolor(color)

    axis.set_yscale("log")
    axis.yaxis.set_major_formatter(FuncFormatter(format_log_tick))
    axis.set_ylabel(f"{spec['label']} 指标值", fontsize=10.5, family=font_family)
    axis.set_xlabel("算法", fontsize=10.5, family=font_family)
    axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="x", labelsize=8.2, rotation=0)
    axis.tick_params(axis="y", labelsize=8.8)
    axis.set_title(f"{spec['label']} 箱线图", fontsize=12, fontweight="bold", family=font_family)

    if own_figure:
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)


def draw_igd_boxplot_overview(metric_rows, algorithms, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), dpi=300)
    for axis, metric in zip(axes, METRICS):
        draw_metric_boxplot(metric_rows, algorithms, metric, output_path=None, axis=axis)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def main():
    configure_fonts()
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = calculate_igd_rows(
        input_dir,
        normalize=not args.no_normalize,
        max_reference_points=args.max_reference_points,
        max_reference_candidates=args.max_reference_candidates,
    )
    algorithms = algorithm_order(metric_rows)
    if args.reference not in algorithms:
        valid = ", ".join(algorithms)
        raise ValueError(f"Reference algorithm {args.reference!r} not found. Available algorithms: {valid}")

    summary_rows = summarize_rows(metric_rows)
    benchmark_ids = sorted({row["benchmark_id"] for row in summary_rows}, key=benchmark_sort_key)
    all_results = grouped_for_statistics(metric_rows, algorithms)
    wilcoxon_rows = write_wilcoxon_rows(output_dir / "wilcoxon_igd_igd_plus_results.csv", all_results, algorithms, args.reference)
    suffixes = cell_suffixes(wilcoxon_rows)

    outputs = []
    metric_csv = output_dir / "igd_igd_plus_results.csv"
    write_csv_rows(
        metric_csv,
        metric_rows,
        [
            "benchmark_id",
            "algorithm",
            "run",
            "seed",
            "archive_size",
            "reference_front_size",
            "used_reference_front_size",
            "igd",
            "igd_plus",
        ],
    )
    outputs.append(metric_csv)

    summary_csv = output_dir / "mo_comparison_igd_summary_by_function.csv"
    write_csv_rows(
        summary_csv,
        summary_rows,
        ["benchmark_id", "variant", "mean_igd", "std_igd", "mean_igd_plus", "std_igd_plus"],
    )
    outputs.append(summary_csv)

    for metric in METRICS:
        suffix = METRIC_SPECS[metric]["suffix"]
        boxplot_png = output_dir / f"mo_comparison_{suffix}_boxplot.png"
        draw_metric_boxplot(metric_rows, algorithms, metric, boxplot_png)
        outputs.append(boxplot_png)

    overview_boxplot = output_dir / "mo_comparison_igd_igd_plus_boxplot.png"
    draw_igd_boxplot_overview(metric_rows, algorithms, overview_boxplot)
    outputs.append(overview_boxplot)

    for metric in METRICS:
        suffix = METRIC_SPECS[metric]["suffix"]
        table_png = output_dir / f"mo_comparison_{suffix}_table.png"
        table_csv = table_png.with_suffix(".csv")
        draw_ablation_table(
            summary_rows,
            benchmark_ids,
            algorithms,
            metric,
            table_png,
            show_title=True,
            title_prefix="表X 多目标算法对比实验结果",
            show_note=True,
            cell_suffixes=suffixes,
            note_suffix="符号 +、=、- 分别表示 MOIABC 的性能优于、相近于或劣于对应对比算法。",
        )
        write_ablation_table_csv(summary_rows, benchmark_ids, algorithms, metric, table_csv, suffixes)
        outputs.extend([table_png, table_csv])

        rank_rows = friedman_rank_rows(summary_rows, benchmark_ids, algorithms, metric)
        rank_png = output_dir / f"mo_comparison_friedman_{suffix}_rank.png"
        rank_csv = rank_png.with_suffix(".csv")
        draw_friedman_rank_bar(rank_rows, rank_png, show_title=False, x_label="算法", y_label="平均排名")
        write_friedman_rank_csv(rank_rows, rank_csv)
        outputs.extend([rank_png, rank_csv])

    outputs.append(output_dir / "wilcoxon_igd_igd_plus_results.csv")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
