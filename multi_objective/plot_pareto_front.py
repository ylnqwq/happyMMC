# -*- coding: utf-8 -*-

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODULE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

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
COLORS = {
    "True PF": "#222222",
    "MO-DE": "#f1b183",
    "MOEA/D": "#7fa6d9",
    "MOPSO": "#a7dce0",
    "MOABC": "#4e8fc7",
    "Zhou-IMOABC": "#8b6bb8",
    "Yang-IGWO": "#58a65c",
    "ISSA": "#d39c31",
    "MOIABC": "#d66b5f",
}
MARKERS = {
    "MO-DE": "o",
    "MOEA/D": "s",
    "MOPSO": "^",
    "MOABC": "D",
    "Zhou-IMOABC": "v",
    "Yang-IGWO": "X",
    "ISSA": "h",
    "MOIABC": "P",
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
    parser = argparse.ArgumentParser(description="Draw a paper-style Pareto front comparison figure.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing *_archive_points.csv files.",
    )
    parser.add_argument(
        "--benchmark",
        default="ZDT1",
        help="Benchmark id, for example ZDT1, ZDT2, UF1 or MMF1. Default: ZDT1.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Default: <input-dir>/<benchmark>_paper_pareto_front.png.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=220,
        help="Maximum non-dominated points drawn per algorithm.",
    )
    return parser.parse_args()


def read_archive_points(path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("f1", "") == "" or row.get("f2", "") == "":
                continue
            rows.append(
                {
                    "algorithm": row["algorithm"],
                    "f1": float(row["f1"]),
                    "f2": float(row["f2"]),
                }
            )
    return rows


def true_pareto_front(benchmark_id):
    x = np.linspace(0.0, 1.0, 700)
    if benchmark_id in {"ZDT1", "ZDT4"}:
        return x, 1.0 - np.sqrt(x)
    if benchmark_id == "ZDT2":
        return x, 1.0 - x**2
    if benchmark_id == "ZDT3":
        intervals = [
            (0.0, 0.0830015349),
            (0.1822287280, 0.2577623634),
            (0.4093136748, 0.4538821041),
            (0.6183967944, 0.6525117038),
            (0.8233317983, 0.8518328654),
        ]
        front_x = np.concatenate([np.linspace(left, right, 120) for left, right in intervals])
        front_y = 1.0 - np.sqrt(front_x) - front_x * np.sin(10.0 * np.pi * front_x)
        return front_x, front_y
    if benchmark_id == "ZDT6":
        front_x = np.linspace(0.280775, 1.0, 700)
        return front_x, 1.0 - front_x**2
    return None


def select_plot_points(points, max_points):
    objectives = np.array([[point["f1"], point["f2"]] for point in points], dtype=float)
    order = np.lexsort((objectives[:, 1], objectives[:, 0]))
    objectives = objectives[order]
    selected = []
    best_f2 = np.inf
    for objective in objectives:
        if objective[1] < best_f2 - 1e-12:
            selected.append(objective)
            best_f2 = objective[1]
    objectives = np.asarray(selected, dtype=float)
    objectives = objectives[np.argsort(objectives[:, 0])]
    if max_points > 0 and len(objectives) > max_points:
        indexes = np.linspace(0, len(objectives) - 1, max_points).astype(int)
        objectives = objectives[indexes]
    return objectives


def draw_pareto_front(rows, benchmark_id, output_path, max_points):
    algorithms = [algorithm for algorithm in ALGORITHM_ORDER if any(row["algorithm"] == algorithm for row in rows)]
    algorithms.extend(sorted({row["algorithm"] for row in rows} - set(algorithms)))

    font_family = ["Times New Roman", "SimSun"]
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=300)

    true_front = true_pareto_front(benchmark_id)
    if true_front is not None:
        ax.plot(
            true_front[0],
            true_front[1],
            color=COLORS["True PF"],
            linewidth=1.8,
            label="True PF",
            zorder=4,
        )

    for algorithm in algorithms:
        points = [row for row in rows if row["algorithm"] == algorithm]
        objectives = select_plot_points(points, max_points)
        ax.scatter(
            objectives[:, 0],
            objectives[:, 1],
            s=22,
            alpha=0.78,
            marker=MARKERS.get(algorithm, "o"),
            color=COLORS.get(algorithm, None),
            edgecolors="white",
            linewidths=0.35,
            label=algorithm,
        )

    ax.set_xlabel("目标 f1", fontsize=11, family=font_family)
    ax.set_ylabel("目标 f2", fontsize=11, family=font_family)
    ax.set_title(f"{benchmark_id} Pareto 前沿对比图", fontsize=13, fontweight="bold", family=font_family)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9.5)
    ax.legend(frameon=False, fontsize=9.2, prop={"family": font_family}, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def main():
    configure_fonts()
    args = parse_args()
    benchmark_id = args.benchmark.upper()
    input_path = args.input_dir / f"{benchmark_id.lower()}_archive_points.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find archive points file: {input_path}")
    output_path = args.output or args.input_dir / f"{benchmark_id.lower()}_paper_pareto_front.png"
    rows = read_archive_points(input_path)
    draw_pareto_front(rows, benchmark_id, output_path, args.max_points)
    print(output_path)


if __name__ == "__main__":
    main()
