import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARCHIVE_CSV = RESULTS_DIR / "microgrid_all_algorithm_archives.csv"
OUTPUT_PNG = RESULTS_DIR / "algorithm_average_pareto_front.png"
OUTPUT_ZOOM_PNG = RESULTS_DIR / "algorithm_average_pareto_front_zoom.png"
OUTPUT_CSV = RESULTS_DIR / "algorithm_average_pareto_front.csv"

ALGORITHM_ORDER = ["MOIABC", "MOABC", "MOPSO", "MOEA/D", "MO-DE"]
ALGORITHM_COLORS = {
    "MOIABC": "#d95f5f",
    "MOABC": "#4c78a8",
    "MOPSO": "#72c7d3",
    "MOEA/D": "#5975a4",
    "MO-DE": "#f28e5c",
}
ALGORITHM_MARKERS = {
    "MOIABC": "o",
    "MOABC": "D",
    "MOPSO": "s",
    "MOEA/D": "^",
    "MO-DE": "X",
}


def configure_chinese_font():
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def read_archive_rows(path):
    grouped = defaultdict(lambda: defaultdict(list))
    with path.open(newline="", encoding="utf-8-sig") as file_obj:
        for row in csv.DictReader(file_obj):
            algorithm = row["algorithm"]
            run = int(row["run"])
            grouped[algorithm][run].append(
                (
                    float(row["economic_cost"]),
                    float(row["environment_cost"]),
                    float(row["penalty"]),
                )
            )
    return grouped


def nondominated_front(points):
    if not points:
        return np.empty((0, 2), dtype=float)

    values = np.asarray([(x, y) for x, y, _ in points], dtype=float)
    order = np.lexsort((values[:, 1], values[:, 0]))
    values = values[order]

    front = []
    best_environment = np.inf
    for economic_cost, environment_cost in values:
        if environment_cost < best_environment - 1e-12:
            front.append((economic_cost, environment_cost))
            best_environment = environment_cost

    if not front:
        return np.empty((0, 2), dtype=float)
    return np.asarray(front, dtype=float)


def unique_front_x(front):
    if len(front) <= 1:
        return front

    best_by_x = {}
    for economic_cost, environment_cost in front:
        previous = best_by_x.get(economic_cost)
        if previous is None or environment_cost < previous:
            best_by_x[economic_cost] = environment_cost

    values = np.asarray(sorted(best_by_x.items()), dtype=float)
    return values


def average_front(run_fronts, grid_size=80):
    usable = [unique_front_x(front) for front in run_fronts if len(front) >= 2]
    if not usable:
        centroids = np.asarray([front[0] for front in run_fronts if len(front) == 1], dtype=float)
        if len(centroids) == 0:
            return np.empty((0, 4), dtype=float)
        mean = np.mean(centroids, axis=0)
        return np.asarray([[mean[0], mean[1], 0.0, len(centroids)]], dtype=float)

    x_min = max(float(np.min(front[:, 0])) for front in usable)
    x_max = min(float(np.max(front[:, 0])) for front in usable)
    if x_min >= x_max:
        x_min = min(float(np.min(front[:, 0])) for front in usable)
        x_max = max(float(np.max(front[:, 0])) for front in usable)

    grid = np.linspace(x_min, x_max, grid_size)
    rows = []
    for x_value in grid:
        interpolated = []
        for front in usable:
            if front[0, 0] <= x_value <= front[-1, 0]:
                interpolated.append(float(np.interp(x_value, front[:, 0], front[:, 1])))
        if interpolated:
            rows.append(
                (
                    x_value,
                    float(np.mean(interpolated)),
                    float(np.std(interpolated, ddof=1)) if len(interpolated) > 1 else 0.0,
                    len(interpolated),
                )
            )

    return np.asarray(rows, dtype=float)


def write_average_csv(path, average_sets):
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "algorithm",
                "point_index",
                "economic_cost",
                "mean_environment_cost",
                "std_environment_cost",
                "contributing_runs",
            ],
        )
        writer.writeheader()
        for algorithm in ALGORITHM_ORDER:
            values = average_sets.get(algorithm)
            if values is None or len(values) == 0:
                continue
            for index, row in enumerate(values, start=1):
                writer.writerow(
                    {
                        "algorithm": algorithm,
                        "point_index": index,
                        "economic_cost": f"{row[0]:.6f}",
                        "mean_environment_cost": f"{row[1]:.6f}",
                        "std_environment_cost": f"{row[2]:.6f}",
                        "contributing_runs": int(row[3]),
                    }
                )


def plot_average_fronts(path, average_sets, xlim=None, ylim=None):
    fig, axis = plt.subplots(figsize=(8.4, 6.4))

    for algorithm in ALGORITHM_ORDER:
        values = average_sets.get(algorithm)
        if values is None or len(values) == 0:
            continue

        color = ALGORITHM_COLORS.get(algorithm)
        marker = ALGORITHM_MARKERS.get(algorithm, "o")
        axis.plot(
            values[:, 0],
            values[:, 1],
            color=color,
            linewidth=1.9,
            alpha=0.95,
            label=f"{algorithm} 平均前沿",
        )
        sample_step = max(1, len(values) // 16)
        axis.scatter(
            values[::sample_step, 0],
            values[::sample_step, 1],
            s=24,
            marker=marker,
            color=color,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.35,
        )

    axis.set_xlabel("经济成本 / 元")
    axis.set_ylabel("环境成本 / 元")
    if xlim is not None:
        axis.set_xlim(*xlim)
    if ylim is not None:
        axis.set_ylim(*ylim)
    axis.grid(True, linestyle="--", alpha=0.28)
    axis.legend(
        fontsize=13,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        framealpha=0.82,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    configure_chinese_font()
    grouped = read_archive_rows(ARCHIVE_CSV)
    average_sets = {}
    for algorithm in ALGORITHM_ORDER:
        run_fronts = [nondominated_front(points) for _, points in sorted(grouped[algorithm].items())]
        average_sets[algorithm] = average_front(run_fronts)

    write_average_csv(OUTPUT_CSV, average_sets)
    plot_average_fronts(OUTPUT_PNG, average_sets)
    zoom_values = [
        values[:, :2]
        for algorithm, values in average_sets.items()
        if algorithm != "MOEA/D" and values is not None and len(values) > 0
    ]
    if zoom_values:
        stacked = np.vstack(zoom_values)
        x_margin = 0.04 * (float(np.max(stacked[:, 0])) - float(np.min(stacked[:, 0])))
        y_margin = 0.08 * (float(np.max(stacked[:, 1])) - float(np.min(stacked[:, 1])))
        plot_average_fronts(
            OUTPUT_ZOOM_PNG,
            average_sets,
            xlim=(float(np.min(stacked[:, 0])) - x_margin, float(np.max(stacked[:, 0])) + x_margin),
            ylim=(float(np.min(stacked[:, 1])) - y_margin, float(np.max(stacked[:, 1])) + y_margin),
        )
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_ZOOM_PNG}")
    print(f"Saved {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
