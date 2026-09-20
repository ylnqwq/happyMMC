# -*- coding: utf-8 -*-

import argparse
import csv
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODULE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from multi_objective.experiment_utils import safe_filename_stem, save_rows_to_csv
from multi_objective.statistical_tests import (
    print_average_rank_overview,
    print_wilcoxon_overview,
    save_average_rank_results,
    save_wilcoxon_results,
)


STATISTICAL_TEST_METRICS = [
    ("hypervolume", True),
    ("spacing", False),
    ("best_sum", False),
    ("igd", False),
    ("igd_plus", False),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild multi-objective Wilcoxon and average-rank files from saved *_results.csv files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing per-benchmark multi-objective *_results.csv files.",
    )
    parser.add_argument(
        "--reference",
        default="MOIABC",
        help="Reference algorithm used as the improved algorithm in Wilcoxon comparisons.",
    )
    return parser.parse_args()


def read_result_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def result_csv_paths(input_dir):
    paths = []
    for path in sorted(input_dir.glob("*_results.csv")):
        if path.name.startswith("wilcoxon_"):
            continue
        rows = read_result_rows(path)
        if rows and {"benchmark_id", "algorithm", "run"}.issubset(rows[0]):
            paths.append((path, rows))
    return paths


def normalize_item(row):
    item = dict(row)
    for key, _ in STATISTICAL_TEST_METRICS:
        if key in item and item[key] != "":
            item[key] = float(item[key])
    item["run"] = int(item["run"])
    return item


def load_all_results(input_dir):
    all_results = {}
    algorithms = []
    for _, rows in result_csv_paths(input_dir):
        benchmark_id = rows[0]["benchmark_id"]
        grouped = {}
        for row in rows:
            item = normalize_item(row)
            algorithm = item["algorithm"]
            grouped.setdefault(algorithm, []).append(item)
            if algorithm not in algorithms:
                algorithms.append(algorithm)

        for items in grouped.values():
            items.sort(key=lambda item: item["run"])
        all_results[benchmark_id] = grouped

    if not all_results:
        raise FileNotFoundError(f"No per-benchmark *_results.csv files found in {input_dir}")
    return all_results, algorithms


def available_statistical_metrics(all_results, algorithms):
    metrics = []
    for metric_name, higher_is_better in STATISTICAL_TEST_METRICS:
        available = all(
            all(
                algorithm in grouped_results
                and all(metric_name in item and item[metric_name] != "" for item in grouped_results[algorithm])
                for algorithm in algorithms
            )
            for grouped_results in all_results.values()
        )
        if available:
            metrics.append((metric_name, higher_is_better))
    return metrics


def main():
    args = parse_args()
    input_dir = args.input_dir
    all_results, algorithms = load_all_results(input_dir)

    if args.reference not in algorithms:
        valid = ", ".join(algorithms)
        raise ValueError(f"Reference algorithm {args.reference!r} not found. Available algorithms: {valid}")

    metrics = available_statistical_metrics(all_results, algorithms)
    if not metrics:
        raise ValueError("No complete statistical metrics found in result CSV files.")

    wilcoxon_rows = []
    for base_algorithm in [name for name in algorithms if name != args.reference]:
        wilcoxon_rows.extend(
            save_wilcoxon_results(
                input_dir / f"wilcoxon_{safe_filename_stem(base_algorithm)}_vs_moiabc_results.csv",
                all_results,
                base_algorithm=base_algorithm,
                improved_algorithm=args.reference,
                metrics=metrics,
            )
        )

    save_rows_to_csv(input_dir / "wilcoxon_test_results.csv", wilcoxon_rows)
    rank_rows = save_average_rank_results(
        input_dir / "average_rank_results.csv",
        all_results,
        algorithms=algorithms,
        metrics=metrics,
    )

    print_wilcoxon_overview(wilcoxon_rows)
    print_average_rank_overview(rank_rows)
    print(f"Rebuilt statistics in: {input_dir.resolve()}")


if __name__ == "__main__":
    main()
