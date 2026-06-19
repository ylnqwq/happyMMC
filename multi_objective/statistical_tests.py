# -*- coding: utf-8 -*-

import csv
import math
from itertools import product

import numpy as np


def _rank_absolute_values(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    index = 0

    while index < len(values):
        end = index + 1
        while end < len(values) and np.isclose(values[order[end]], values[order[index]]):
            end += 1
        average_rank = (index + 1 + end) / 2.0
        ranks[order[index:end]] = average_rank
        index = end

    return ranks


def _normal_cdf(value):
    return 0.5 * (1.0 + math.erf(value / np.sqrt(2.0)))


def wilcoxon_signed_rank(base_values, improved_values, higher_is_better=True):
    base_values = np.asarray(base_values, dtype=float)
    improved_values = np.asarray(improved_values, dtype=float)
    if base_values.shape != improved_values.shape:
        raise ValueError("Wilcoxon test requires paired samples with the same shape.")

    if higher_is_better:
        differences = improved_values - base_values
    else:
        differences = base_values - improved_values

    non_zero_mask = ~np.isclose(differences, 0.0)
    differences = differences[non_zero_mask]
    n = len(differences)
    if n == 0:
        return {
            "n": 0,
            "w_plus": 0.0,
            "w_minus": 0.0,
            "statistic": 0.0,
            "p_two_sided": 1.0,
            "p_improved": 1.0,
        }

    ranks = _rank_absolute_values(np.abs(differences))
    w_plus = float(np.sum(ranks[differences > 0]))
    w_minus = float(np.sum(ranks[differences < 0]))
    statistic = min(w_plus, w_minus)

    if n <= 20:
        rank_sums = []
        for signs in product((0, 1), repeat=n):
            signed_sum = float(np.sum(ranks[np.array(signs, dtype=bool)]))
            rank_sums.append(signed_sum)

        rank_sums = np.asarray(rank_sums, dtype=float)
        total_rank = float(np.sum(ranks))
        min_sums = np.minimum(rank_sums, total_rank - rank_sums)
        p_two_sided = float(np.mean(min_sums <= statistic + 1e-12))
        p_improved = float(np.mean(rank_sums >= w_plus - 1e-12))
    else:
        mean = n * (n + 1) / 4.0
        variance = n * (n + 1) * (2 * n + 1) / 24.0
        z_two_sided = (abs(w_plus - mean) - 0.5) / np.sqrt(variance)
        z_improved = (w_plus - mean - 0.5) / np.sqrt(variance)
        p_two_sided = float(2.0 * (1.0 - _normal_cdf(abs(z_two_sided))))
        p_improved = float(1.0 - _normal_cdf(z_improved))

    return {
        "n": n,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "statistic": statistic,
        "p_two_sided": max(0.0, min(1.0, p_two_sided)),
        "p_improved": max(0.0, min(1.0, p_improved)),
    }


def paired_metric_summary(grouped_results, base_algorithm, improved_algorithm, metric, higher_is_better):
    base = np.array([item[metric] for item in grouped_results[base_algorithm]], dtype=float)
    improved = np.array([item[metric] for item in grouped_results[improved_algorithm]], dtype=float)

    if higher_is_better:
        differences = improved - base
    else:
        differences = base - improved

    test = wilcoxon_signed_rank(base, improved, higher_is_better=higher_is_better)
    win_count = int(np.sum(differences > 0))
    tie_count = int(np.sum(np.isclose(differences, 0.0)))
    lose_count = int(np.sum(differences < 0))

    return {
        "base_mean": float(np.mean(base)),
        "improved_mean": float(np.mean(improved)),
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "win_count": win_count,
        "tie_count": tie_count,
        "lose_count": lose_count,
        **test,
    }


def save_wilcoxon_results(filename, all_results, base_algorithm, improved_algorithm, metrics):
    rows = []
    for benchmark_id, grouped_results in all_results.items():
        for metric_name, higher_is_better in metrics:
            summary = paired_metric_summary(
                grouped_results,
                base_algorithm,
                improved_algorithm,
                metric_name,
                higher_is_better,
            )
            rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "metric": metric_name,
                    "better_direction": "higher" if higher_is_better else "lower",
                    "base_algorithm": base_algorithm,
                    "improved_algorithm": improved_algorithm,
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

    with open(filename, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return rows


def average_rank_summary(all_results, algorithms, metrics):
    rows = []
    for metric_name, higher_is_better in metrics:
        rank_sums = {algorithm: 0.0 for algorithm in algorithms}
        win_counts = {algorithm: 0 for algorithm in algorithms}

        for grouped_results in all_results.values():
            means = {
                algorithm: float(np.mean([item[metric_name] for item in grouped_results[algorithm]]))
                for algorithm in algorithms
            }
            ordered = sorted(algorithms, key=lambda item: means[item], reverse=higher_is_better)
            for rank, algorithm in enumerate(ordered, start=1):
                rank_sums[algorithm] += rank
            win_counts[ordered[0]] += 1

        benchmark_count = len(all_results)
        for algorithm in algorithms:
            rows.append(
                {
                    "metric": metric_name,
                    "better_direction": "higher" if higher_is_better else "lower",
                    "algorithm": algorithm,
                    "average_rank": rank_sums[algorithm] / benchmark_count,
                    "best_count": win_counts[algorithm],
                    "benchmark_count": benchmark_count,
                }
            )

    return rows


def save_average_rank_results(filename, all_results, algorithms, metrics):
    rows = average_rank_summary(all_results, algorithms, metrics)
    with open(filename, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def print_wilcoxon_overview(rows):
    print("\n" + "=" * 80)
    print("Wilcoxon paired signed-rank test summary")
    print("Positive difference means the improved algorithm is better under the metric direction.")
    print("-" * 80)
    print("benchmark, metric, comparison, wins/ties/losses, mean_diff, p_two_sided, p_improved")
    for row in rows:
        print(
            f"{row['benchmark_id']}, {row['metric']}, "
            f"{row['base_algorithm']} -> {row['improved_algorithm']}, "
            f"{row['wins']}/{row['ties']}/{row['losses']}, "
            f"{row['mean_difference']:.6e}, "
            f"{row['p_two_sided']:.6g}, {row['p_improved']:.6g}"
        )


def print_average_rank_overview(rows):
    print("\n" + "=" * 80)
    print("Average rank summary across benchmarks")
    print("Lower average rank is better.")
    print("-" * 80)
    for row in rows:
        print(
            f"{row['metric']}, {row['algorithm']}: "
            f"average_rank={row['average_rank']:.3f}, "
            f"best_count={row['best_count']}/{row['benchmark_count']}"
        )
