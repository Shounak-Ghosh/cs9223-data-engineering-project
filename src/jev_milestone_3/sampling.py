"""Pure sampling and estimation functions: no I/O, no SDK.

`estimate_filter` output is the per-filter input for the cost-based optimizer;
keep its keys stable.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any, TypeVar

from jev_milestone_1.common import binary_metrics

T = TypeVar("T")

# Quantities `compare` reports, in output order.
COMPARED_QUANTITIES = (
    "selectivity",
    "gold_selectivity",
    "mean_input_tokens",
    "mean_output_tokens",
    "mean_latency_ms",
    "mean_cost_usd",
    "predicted_total_input_tokens",
    "predicted_total_output_tokens",
    "predicted_total_latency_s",
    "predicted_total_cost_usd",
    "precision",
    "recall",
    "f1",
    "accuracy",
)


def draw_sample(records: Sequence[T], k: int, seed: int) -> list[T]:
    """Seeded sample of k records without replacement, in original input order."""
    if type(k) is not int or k < 1 or k > len(records):
        raise ValueError(f"Sample size must be between 1 and {len(records)}; got {k}")
    indices = sorted(random.Random(seed).sample(range(len(records)), k))
    return [records[i] for i in indices]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, clamped to [0, 1]."""
    if n < 1:
        raise ValueError("Wilson interval needs n >= 1")
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    p = successes / n
    z2 = z * z
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    low = 0.0 if successes == 0 else max(0.0, center - half)
    high = 1.0 if successes == n else min(1.0, center + half)
    return low, high


def safe_binary_metrics(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    """`binary_metrics` with None (JSON null) for undefined precision/recall/F1."""
    metrics = binary_metrics(labels, predictions)
    tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    metrics.update(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=(tp + metrics["tn"]) / len(labels) if labels else None,
    )
    return metrics


def _mean_or_none(values: list[Any]) -> float | None:
    if not values or any(v is None for v in values):
        return None
    return sum(values) / len(values)


def _times(n: int, value: float | None) -> float | None:
    return None if value is None else n * value


def estimate_filter(rows: list[dict], *, population_size: int) -> dict[str, Any]:
    """Estimate one filter's selectivity, per-call cost, and quality from result rows.

    `rows` are successful per-call rows for one filter and one repeat. Totals are
    extrapolated to `population_size` records as N x the per-call mean.
    """
    if not rows:
        raise ValueError("estimate_filter needs at least one row")
    if type(population_size) is not int or population_size < 1:
        raise ValueError("population_size must be a positive integer")
    if any(row.get("status", "ok") != "ok" or "error_type" in row for row in rows):
        raise ValueError("estimate_filter received an error row")
    filters = {row["filter"] for row in rows}
    repeats = {row["repeat"] for row in rows}
    if len(filters) != 1 or len(repeats) != 1:
        raise ValueError(
            "estimate_filter needs rows from exactly one filter and repeat"
        )
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("estimate_filter received duplicate record IDs")
    thresholds = {row["threshold"] for row in rows}
    if len(thresholds) != 1:
        raise ValueError("estimate_filter rows use different thresholds")

    k = len(rows)
    predictions = [int(row["predicted"]) for row in rows]
    labels = [int(row["gold"]) for row in rows]
    passes = sum(predictions)
    low, high = wilson_interval(passes, k)
    quality = safe_binary_metrics(labels, predictions)

    has_usage = all(
        row.get("input_tokens") is not None and row.get("output_tokens") is not None
        for row in rows
    )
    mean_input = (
        _mean_or_none([row["input_tokens"] for row in rows]) if has_usage else None
    )
    mean_output = (
        _mean_or_none([row["output_tokens"] for row in rows]) if has_usage else None
    )
    mean_latency = _mean_or_none([row["latency_ms"] for row in rows])
    mean_cost = _mean_or_none([row.get("estimated_cost_usd") for row in rows])
    n = population_size
    return {
        "filter": filters.pop(),
        "threshold": thresholds.pop(),
        "population_size": n,
        "k": k,
        "passes": passes,
        "selectivity": passes / k,
        "selectivity_ci_low": low,
        "selectivity_ci_high": high,
        "gold_selectivity": sum(labels) / k,
        "mean_input_tokens": mean_input,
        "mean_output_tokens": mean_output,
        "mean_latency_ms": mean_latency,
        "mean_cost_usd": mean_cost,
        "predicted_total_input_tokens": _times(n, mean_input),
        "predicted_total_output_tokens": _times(n, mean_output),
        "predicted_total_latency_s": None
        if mean_latency is None
        else n * mean_latency / 1000,
        "predicted_total_cost_usd": _times(n, mean_cost),
        "tp": quality["tp"],
        "fp": quality["fp"],
        "fn": quality["fn"],
        "tn": quality["tn"],
        "precision": quality["precision"],
        "recall": quality["recall"],
        "f1": quality["f1"],
        "accuracy": quality["accuracy"],
    }


def compare(estimate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Per-quantity estimate vs. reference, with absolute and relative error.

    Errors are None when either side is None; relative error is None when the
    reference is 0. Relative error is |estimate - reference| / |reference|.
    """
    quantities = {}
    for name in COMPARED_QUANTITIES:
        est, ref = estimate.get(name), reference.get(name)
        if est is None or ref is None:
            error = abs_error = rel_error = None
        else:
            error = est - ref
            abs_error = abs(error)
            rel_error = None if ref == 0 else abs_error / abs(ref)
        quantities[name] = {
            "estimate": est,
            "reference": ref,
            "error": error,
            "abs_error": abs_error,
            "rel_error": rel_error,
        }
    return {
        "quantities": quantities,
        "ci_contains_reference": estimate["selectivity_ci_low"]
        <= reference["selectivity"]
        <= estimate["selectivity_ci_high"],
    }
