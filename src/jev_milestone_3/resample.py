"""Offline resampling analysis over a full-data run: zero API calls.

A full run already holds Jev's answer, tokens, and latency for every record, so
drawing k of its repeat-1 rows is statistically equivalent to calling Jev on those
k records, except for run-to-run model noise (measured by repeat consistency).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any

from .sampling import compare, draw_sample, estimate_filter

DEFAULT_SIZES = (5, 10, 15, 20, 30, 50)
DEFAULT_DRAWS = 1000
DEFAULT_SEED = 0


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _p90(values: list[float]) -> float | None:
    """Nearest-rank 90th percentile (deterministic, no interpolation)."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(0.9 * len(ordered)) - 1]


def _mean_defined(values: list[float | None]) -> float | None:
    """Mean over draws, or None if any draw's value is undefined (e.g. no usage)."""
    if not values or any(v is None for v in values):
        return None
    return sum(values) / len(values)


def repeat_rows(full_rows: list[dict], repeat: int = 1) -> list[dict]:
    return [row for row in full_rows if row["repeat"] == repeat]


def repeat_consistency(full_rows: list[dict]) -> dict[str, Any] | None:
    """Decision agreement and probability spread across repeats; None if 1 repeat."""
    repeats = sorted({row["repeat"] for row in full_rows})
    if len(repeats) < 2:
        return None
    by_id: dict[str, list[dict]] = {}
    for row in full_rows:
        by_id.setdefault(row["id"], []).append(row)
    groups = [group for group in by_id.values() if len(group) == len(repeats)]
    if len(groups) != len(by_id):
        raise ValueError("Every record must appear once in every repeat")
    ranges = [
        max(r["probability"] for r in group) - min(r["probability"] for r in group)
        for group in groups
    ]
    selectivity = {
        str(rep): sum(r["predicted"] for r in full_rows if r["repeat"] == rep)
        / len(groups)
        for rep in repeats
    }
    return {
        "repeats": repeats,
        "n_records": len(groups),
        "decision_consistency": sum(
            len({r["predicted"] for r in group}) == 1 for group in groups
        )
        / len(groups),
        "n_inconsistent_records": sum(
            len({r["predicted"] for r in group}) > 1 for group in groups
        ),
        "max_probability_range": max(ranges),
        "mean_probability_range": sum(ranges) / len(ranges),
        "selectivity_by_repeat": selectivity,
    }


def validate_full_rows(full_rows: list[dict]) -> None:
    if not full_rows:
        raise ValueError("Full run has no rows")
    if any(row.get("status", "ok") != "ok" or "error_type" in row for row in full_rows):
        raise ValueError("Full run contains error rows")
    if len({row["filter"] for row in full_rows}) != 1:
        raise ValueError("Full run must contain exactly one filter")
    if not repeat_rows(full_rows):
        raise ValueError("Full run has no repeat-1 rows")


def aggregate_draws(
    comparisons: list[dict], estimates: list[dict], k: int
) -> dict[str, Any]:
    def errors(name: str, kind: str) -> list[float | None]:
        return [c["quantities"][name][kind] for c in comparisons]

    selectivity_errors = errors("selectivity", "abs_error")
    f1_values = [e["f1"] for e in estimates if e["f1"] is not None]
    draws = len(estimates)
    return {
        "k": k,
        "draws": draws,
        "mean_abs_error_selectivity": _mean(selectivity_errors),
        "p90_abs_error_selectivity": _p90(selectivity_errors),
        "max_abs_error_selectivity": max(selectivity_errors),
        "ci_coverage": sum(c["ci_contains_reference"] for c in comparisons) / draws,
        "mean_ci_width": _mean(
            [e["selectivity_ci_high"] - e["selectivity_ci_low"] for e in estimates]
        ),
        "zero_pass_fraction": sum(e["passes"] == 0 for e in estimates) / draws,
        "undefined_precision_fraction": sum(e["precision"] is None for e in estimates)
        / draws,
        "undefined_recall_fraction": sum(e["recall"] is None for e in estimates)
        / draws,
        "mean_rel_error_input_tokens": _mean_defined(
            errors("mean_input_tokens", "rel_error")
        ),
        "mean_rel_error_output_tokens": _mean_defined(
            errors("mean_output_tokens", "rel_error")
        ),
        "mean_rel_error_latency": _mean_defined(errors("mean_latency_ms", "rel_error")),
        "mean_rel_error_cost": _mean_defined(errors("mean_cost_usd", "rel_error")),
        "f1_defined_fraction": len(f1_values) / draws,
        "mean_f1": _mean(f1_values),
        "std_f1": statistics.pstdev(f1_values) if f1_values else None,
    }


def resampling_analysis(
    full_rows: list[dict],
    *,
    sizes: Sequence[int] = DEFAULT_SIZES,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
    population_size: int,
) -> dict[str, Any]:
    """Sampling error of `estimate_filter` vs. the full-data reference, per k.

    Draw i for every k uses seed `seed + i`, so the analysis is deterministic.
    """
    validate_full_rows(full_rows)
    reference_rows = repeat_rows(full_rows)
    if type(draws) is not int or draws < 1:
        raise ValueError("draws must be a positive integer")
    if not sizes or any(
        type(k) is not int or not 1 <= k <= len(reference_rows) for k in sizes
    ):
        raise ValueError(
            f"Every sample size must be between 1 and {len(reference_rows)}"
        )
    if len(set(sizes)) != len(sizes):
        raise ValueError("Sample sizes must be distinct")
    reference = estimate_filter(reference_rows, population_size=population_size)
    per_k = []
    for k in sizes:
        estimates, comparisons = [], []
        for i in range(draws):
            sample = draw_sample(reference_rows, k, seed + i)
            estimate = estimate_filter(sample, population_size=population_size)
            estimates.append(estimate)
            comparisons.append(compare(estimate, reference))
        per_k.append(aggregate_draws(comparisons, estimates, k))
    return {
        "filter": reference["filter"],
        "population_size": population_size,
        "n_reference_records": len(reference_rows),
        "sizes": list(sizes),
        "draws": draws,
        "seed": seed,
        "reference": reference,
        "per_k": per_k,
        "repeat_consistency": repeat_consistency(full_rows),
    }


# ---------------------------------------------------------------------------
# Ordering agreement between two filters (needs both full runs)

# Per-call cost measures, in order of preference; the first defined for both wins.
COST_BASES = ("mean_cost_usd", "mean_input_tokens", "mean_latency_ms")


def _cost_basis(*estimates: dict) -> str:
    for basis in COST_BASES:
        if all(e[basis] is not None for e in estimates):
            return basis
    raise ValueError("No per-call cost measure is defined for both filters")


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def order_costs(a: dict, b: dict, basis: str) -> dict[str, float]:
    """Expected per-record cost of each sequential order: cost(X) + sel(X) * cost(Y)."""
    return {
        "a_first": a[basis] + a["selectivity"] * b[basis],
        "b_first": b[basis] + b["selectivity"] * a[basis],
    }


def _choose(costs: dict[str, float]) -> str:
    # Ties go to a_first so the choice is deterministic; tie_fraction reports them.
    return "a_first" if costs["a_first"] <= costs["b_first"] else "b_first"


def ordering_agreement(
    rows_a: list[dict],
    rows_b: list[dict],
    *,
    sizes: Sequence[int] = DEFAULT_SIZES,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
    population_size: int,
) -> dict[str, Any]:
    """How often k-record samples lead to the same filter order as the full data.

    Each draw samples the same k record IDs for both filters (as the optimizer
    would: run both filters on one sample), using seed `seed + i`.
    - selectivity_order_agreement: the sampled pass rates rank the filters the
      same way as the full-data pass rates (a sampled tie counts as wrong).
    - plan_agreement: the order minimizing cost(X) + sel(X) * cost(Y) on the
      sample equals the full-data optimum.
    - regret: full-data cost of the chosen order / optimal full-data cost - 1.
    """
    for rows in (rows_a, rows_b):
        validate_full_rows(rows)
    first_a, first_b = repeat_rows(rows_a), repeat_rows(rows_b)
    name_a, name_b = first_a[0]["filter"], first_b[0]["filter"]
    if name_a == name_b:
        raise ValueError("Ordering agreement needs two different filters")
    ids = [row["id"] for row in first_a]
    by_id_b = {row["id"]: row for row in first_b}
    if len(set(ids)) != len(ids) or set(ids) != set(by_id_b):
        raise ValueError("Both full runs must cover the same record IDs once")
    if type(draws) is not int or draws < 1:
        raise ValueError("draws must be a positive integer")
    if not sizes or any(type(k) is not int or not 1 <= k <= len(ids) for k in sizes):
        raise ValueError(f"Every sample size must be between 1 and {len(ids)}")

    ref_a = estimate_filter(first_a, population_size=population_size)
    ref_b = estimate_filter(first_b, population_size=population_size)
    basis = _cost_basis(ref_a, ref_b)
    true_costs = order_costs(ref_a, ref_b, basis)
    best = _choose(true_costs)
    optimal = true_costs[best]
    true_sign = _sign(ref_a["selectivity"] - ref_b["selectivity"])
    by_id_a = {row["id"]: row for row in first_a}

    per_k = []
    for k in sizes:
        sel_agree = sel_ties = plan_agree = plan_ties = 0
        regrets = []
        for i in range(draws):
            sample = draw_sample(ids, k, seed + i)
            est_a = estimate_filter(
                [by_id_a[x] for x in sample], population_size=population_size
            )
            est_b = estimate_filter(
                [by_id_b[x] for x in sample], population_size=population_size
            )
            sampled_sign = _sign(est_a["selectivity"] - est_b["selectivity"])
            sel_agree += sampled_sign == true_sign and sampled_sign != 0
            sel_ties += sampled_sign == 0
            costs = order_costs(est_a, est_b, basis)
            plan_ties += costs["a_first"] == costs["b_first"]
            chosen = _choose(costs)
            plan_agree += chosen == best
            regrets.append(true_costs[chosen] / optimal - 1)
        per_k.append(
            {
                "k": k,
                "draws": draws,
                "selectivity_order_agreement": sel_agree / draws,
                "selectivity_tie_fraction": sel_ties / draws,
                "plan_agreement": plan_agree / draws,
                "plan_tie_fraction": plan_ties / draws,
                "mean_regret": sum(regrets) / draws,
                "max_regret": max(regrets),
            }
        )
    return {
        "filters": {"a": name_a, "b": name_b},
        "population_size": population_size,
        "sizes": list(sizes),
        "draws": draws,
        "seed": seed,
        "cost_basis": basis,
        "reference": {
            "selectivity": {name_a: ref_a["selectivity"], name_b: ref_b["selectivity"]},
            "per_call_cost": {name_a: ref_a[basis], name_b: ref_b[basis]},
            "order_costs_per_record": {
                f"{name_a} first": true_costs["a_first"],
                f"{name_b} first": true_costs["b_first"],
            },
            "best_order": [name_a, name_b] if best == "a_first" else [name_b, name_a],
            "wrong_order_regret": max(true_costs.values()) / optimal - 1,
        },
        "per_k": per_k,
    }
