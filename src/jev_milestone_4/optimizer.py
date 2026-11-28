"""Pure cost model and plan choice: no I/O, no SDK.

Input is the result of running two Milestone 2 plans (independent and combined)
on a seeded sample. Output is a predicted per-row cost for every candidate plan
and the cheapest allowed choice.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

from jev_milestone_2.experiment import Plan

# objective name -> field on a Milestone 2 call row ("calls" counts requests)
OBJECTIVES = {
    "usd": "estimated_cost_usd",
    "input_tokens": "input_tokens",
    "latency": "latency_seconds",
    "calls": None,
}
# objective name -> field on a Milestone 2 decision row (per-record totals)
DECISION_FIELDS = {
    "usd": "estimated_cost_usd",
    "input_tokens": "input_tokens",
    "latency": "api_latency_seconds",
    "calls": "model_calls",
}
MAX_FILTERS = 6  # 6! = 720 orders; beyond that, enumeration is not "simple"


def _mean(values: list[float | None]) -> float | None:
    if not values or any(v is None for v in values):
        return None
    return sum(values) / len(values)


def _call_costs(calls: list[dict]) -> dict[str, float | None]:
    return {
        objective: (
            (1.0 if calls else None)
            if field is None
            else _mean([c[field] for c in calls])
        )
        for objective, field in OBJECTIVES.items()
    }


SAMPLE_MODES = ("full", "cheap")


def build_estimates(
    filters: tuple[str, ...],
    *,
    matrix: dict[str, dict[str, int]],
    single_calls: dict[str, list[dict]],
    combined_calls: list[dict],
    guard_pairs: list[tuple[dict[str, int], dict[str, int]]],
    mode: str = "full",
) -> dict[str, Any]:
    """Shared estimate builder used by live runs (both modes) and offline replay.

    - `matrix`: per sample row, per filter, the 0/1 Jev decision used for pass rates.
    - `single_calls`: per filter, the one-question call rows used for per-call cost.
    - `combined_calls`: combined call rows used for combined cost per row.
    - `guard_pairs`: (combined per-filter decisions, independent per-filter decisions)
      for every row where both exist; the quality guard compares their ANDs.
    """
    if not matrix:
        raise ValueError("Sample has no rows")
    if not guard_pairs:
        raise ValueError("Quality guard needs at least one row with both plans")
    n = len(matrix)
    per_filter = {}
    for name in filters:
        passes = sum(row[name] for row in matrix.values())
        per_filter[name] = {
            "passes": passes,
            "selectivity": passes / n,
            "mean_cost": _call_costs(single_calls[name]),
            "n_calls": len(single_calls[name]),
        }
    g = len(guard_pairs)
    agreement = sum(all(c.values()) == all(i.values()) for c, i in guard_pairs) / g
    per_filter_agreement = {
        name: sum(c[name] == i[name] for c, i in guard_pairs) / g for name in filters
    }
    return {
        "mode": mode,
        "k": n,
        "guard_rows": g,
        "filters": list(filters),
        "per_filter": per_filter,
        "combined": {
            "mean_cost": _call_costs(combined_calls),
            "n_calls": len(combined_calls),
            "agreement_with_independent": agreement,
            "per_filter_agreement": per_filter_agreement,
        },
        "decision_matrix": matrix,
    }


def _decisions(result: dict[str, list[dict]], plan_name: str) -> dict[str, dict]:
    return {
        d["id"]: d
        for d in result["decisions"]
        if d["plan"] == plan_name and d["repeat"] == 1
    }


def _filter_decisions(d: dict, filters: tuple[str, ...]) -> dict[str, int]:
    return {name: int(d["filter_decisions"][name]) for name in filters}


def estimate_from_sample(
    sample_result: dict[str, list[dict]], filters: tuple[str, ...]
) -> dict[str, Any]:
    """Full mode (Milestone 4): independent and combined plans on every sample row.

    `sample_result` must come from `run_experiment` with exactly the plans
    `Plan("independent", filters)` and `Plan("combined", filters)`, one repeat.
    """
    independent = Plan("independent", filters).name
    combined = Plan("combined", filters).name
    calls = [c for c in sample_result["calls"] if c["status"] == "ok"]
    ind_decisions = _decisions(sample_result, independent)
    comb_decisions = _decisions(sample_result, combined)
    if not ind_decisions or set(ind_decisions) != set(comb_decisions):
        raise ValueError(
            "Sample must contain independent and combined runs over the same rows"
        )
    return build_estimates(
        filters,
        matrix={i: _filter_decisions(d, filters) for i, d in ind_decisions.items()},
        single_calls={
            name: [
                c for c in calls if c["plan"] == independent and c["filters"] == [name]
            ]
            for name in filters
        },
        combined_calls=[c for c in calls if c["plan"] == combined],
        guard_pairs=[
            (
                _filter_decisions(comb_decisions[i], filters),
                _filter_decisions(ind_decisions[i], filters),
            )
            for i in ind_decisions
        ],
        mode="full",
    )


def estimate_from_cheap_sample(
    combined_result: dict[str, list[dict]],
    guard_result: dict[str, list[dict]],
    filters: tuple[str, ...],
) -> dict[str, Any]:
    """Cheap mode (Milestone 5): combined plan on all k sample rows, independent
    plan only on a guard subset.

    The combined call already returns a decision for every filter, so pass rates
    come from the k combined calls. One-question costs and the quality guard come
    from the guard subset. Sampling then costs about k + g × n calls instead of
    k × (n + 1).
    """
    independent = Plan("independent", filters).name
    combined = Plan("combined", filters).name
    comb_decisions = _decisions(combined_result, combined)
    ind_decisions = _decisions(guard_result, independent)
    if not comb_decisions or not ind_decisions:
        raise ValueError("Cheap sample needs combined and guard runs")
    if not set(ind_decisions) <= set(comb_decisions):
        raise ValueError("Guard rows must be a subset of the sample rows")
    guard_calls = [c for c in guard_result["calls"] if c["status"] == "ok"]
    return build_estimates(
        filters,
        matrix={i: _filter_decisions(d, filters) for i, d in comb_decisions.items()},
        single_calls={
            name: [c for c in guard_calls if c["filters"] == [name]] for name in filters
        },
        combined_calls=[
            c
            for c in combined_result["calls"]
            if c["status"] == "ok" and c["plan"] == combined
        ],
        guard_pairs=[
            (
                _filter_decisions(comb_decisions[i], filters),
                _filter_decisions(ind_decisions[i], filters),
            )
            for i in ind_decisions
        ],
        mode="cheap",
    )


def reach_fractions(
    order: tuple[str, ...], matrix: dict[str, dict[str, int]]
) -> list[float]:
    """Fraction of sample rows that reach each stage (pass every earlier filter).

    Estimated jointly from the sample, so correlated filters are handled
    without assuming independence.
    """
    rows = list(matrix.values())
    reach, alive = [], rows
    for name in order:
        reach.append(len(alive) / len(rows))
        alive = [r for r in alive if r[name]]
    return reach


def sequential_cost(
    order: tuple[str, ...], estimates: dict[str, Any], objective: str
) -> float | None:
    """Expected cost per row: sum over stages of cost(stage) * P(reach stage)."""
    costs = [estimates["per_filter"][name]["mean_cost"][objective] for name in order]
    if any(c is None for c in costs):
        return None
    reach = reach_fractions(order, estimates["decision_matrix"])
    return sum(c * r for c, r in zip(costs, reach))


def independence_cost(
    order: tuple[str, ...], estimates: dict[str, Any], objective: str
) -> float | None:
    """Same formula but multiplying single selectivities (for comparison only)."""
    total, reach = 0.0, 1.0
    for name in order:
        cost = estimates["per_filter"][name]["mean_cost"][objective]
        if cost is None:
            return None
        total += cost * reach
        reach *= estimates["per_filter"][name]["selectivity"]
    return total


def choose_plan(
    estimates: dict[str, Any], *, objective: str, agreement_threshold: float
) -> dict[str, Any]:
    """Score every fixed order and the combined plan; pick the cheapest allowed."""
    if objective not in OBJECTIVES:
        raise ValueError(
            f"Unknown objective {objective!r}; choose from {sorted(OBJECTIVES)}"
        )
    if not 0 <= agreement_threshold <= 1:
        raise ValueError("agreement_threshold must be between 0 and 1")
    filters = tuple(estimates["filters"])
    if len(filters) > MAX_FILTERS:
        raise ValueError(f"At most {MAX_FILTERS} filters are supported")
    candidates = []
    for order in itertools.permutations(filters):
        plan = Plan("sequential", order)
        candidates.append(
            {
                "plan": plan.name,
                "strategy": "sequential",
                "filters": list(order),
                "predicted_cost_per_row": sequential_cost(order, estimates, objective),
                "independence_assumption_cost_per_row": independence_cost(
                    order, estimates, objective
                ),
                "reach_fractions": reach_fractions(order, estimates["decision_matrix"]),
                "allowed": True,
                "reason": None,
            }
        )
    agreement = estimates["combined"]["agreement_with_independent"]
    combined_ok = agreement >= agreement_threshold
    candidates.append(
        {
            "plan": Plan("combined", filters).name,
            "strategy": "combined",
            "filters": list(filters),
            "predicted_cost_per_row": estimates["combined"]["mean_cost"][objective],
            "independence_assumption_cost_per_row": None,
            "reach_fractions": None,
            "allowed": combined_ok,
            "reason": None
            if combined_ok
            else (
                f"quality guard: combined agreed with independent on {agreement:.3f} "
                f"of sample rows, below threshold {agreement_threshold}"
            ),
        }
    )
    scored = [
        c
        for c in candidates
        if c["allowed"] and c["predicted_cost_per_row"] is not None
    ]
    if not scored:
        raise ValueError(
            f"No candidate has a defined {objective!r} cost; supply pricing or pick another objective"
        )
    # Deterministic tie-break: cost, then sequential before combined, then name.
    chosen = min(
        scored,
        key=lambda c: (
            round(c["predicted_cost_per_row"], 12),
            c["strategy"] != "sequential",
            c["plan"],
        ),
    )
    ranked = sorted(
        candidates,
        key=lambda c: (
            not c["allowed"],
            math.inf
            if c["predicted_cost_per_row"] is None
            else c["predicted_cost_per_row"],
            c["plan"],
        ),
    )
    return {
        "objective": objective,
        "agreement_threshold": agreement_threshold,
        "combined_agreement_with_independent": agreement,
        "chosen": chosen["plan"],
        "chosen_strategy": chosen["strategy"],
        "chosen_filters": chosen["filters"],
        "candidates": ranked,
    }


def plan_from_choice(choice: dict[str, Any]) -> Plan:
    return Plan(choice["chosen_strategy"], tuple(choice["chosen_filters"]))


def actual_cost_per_row(
    decisions: list[dict], plan_name: str, objective: str
) -> float | None:
    field = DECISION_FIELDS[objective]
    return _mean(
        [d[field] for d in decisions if d["plan"] == plan_name and d["repeat"] == 1]
    )


def predicted_vs_actual(
    choice: dict[str, Any], execution: dict[str, list[dict]]
) -> dict[str, Any]:
    """Compare predicted per-row cost with the measured cost for every executed plan."""
    objective = choice["objective"]
    predicted = {c["plan"]: c["predicted_cost_per_row"] for c in choice["candidates"]}
    rows = {}
    for plan_name in dict.fromkeys(r["plan"] for r in execution["runs"]):
        actual = actual_cost_per_row(execution["decisions"], plan_name, objective)
        pred = predicted.get(plan_name)
        rows[plan_name] = {
            "predicted_cost_per_row": pred,
            "actual_cost_per_row": actual,
            "relative_error": (pred - actual) / actual
            if pred is not None and actual not in (None, 0)
            else None,
        }
    measured = {
        k: v["actual_cost_per_row"]
        for k, v in rows.items()
        if v["actual_cost_per_row"] is not None
    }
    best = min(measured, key=measured.get) if measured else None
    chosen_actual = measured.get(choice["chosen"])
    return {
        "objective": objective,
        "plans": rows,
        "cheapest_measured_plan": best,
        "chosen_was_cheapest": best == choice["chosen"] if len(measured) > 1 else None,
        "regret": chosen_actual / measured[best] - 1
        if len(measured) > 1 and chosen_actual is not None and measured[best]
        else None,
    }
