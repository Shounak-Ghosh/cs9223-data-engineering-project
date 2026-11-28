"""Offline replay of execution plans and of the optimizer from recorded Jev calls.

Every plan in this project is built from two kinds of request: a one-question call
(`filters == [f]`) and a combined call (`filters == query filters`). Jev input
tokens depend only on the request contents, so once every row has a recorded
one-question call per filter and a combined call per query, any plan (and the
optimizer with any seed, sample size, or sampling mode) can be replayed without
new API calls. `check_repeats` measures how far that assumption holds.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from jev_milestone_1.common import binary_metrics
from jev_milestone_2.experiment import Plan, Pricing
from jev_milestone_3.sampling import draw_sample
from jev_milestone_4.optimizer import (
    DECISION_FIELDS,
    OBJECTIVES,
    build_estimates,
    choose_plan,
    plan_from_choice,
)

COST_FIELDS = ("input_tokens", "output_tokens", "estimated_cost_usd", "latency_seconds")


def query_key(filters: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(filters))


@dataclass
class CallStore:
    """Recorded calls keyed by (row id, filter) and (row id, sorted query)."""

    threshold: float = 0.5
    single: dict[tuple[str, str], dict] = field(default_factory=dict)
    combined: dict[tuple[str, tuple[str, ...]], dict] = field(default_factory=dict)
    repeats: list[tuple[dict, dict]] = field(default_factory=list)

    def add(self, call: dict, pricing: Pricing | None = None) -> None:
        if call.get("status") != "ok":
            return
        call = {
            k: call.get(k) for k in (*COST_FIELDS, "id", "filters", "probabilities")
        }
        if pricing is not None:
            call["estimated_cost_usd"] = (
                call["input_tokens"] * pricing.input_per_million
                + call["output_tokens"] * pricing.output_per_million
            ) / 1_000_000
        if len(call["filters"]) == 1:
            store, key = self.single, (call["id"], call["filters"][0])
        else:
            store, key = self.combined, (call["id"], query_key(call["filters"]))
        if key in store:
            self.repeats.append((store[key], call))  # first recording wins
        else:
            store[key] = call

    def decision(self, call: dict, name: str) -> int:
        return int(call["probabilities"][name] >= self.threshold)

    def missing(self, ids: list[str], filters: tuple[str, ...]) -> dict[str, int]:
        gaps = {f: sum((i, f) not in self.single for i in ids) for f in filters}
        gaps["combined"] = sum(
            (i, query_key(filters)) not in self.combined for i in ids
        )
        return {k: v for k, v in gaps.items() if v}


def check_repeats(store: CallStore) -> dict[str, Any]:
    """How often repeated requests for the same row and question set agree."""
    pairs = store.repeats
    same_tokens = sum(a["input_tokens"] == b["input_tokens"] for a, b in pairs)
    same_decisions = sum(
        all(store.decision(a, f) == store.decision(b, f) for f in a["probabilities"])
        for a, b in pairs
    )
    return {
        "repeated_requests": len(pairs),
        "same_input_tokens": same_tokens,
        "same_decisions": same_decisions,
    }


@dataclass
class PlanRun:
    calls: list[dict]
    filter_decisions: dict[str, dict[str, int]]  # evaluated filters only
    predicted: dict[str, int]


def run_plan(store: CallStore, plan: Plan, ids: list[str]) -> PlanRun:
    """Replay one plan over rows, exactly as Milestone 2 executes it."""
    calls, fds, predicted = [], {}, {}
    for i in ids:
        fd: dict[str, int] = {}
        if plan.strategy == "combined":
            call = store.combined[(i, query_key(plan.filters))]
            calls.append(call)
            fd = {f: store.decision(call, f) for f in plan.filters}
        else:
            for f in plan.filters:
                call = store.single[(i, f)]
                calls.append(call)
                fd[f] = store.decision(call, f)
                if plan.strategy == "sequential" and not fd[f]:
                    break
        fds[i] = fd
        predicted[i] = int(len(fd) == len(plan.filters) and all(fd.values()))
    return PlanRun(calls, fds, predicted)


def totals(calls: list[dict]) -> dict[str, float | None]:
    out: dict[str, float | None] = {"calls": float(len(calls))}
    for objective, name in OBJECTIVES.items():
        if name is None:
            continue
        values = [c[name] for c in calls]
        out[objective] = None if any(v is None for v in values) else float(sum(values))
    return out


def candidate_plans(filters: tuple[str, ...]) -> list[Plan]:
    return [Plan("sequential", o) for o in itertools.permutations(filters)] + [
        Plan("combined", filters)
    ]


def quality(gold: dict[str, int], predicted: dict[str, int]) -> dict[str, Any]:
    ids = sorted(gold)
    return binary_metrics([gold[i] for i in ids], [predicted[i] for i in ids])


def baselines(
    store: CallStore, filters: tuple[str, ...], ids: list[str], gold: dict[str, int]
) -> dict[str, dict]:
    """Every candidate plan (and the independent plan) over all rows."""
    out = {}
    for plan in [*candidate_plans(filters), Plan("independent", filters)]:
        run = run_plan(store, plan, ids)
        out[plan.name] = {
            "strategy": plan.strategy,
            "totals": totals(run.calls),
            "per_row": {
                k: v / len(ids) for k, v in totals(run.calls).items() if v is not None
            },
            "quality": quality(gold, run.predicted),
            "predicted": run.predicted,
        }
    return out


def simulate_optimizer(
    store: CallStore,
    filters: tuple[str, ...],
    records: list[dict],
    gold: dict[str, int],
    *,
    k: int,
    seed: int,
    mode: str,
    guard_rows: int,
    objective: str,
    agreement_threshold: float,
) -> dict[str, Any]:
    """Replay one `jev-optimize` run (execute-chosen mode) end to end."""
    sample = draw_sample(records, k, seed)
    s_ids = [r["id"] for r in sample]
    rest = [r["id"] for r in records if r["id"] not in set(s_ids)]
    independent, combined = Plan("independent", filters), Plan("combined", filters)
    if mode == "full":
        ind, comb = (
            run_plan(store, independent, s_ids),
            run_plan(store, combined, s_ids),
        )
        guard_ids = s_ids
        matrix = ind.filter_decisions
        sampling_calls = ind.calls + comb.calls
    elif mode == "cheap":
        guard_ids = [r["id"] for r in draw_sample(sample, guard_rows, seed)]
        comb = run_plan(store, combined, s_ids)
        ind = run_plan(store, independent, guard_ids)
        matrix = comb.filter_decisions
        sampling_calls = comb.calls + ind.calls
    else:
        raise ValueError(f"Unknown sample mode {mode!r}")
    estimates = build_estimates(
        filters,
        matrix=matrix,
        single_calls={f: [store.single[(i, f)] for i in guard_ids] for f in filters},
        combined_calls=comb.calls,
        guard_pairs=[
            (comb.filter_decisions[i], ind.filter_decisions[i]) for i in guard_ids
        ],
        mode=mode,
    )
    choice = choose_plan(
        estimates, objective=objective, agreement_threshold=agreement_threshold
    )
    chosen = plan_from_choice(choice)

    fill_calls: list[dict] = []
    if mode == "full":
        answers = {i: ind.predicted[i] for i in s_ids}
    elif chosen.strategy == "combined":
        answers = {i: comb.predicted[i] for i in s_ids}
    else:
        answers = {i: ind.predicted[i] for i in guard_ids}
        fill_ids = [i for i in s_ids if i not in set(guard_ids)]
        fill = run_plan(store, chosen, fill_ids)
        answers.update(fill.predicted)
        fill_calls = fill.calls
    execution = run_plan(store, chosen, rest)
    answers.update(execution.predicted)

    predicted = next(
        c["predicted_cost_per_row"]
        for c in choice["candidates"]
        if c["plan"] == chosen.name
    )
    exec_totals = totals(execution.calls)
    actual = (
        exec_totals[objective] / len(rest)
        if rest and exec_totals[objective] is not None
        else None
    )
    return {
        "chosen": chosen.name,
        "chosen_strategy": chosen.strategy,
        "combined_agreement": estimates["combined"]["agreement_with_independent"],
        "combined_allowed": next(
            c["allowed"] for c in choice["candidates"] if c["strategy"] == "combined"
        ),
        "sampling": totals(sampling_calls),
        "sample_fill": totals(fill_calls),
        "execution": exec_totals,
        "total": totals(sampling_calls + fill_calls + execution.calls),
        "predicted_cost_per_row": predicted,
        "actual_cost_per_row": actual,
        "quality": quality(gold, answers),
        "answers": answers,
        "choice": choice,
    }


def _ratio(a: float | None, b: float | None) -> float | None:
    return a / b if a is not None and b else None


def summarize_runs(
    runs: list[dict], base: dict[str, dict], filters: tuple[str, ...], objective: str
) -> dict[str, Any]:
    """Aggregate optimizer replays for one query, mode, and k across seeds."""
    plan_costs = {
        name: b["totals"][objective]
        for name, b in base.items()
        if b["strategy"] != "independent"
    }
    sequential = {n: v for n, v in plan_costs.items() if n.startswith("sequential:")}
    oracle = min(plan_costs, key=plan_costs.get)
    combined = Plan("combined", filters).name
    best_seq = min(sequential, key=sequential.get)
    worst_seq = max(sequential, key=sequential.get)
    separate = Plan("independent", filters).name
    separate_cost = base[separate]["totals"][objective]
    independent = base[separate]["predicted"]

    def stat(values: list[float | None]) -> dict[str, float | None]:
        vals = [v for v in values if v is not None]
        if not vals:
            return {"mean": None, "min": None, "max": None}
        return {"mean": mean(vals), "min": min(vals), "max": max(vals)}

    totals_ = [r["total"][objective] for r in runs]
    return {
        "n_seeds": len(runs),
        "oracle_plan": oracle,
        "chosen_counts": {
            p: sum(r["chosen"] == p for r in runs)
            for p in sorted({r["chosen"] for r in runs})
        },
        "picked_oracle_plan": sum(r["chosen"] == oracle for r in runs) / len(runs),
        "plan_regret": stat(
            [_ratio(plan_costs[r["chosen"]], plan_costs[oracle]) - 1 for r in runs]
        ),
        "total_cost": stat(totals_),
        "sampling_cost": stat([r["sampling"][objective] for r in runs]),
        "sample_fill_cost": stat([r["sample_fill"][objective] for r in runs]),
        "ratio_vs_always_combined": stat(
            [_ratio(t, plan_costs[combined]) for t in totals_]
        ),
        "ratio_vs_always_separate": stat([_ratio(t, separate_cost) for t in totals_]),
        "ratio_vs_best_fixed_order": stat(
            [_ratio(t, plan_costs[best_seq]) for t in totals_]
        ),
        "ratio_vs_worst_fixed_order": stat(
            [_ratio(t, plan_costs[worst_seq]) for t in totals_]
        ),
        "ratio_vs_oracle": stat([_ratio(t, plan_costs[oracle]) for t in totals_]),
        "relative_prediction_error": stat(
            [
                (r["predicted_cost_per_row"] - r["actual_cost_per_row"])
                / r["actual_cost_per_row"]
                if r["predicted_cost_per_row"] is not None and r["actual_cost_per_row"]
                else None
                for r in runs
            ]
        ),
        "guard_rejected_combined": sum(not r["combined_allowed"] for r in runs)
        / len(runs),
        "f1": stat([r["quality"]["f1"] for r in runs]),
        "agreement_with_independent_answers": stat(
            [
                sum(r["answers"][i] == independent[i] for i in independent)
                / len(independent)
                for r in runs
            ]
        ),
        "reference_plans": {
            "always_combined": combined,
            "always_separate": separate,
            "best_fixed_order": best_seq,
            "worst_fixed_order": worst_seq,
        },
    }


def project(
    runs: list[dict],
    base: dict[str, dict],
    filters: tuple[str, ...],
    objective: str,
    sizes: list[int],
    k: int,
) -> dict[str, Any]:
    """Project optimizer vs fixed strategies to larger inputs.

    Assumes new rows cost what the recorded rows cost on average. The optimizer
    pays its mean sampling (and sample-fill) cost once, then its chosen plan's
    full-data mean cost per row on the remaining N - k rows, averaged over seeds.
    """
    per_row = {
        name: b["per_row"][objective]
        for name, b in base.items()
        if b["strategy"] != "independent"
    }
    separate_per_row = base[Plan("independent", filters).name]["per_row"][objective]
    overhead = mean(
        r["sampling"][objective] + r["sample_fill"][objective] for r in runs
    )
    chosen_per_row = mean(per_row[r["chosen"]] for r in runs)
    sequential = {n: v for n, v in per_row.items() if n.startswith("sequential:")}
    refs = {
        "always_combined": per_row[Plan("combined", filters).name],
        "always_separate": separate_per_row,
        "best_fixed_order": min(sequential.values()),
        "worst_fixed_order": max(sequential.values()),
        "oracle": min(per_row.values()),
    }
    out: dict[str, Any] = {"overhead": overhead, "chosen_cost_per_row": chosen_per_row}
    for name, c in refs.items():
        extra = (
            overhead - k * chosen_per_row
        )  # fixed cost beyond answering the sample rows
        saving = c - chosen_per_row
        out[f"break_even_rows_vs_{name}"] = (
            0 if extra <= 0 else (extra / saving if saving > 0 else math.inf)
        )
        for n in sizes:
            optimizer = overhead + max(n - k, 0) * chosen_per_row
            out[f"ratio_vs_{name}@{n}"] = optimizer / (n * c)
    return out


__all__ = [
    "DECISION_FIELDS",
    "CallStore",
    "baselines",
    "candidate_plans",
    "check_repeats",
    "project",
    "query_key",
    "run_plan",
    "simulate_optimizer",
    "summarize_runs",
    "totals",
]
