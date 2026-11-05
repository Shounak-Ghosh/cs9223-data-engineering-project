"""Execute and measure independent, combined, and fixed-order sequential plans."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

from jev_milestone_1.common import binary_metrics
from jev_milestone_1.config import FILTER_PROMPTS, MODEL
from .costs import QuestionPricing


@dataclass(frozen=True)
class Plan:
    strategy: str
    filters: tuple[str, ...]

    @property
    def name(self) -> str:
        return f"{self.strategy}:{','.join(self.filters)}"


@dataclass(frozen=True)
class Pricing:
    """Explicit USD per million token rates; absent rates mean unknown cost."""

    input_per_million: float
    output_per_million: float

    def __post_init__(self) -> None:
        for rate in (self.input_per_million, self.output_per_million):
            if not math.isfinite(rate) or rate < 0:
                raise ValueError("Token prices must be finite and nonnegative")

    def estimate(
        self, input_tokens: int | None, output_tokens: int | None
    ) -> float | None:
        if input_tokens is None or output_tokens is None:
            return None
        return (
            input_tokens * self.input_per_million
            + output_tokens * self.output_per_million
        ) / 1_000_000


def validate(
    records: list[dict],
    plans: list[Plan],
    repeats: int,
    threshold: float,
    prompts: dict[str, str] | None = None,
) -> None:
    """Reject invalid experiments before any billable calls."""
    prompts = FILTER_PROMPTS if prompts is None else prompts
    if repeats < 1 or not 0 <= threshold <= 1:
        raise ValueError(
            "repeats must be positive and threshold must be between 0 and 1"
        )
    if not records or not plans:
        raise ValueError("At least one record and one plan are required")
    if len({p.name for p in plans}) != len(plans):
        raise ValueError("Duplicate execution plan")
    expected = set(plans[0].filters)
    for plan in plans:
        if plan.strategy not in {"independent", "combined", "sequential"}:
            raise ValueError(f"Unknown strategy: {plan.strategy}")
        if len(plan.filters) < 2 or len(set(plan.filters)) != len(plan.filters):
            raise ValueError("Plans require at least two distinct filters")
        if set(plan.filters) != expected or not expected <= prompts.keys():
            raise ValueError("All plans must use the same known filters")
        if any(
            not isinstance(prompts[name], str) or not prompts[name].strip()
            for name in expected
        ):
            raise ValueError("Every selected filter needs a nonempty prompt")
    ids = []
    for record in records:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("id"), str)
            or not record["id"]
            or not isinstance(record.get("text"), str)
        ):
            raise ValueError("Each record needs a nonempty string id and string text")
        ids.append(record["id"])
        gold = record.get("gold")
        if not isinstance(gold, dict) or any(
            type(gold.get(name)) not in (int, bool) or gold[name] not in (0, 1)
            for name in expected
        ):
            raise ValueError(
                "Each record needs binary gold labels for every selected filter"
            )
    if len(set(ids)) != len(ids):
        raise ValueError("Record ids must be unique")


def _token_count(usage: Any, name: str) -> int | None:
    count = getattr(usage, name, None)
    if count is None:
        return None
    if not isinstance(count, int) or count < 0:
        raise ValueError(f"Invalid {name} usage: {count!r}")
    return count


def run_experiment(
    records: list[dict],
    plans: list[Plan],
    *,
    client: Any,
    repeats: int = 1,
    threshold: float = 0.5,
    pricing: Pricing | QuestionPricing | None = None,
    on_event: Callable[[str, dict], None] | None = None,
    prompts: dict[str, str] | None = None,
    requested_model: str = MODEL,
) -> dict[str, list[dict]]:
    """Fresh, serial requests for every plan/repeat; no result reuse or caching.

    Calls store request-level usage exactly once, even with multiple questions.
    Decisions cover every input row; skipped sequential predicates remain absent.
    """
    prompts = FILTER_PROMPTS if prompts is None else prompts
    validate(records, plans, repeats, threshold, prompts)
    from typesafe_sdk import Noul, TypeSafeError

    calls, decisions, runs = [], [], []
    for repeat in range(1, repeats + 1):
        # Rotate plan order across repeats to reduce systematic timing-order bias.
        offset = (repeat - 1) % len(plans)
        for plan in plans[offset:] + plans[:offset]:
            run_start = time.perf_counter()
            call_start, decision_start = len(calls), len(decisions)
            stage_input = {name: 0 for name in plan.filters}
            stage_pass = {name: 0 for name in plan.filters}
            for record in records:
                record_start = time.perf_counter()
                record_call_start = len(calls)
                probabilities: dict[str, float] = {}
                groups = (
                    [plan.filters]
                    if plan.strategy == "combined"
                    else [(name,) for name in plan.filters]
                )
                for group in groups:
                    questions = {
                        name: Noul(instructions=prompts[name]) for name in group
                    }
                    start = time.perf_counter()
                    try:
                        response = client.system_one(
                            state=record["text"], questions=questions
                        )
                        elapsed = time.perf_counter() - start
                        answers = {
                            name: float(response.nouls[name].noul) for name in group
                        }
                        if any(
                            not math.isfinite(p) or not 0 <= p <= 1
                            for p in answers.values()
                        ):
                            raise ValueError(
                                "Model returned a probability outside [0, 1]"
                            )
                        input_tokens = _token_count(
                            getattr(response, "usage", None), "input_tokens"
                        )
                        output_tokens = _token_count(
                            getattr(response, "usage", None), "output_tokens"
                        )
                        try:
                            request_id = getattr(response, "request_id", None)
                        except TypeSafeError:
                            request_id = None
                    except Exception as error:
                        event = {
                            "call_index": len(calls) + 1,
                            "plan": plan.name,
                            "repeat": repeat,
                            "id": record["id"],
                            "filters": list(group),
                            "status": "error",
                            "error_type": type(error).__name__,
                            "latency_seconds": time.perf_counter() - start,
                            "input_tokens": None,
                            "output_tokens": None,
                            "estimated_credits": None,
                            "estimated_cost_usd": None,
                        }
                        calls.append(event)
                        if on_event:
                            on_event("calls", event)
                        raise
                    costs = {
                        "estimated_credits": None,
                        "estimated_cost_usd": pricing.estimate(
                            input_tokens, output_tokens
                        )
                        if isinstance(pricing, Pricing)
                        else None,
                    }
                    if isinstance(pricing, QuestionPricing):
                        costs = pricing.estimate_request(
                            state=record["text"],
                            names=group,
                            prompts=prompts,
                            model=requested_model,
                        )
                    calls.append(
                        {
                            "call_index": len(calls) + 1,
                            "plan": plan.name,
                            "repeat": repeat,
                            "status": "ok",
                            "id": record["id"],
                            "filters": list(group),
                            "probabilities": answers,
                            "model": response.model,
                            "request_id": request_id,
                            "latency_seconds": elapsed,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            **costs,
                        }
                    )
                    if on_event:
                        on_event("calls", calls[-1])
                    probabilities.update(answers)
                    for name, probability in answers.items():
                        stage_input[name] += 1
                        stage_pass[name] += int(probability >= threshold)
                    if plan.strategy == "sequential" and any(
                        p < threshold for p in answers.values()
                    ):
                        break
                record_calls = calls[record_call_start:]
                decisions.append(
                    {
                        "plan": plan.name,
                        "repeat": repeat,
                        "id": record["id"],
                        "gold": int(all(record["gold"][name] for name in plan.filters)),
                        "predicted": int(
                            all(p >= threshold for p in probabilities.values())
                        ),
                        "probabilities": probabilities,
                        "skipped_filters": [
                            name for name in plan.filters if name not in probabilities
                        ],
                        "filter_decisions": {
                            name: int(probabilities[name] >= threshold)
                            if name in probabilities
                            else None
                            for name in plan.filters
                        },
                        "model_calls": len(record_calls),
                        "runtime_seconds": time.perf_counter() - record_start,
                        "api_latency_seconds": sum(
                            c["latency_seconds"] for c in record_calls
                        ),
                        "input_tokens": _complete_sum(record_calls, "input_tokens"),
                        "output_tokens": _complete_sum(record_calls, "output_tokens"),
                        "estimated_credits": _complete_sum(
                            record_calls, "estimated_credits"
                        ),
                        "estimated_cost_usd": _complete_sum(
                            record_calls, "estimated_cost_usd"
                        ),
                    }
                )
                if on_event:
                    on_event("decisions", decisions[-1])
            runs.append(
                {
                    "plan": plan.name,
                    "strategy": plan.strategy,
                    "filter_order": list(plan.filters),
                    "repeat": repeat,
                    "execution_position": len(runs) % len(plans) + 1,
                    "runtime_seconds": time.perf_counter() - run_start,
                    "model_calls": len(calls) - call_start,
                    "n_records": len(decisions) - decision_start,
                    "stages": [
                        {
                            "filter": name,
                            "input_rows": stage_input[name],
                            "passed_rows": stage_pass[name],
                        }
                        for name in plan.filters
                    ],
                }
            )
            if on_event:
                on_event("runs", runs[-1])
    return {"calls": calls, "decisions": decisions, "runs": runs}


def _complete_sum(rows: list[dict], field: str) -> int | float | None:
    values = [row[field] for row in rows]
    return sum(values) if all(value is not None for value in values) else None


def summarize_experiment(result: dict[str, list[dict]]) -> dict:
    """Report conjunction quality over ALL rows, with repeat and plan agreement."""
    summary: dict[str, Any] = {"plans": {}}
    baseline = next(
        (r["plan"] for r in result["runs"] if r["strategy"] == "independent"), None
    )
    baseline_decisions = {
        (d["repeat"], d["id"]): d["predicted"]
        for d in result["decisions"]
        if d["plan"] == baseline
    }
    for name in dict.fromkeys(r["plan"] for r in result["runs"]):
        runs = [r for r in result["runs"] if r["plan"] == name]
        calls = [c for c in result["calls"] if c["plan"] == name]
        decisions = [d for d in result["decisions"] if d["plan"] == name]
        by_repeat = []
        for run in runs:
            rows = [d for d in decisions if d["repeat"] == run["repeat"]]
            repeat_calls = [c for c in calls if c["repeat"] == run["repeat"]]
            metrics = binary_metrics(
                [d["gold"] for d in rows], [d["predicted"] for d in rows]
            )
            metrics["accuracy"] = (metrics["tp"] + metrics["tn"]) / metrics["n"]
            metrics["gold_positive_count"] = metrics["tp"] + metrics["fn"]
            metrics["quality_note"] = (
                "No gold positives; positive-class recall/F1 are uninformative"
                if not metrics["gold_positive_count"]
                else None
            )
            by_repeat.append(
                {
                    **run,
                    "quality": metrics,
                    "input_tokens": _complete_sum(repeat_calls, "input_tokens"),
                    "output_tokens": _complete_sum(repeat_calls, "output_tokens"),
                    "estimated_credits": _complete_sum(
                        repeat_calls, "estimated_credits"
                    ),
                    "estimated_cost_usd": _complete_sum(
                        repeat_calls, "estimated_cost_usd"
                    ),
                }
            )
        groups: dict[str, list[int]] = {}
        for row in decisions:
            groups.setdefault(row["id"], []).append(row["predicted"])
        compared = [g for g in groups.values() if len(g) > 1]
        agreement = (
            sum(
                d["predicted"] == baseline_decisions[d["repeat"], d["id"]]
                for d in decisions
            )
            / len(decisions)
            if baseline
            else None
        )
        summary["plans"][name] = {
            "model_calls": len(calls),
            "semantic_questions": sum(len(c["filters"]) for c in calls),
            "runtime_seconds": sum(r["runtime_seconds"] for r in runs),
            "api_latency_seconds": sum(c["latency_seconds"] for c in calls),
            "input_tokens": _complete_sum(calls, "input_tokens"),
            "output_tokens": _complete_sum(calls, "output_tokens"),
            "estimated_credits": _complete_sum(calls, "estimated_credits"),
            "estimated_cost_usd": _complete_sum(calls, "estimated_cost_usd"),
            "repeat_consistency": sum(len(set(g)) == 1 for g in compared)
            / len(compared)
            if compared
            else None,
            "agreement_with_independent": agreement,
            "repeats": by_repeat,
        }
    combined = next(
        (r["plan"] for r in result["runs"] if r["strategy"] == "combined"), None
    )
    summary["comparisons_vs_combined"] = {}
    if combined:
        base = summary["plans"][combined]
        base_predictions = {
            d["id"]: d["predicted"]
            for d in result["decisions"]
            if d["plan"] == combined and d["repeat"] == 1
        }
        for name, measured in summary["plans"].items():
            first = [
                d for d in result["decisions"] if d["plan"] == name and d["repeat"] == 1
            ]
            summary["comparisons_vs_combined"][name] = {
                "first_repeat_prediction_disagreements": sum(
                    d["predicted"] != base_predictions[d["id"]] for d in first
                ),
                "model_calls_saved": base["model_calls"] - measured["model_calls"],
                "estimated_credit_savings": base["estimated_credits"]
                - measured["estimated_credits"]
                if base["estimated_credits"] is not None
                and measured["estimated_credits"] is not None
                else None,
            }
    summary["total_model_calls"] = len(result["calls"])
    summary["total_estimated_credits"] = _complete_sum(
        result["calls"], "estimated_credits"
    )
    summary["total_estimated_cost_usd"] = _complete_sum(
        result["calls"], "estimated_cost_usd"
    )
    return summary
