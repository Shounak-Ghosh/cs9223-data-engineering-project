"""Single-filter live runner: one Noul question per call, fresh calls every repeat.

Milestone 2's `run_experiment` rejects single-filter plans by design, so this is a
deliberately small separate runner rather than a loosened Milestone 2 validation.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

from jev_milestone_1.config import MODEL, THRESHOLD
from jev_milestone_2.costs import QuestionPricing
from jev_milestone_2.experiment import Pricing, _token_count


def validate_records(records: list[dict], filter_name: str) -> None:
    """Every record needs a string id and text and a binary gold label for the filter."""
    if not records:
        raise ValueError("No input records")
    ids = set()
    for record in records:
        if not isinstance(record.get("id"), str) or not isinstance(
            record.get("text"), str
        ):
            raise ValueError("Every record needs string 'id' and 'text' fields")
        if record["id"] in ids:
            raise ValueError(f"Duplicate record id {record['id']!r}")
        ids.add(record["id"])
        gold = record.get("gold")
        label = gold.get(filter_name) if isinstance(gold, dict) else None
        if type(label) is not int or label not in (0, 1):
            raise ValueError(
                f"Record {record['id']} lacks a binary gold label for {filter_name!r}"
            )


def run_single_filter(
    records: list[dict],
    *,
    filter_name: str,
    prompt: str,
    client: Any,
    repeats: int = 1,
    threshold: float = THRESHOLD,
    pricing: Pricing | QuestionPricing | None = None,
    model: str = MODEL,
    on_event: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Evaluate one filter on every record, `repeats` times, one question per call.

    Each row is passed to `on_event` as soon as it exists. On failure an error row
    (exception class name only) is emitted and the exception is re-raised.
    """
    validate_records(records, filter_name)
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    from typesafe_sdk import Noul, TypeSafeError

    rows: list[dict] = []

    def emit(row: dict) -> None:
        rows.append(row)
        if on_event:
            on_event(row)

    for repeat in range(1, repeats + 1):
        for record in records:
            base = {
                "call_index": len(rows) + 1,
                "id": record["id"],
                "filter": filter_name,
                "repeat": repeat,
                "gold": record["gold"][filter_name],
                "threshold": threshold,
            }
            questions = {filter_name: Noul(instructions=prompt)}
            start = time.perf_counter()
            try:
                response = client.system_one(state=record["text"], questions=questions)
                latency_ms = (time.perf_counter() - start) * 1000
                probability = float(response.nouls[filter_name].noul)
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError("Model returned a probability outside [0, 1]")
                usage = getattr(response, "usage", None)
                input_tokens = _token_count(usage, "input_tokens")
                output_tokens = _token_count(usage, "output_tokens")
                try:
                    request_id = getattr(response, "request_id", None)
                except TypeSafeError:
                    request_id = None
                resolved_model = response.model
            except Exception as error:
                emit(
                    {
                        **base,
                        "status": "error",
                        "error_type": type(error).__name__,
                        "probability": None,
                        "predicted": None,
                        "model": None,
                        "request_id": None,
                        "latency_ms": (time.perf_counter() - start) * 1000,
                        "input_tokens": None,
                        "output_tokens": None,
                        "estimated_credits": None,
                        "estimated_cost_usd": None,
                    }
                )
                raise
            costs: dict[str, Any] = {
                "estimated_credits": None,
                "estimated_cost_usd": pricing.estimate(input_tokens, output_tokens)
                if isinstance(pricing, Pricing)
                else None,
            }
            if isinstance(pricing, QuestionPricing):
                estimate = pricing.estimate_request(
                    state=record["text"],
                    names=(filter_name,),
                    prompts={filter_name: prompt},
                    model=model,
                )
                costs = {
                    "estimated_credits": estimate["estimated_credits"],
                    "estimated_cost_usd": estimate["estimated_cost_usd"],
                }
            emit(
                {
                    **base,
                    "status": "ok",
                    "probability": probability,
                    "predicted": int(probability >= threshold),
                    "model": resolved_model,
                    "request_id": request_id,
                    "latency_ms": latency_ms,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    **costs,
                }
            )
    return rows
