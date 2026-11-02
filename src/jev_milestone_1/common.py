"""Minimal serialization and metrics utilities (no ML dependencies)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def binary_metrics(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    if len(labels) != len(predictions):
        raise ValueError("Mismatched number of labels and predictions")
    tp = sum(a == 1 and b == 1 for a, b in zip(labels, predictions))
    fp = sum(a == 0 and b == 1 for a, b in zip(labels, predictions))
    fn = sum(a == 1 and b == 0 for a, b in zip(labels, predictions))
    tn = sum(a == 0 and b == 0 for a, b in zip(labels, predictions))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "n": len(labels), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": p, "recall": r, "f1": f1,
        "gold_selectivity": sum(labels) / len(labels) if labels else 0.0,
        "predicted_selectivity": sum(predictions) / len(predictions) if predictions else 0.0,
    }


def summarize(rows: list[dict[str, Any]], filters: list[str], threshold: float) -> dict[str, Any]:
    """Summarize first-pass quality and repeat agreement per filter."""
    summary: dict[str, Any] = {"threshold": threshold, "filters": {}}
    for name in filters:
        records = [r for r in rows if r["filter"] == name]
        first = [r for r in records if r["repeat"] == 1]
        first.sort(key=lambda r: r["id"])
        metric = binary_metrics([int(r["gold"]) for r in first], [int(r["predicted"]) for r in first])
        by_id: dict[str, list[dict[str, Any]]] = {}
        for r in records:
            by_id.setdefault(r["id"], []).append(r)
        comparisons = [sorted(r, key=lambda x: x["repeat"]) for r in by_id.values() if len(r) >= 2]
        metric["repeat_consistency"] = (
            sum(len({int(x["predicted"]) for x in group}) == 1 for group in comparisons) / len(comparisons)
            if comparisons else None
        )
        metric["max_repeat_probability_range"] = (
            max(max(x["probability"] for x in group) - min(x["probability"] for x in group) for group in comparisons)
            if comparisons else None
        )
        metric["api_calls"] = len(records)
        metric["input_tokens"] = sum(r.get("input_tokens") or 0 for r in records)
        metric["total_api_latency_seconds"] = round(sum(r["latency_ms"] for r in records) / 1000, 3)
        summary["filters"][name] = metric
    summary["total_api_calls"] = len(rows)
    summary["total_input_tokens"] = sum(r.get("input_tokens") or 0 for r in rows)
    return summary
