"""Compare fixed semantic AND-query plans with explicit dataset and pricing provenance."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from jev_milestone_1.common import canonical_json, digest_file, read_jsonl, write_json
from jev_milestone_1.config import FILTER_PROMPTS, MODEL, THRESHOLD
from .costs import QuestionPricing
from .datasets import check_sample_manifest, load_task, task_prompts
from .experiment import Plan, Pricing, run_experiment, summarize_experiment, validate


def write_summary_csv(path: Path, summary: dict) -> None:
    fields = [
        "plan",
        "n",
        "tp",
        "fp",
        "fn",
        "tn",
        "precision",
        "recall",
        "f1",
        "gold_positive_count",
        "model_calls",
        "semantic_questions",
        "runtime_seconds",
        "input_tokens",
        "output_tokens",
        "estimated_credits",
        "estimated_cost_usd",
        "repeat_consistency",
    ]
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, stats in summary["plans"].items():
            first = next(r for r in stats["repeats"] if r["repeat"] == 1)
            values = {**stats, **first["quality"], "plan": name}
            writer.writerow({key: values.get(key) for key in fields})


def main(argv: Sequence[str] | None = None, *, comparison_only: bool = False) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/sample.jsonl"))
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--output-dir",
        type=Path,
        help="New directory; existing directories are never overwritten",
    )
    output.add_argument(
        "--output",
        type=Path,
        help="New result JSONL base path (overlay-compatible file naming)",
    )
    parser.add_argument(
        "--task",
        type=Path,
        help="Verified task definition for SemBench-derived or other labeled text tasks",
    )
    parser.add_argument(
        "--filters",
        nargs="+",
        help="Selected predicate names; defaults to task predicates or gratitude admiration",
    )
    parser.add_argument(
        "--order",
        nargs="+",
        action="append",
        help="Fixed sequential order; repeat flag for multiple orders. Default: forward and reverse",
    )
    parser.add_argument("--max-records", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1 if comparison_only else 2)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--max-api-calls",
        type=int,
        default=100 if comparison_only else None,
        help="Reject experiments whose worst-case call count exceeds this budget",
    )
    parser.add_argument("--input-usd-per-million", type=float)
    parser.add_argument("--output-usd-per-million", type=float)
    parser.add_argument("--credits-per-question-per-block", type=float)
    parser.add_argument("--bytes-per-block", type=int, default=4096)
    parser.add_argument("--usd-per-credit", type=float)
    parser.add_argument(
        "--pricing-source",
        help="Verified rate source/date or account contract used for the estimate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and show request bounds without API access or output files",
    )
    args = parser.parse_args(argv)
    if args.max_records < 1 or (
        args.max_api_calls is not None and args.max_api_calls < 1
    ):
        parser.error("--max-records and --max-api-calls must be positive")
    rates = (args.input_usd_per_million, args.output_usd_per_million)
    if (rates[0] is None) != (rates[1] is None):
        parser.error("Supply both input and output token rates, or neither")
    if args.usd_per_credit is not None and args.credits_per_question_per_block is None:
        parser.error("--usd-per-credit requires --credits-per-question-per-block")
    if rates[0] is not None and args.credits_per_question_per_block is not None:
        parser.error("Choose token pricing or question/block pricing, not both")
    if args.bytes_per_block < 1:
        parser.error("--bytes-per-block must be positive")
    try:
        pricing = Pricing(*rates) if rates[0] is not None else None
        if args.credits_per_question_per_block is not None:
            pricing = QuestionPricing(
                args.credits_per_question_per_block,
                args.bytes_per_block,
                args.usd_per_credit,
            )
        task = load_task(args.task) if args.task else None
        prompts = task_prompts(task) if task else FILTER_PROMPTS
        filters = tuple(
            args.filters
            if args.filters is not None
            else (list(prompts) if task else ["gratitude", "admiration"])
        )
        orders = args.order if args.order is not None else [filters, filters[::-1]]
        plans = ([] if comparison_only else [Plan("independent", filters)]) + [
            Plan("combined", filters)
        ]
        plans.extend(Plan("sequential", tuple(order)) for order in orders)
        source_manifest = check_sample_manifest(args.input, task)
        records = read_jsonl(args.input)[: args.max_records]
        validate(records, plans, args.repeats, args.threshold, prompts)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    n = len(records) * args.repeats
    bounds = [
        {
            "plan": plan.name,
            "min_calls": n if plan.strategy != "independent" else n * len(filters),
            "max_calls": n if plan.strategy == "combined" else n * len(filters),
        }
        for plan in plans
    ]
    worst_case = sum(bound["max_calls"] for bound in bounds)
    if args.max_api_calls is not None and worst_case > args.max_api_calls:
        parser.error(
            f"Worst case {worst_case} calls exceeds --max-api-calls {args.max_api_calls}; raise budget explicitly"
        )
    print(
        json.dumps(
            {"records": len(records), "repeats": args.repeats, "call_bounds": bounds},
            indent=2,
        )
    )
    if args.dry_run:
        return

    if args.output:
        if args.output.suffix != ".jsonl":
            parser.error("--output must end in .jsonl")
        paths = {
            "decisions": args.output,
            "calls": args.output.with_suffix(".calls.jsonl"),
            "runs": args.output.with_suffix(".runs.jsonl"),
            "manifest": args.output.with_suffix(".manifest.json"),
            "summary": args.output.with_suffix(".summary.json"),
            "csv": args.output.with_suffix(".summary.csv"),
        }
    else:
        args.output_dir = args.output_dir or Path(
            "results/milestone2-compare" if comparison_only else "results/milestone2"
        )
        if args.output_dir.exists():
            parser.error(
                f"Output directory already exists: {args.output_dir}; choose a new --output-dir"
            )
        paths = {
            kind: args.output_dir / f"{kind}.jsonl"
            for kind in ("calls", "decisions", "runs")
        }
        paths.update(
            manifest=args.output_dir / "manifest.json",
            summary=args.output_dir / "summary.json",
            csv=args.output_dir / "summary.csv",
        )
    if any(path.exists() for path in paths.values()):
        parser.error("An output file already exists; choose a new output path")

    from dotenv import load_dotenv
    from typesafe_sdk import RetryPolicy, TypeSafeClient

    load_dotenv()
    if not os.getenv("TYPESAFE_API_KEY"):
        parser.error("TYPESAFE_API_KEY missing; export it or put it in .env")
    manifest = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "requested_model": args.model,
        "filters": {name: prompts[name] for name in filters},
        "plans": [
            {"name": p.name, "strategy": p.strategy, "filter_order": list(p.filters)}
            for p in plans
        ],
        "semantics": "AND of thresholded per-filter probabilities; combined uses multiple Noul questions per record",
        "threshold": args.threshold,
        "repeats": args.repeats,
        "n_records": len(records),
        "input_file": str(args.input),
        "source_sample_sha256": digest_file(args.input),
        "input_manifest": source_manifest,
        "task": task,
        "task_sha256": digest_file(args.task) if args.task else None,
        "selected_records_sha256": hashlib.sha256(
            canonical_json(records).encode("utf-8")
        ).hexdigest(),
        "selected_ids": [r["id"] for r in records],
        "pricing": asdict(pricing) if pricing else None,
        "pricing_source": args.pricing_source,
        "pricing_basis": "question_request_blocks"
        if isinstance(pricing, QuestionPricing)
        else "tokens"
        if pricing
        else None,
        "cost_note": "Estimate, not invoiced cost; request blocks use local canonical JSON, not measured SDK wire bytes; null = unknown",
        "requested_call_budget": args.max_api_calls,
        "worst_case_model_calls": worst_case,
        "execution": "Serial, fresh requests, no application cache; plan order rotates each repeat",
        "sdk_max_retries": 0,
        "call_definition": "One system_one invocation; SDK retries disabled",
        "typesafe_sdk_version": importlib.metadata.version("typesafe-sdk"),
        "python": sys.version.split()[0],
    }
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = paths["manifest"]
    with manifest_path.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(manifest) + "\n")
    try:
        with ExitStack() as stack:
            streams = {
                kind: stack.enter_context(paths[kind].open("x", encoding="utf-8"))
                for kind in ("calls", "decisions", "runs")
            }

            def save_event(kind: str, row: dict) -> None:
                streams[kind].write(canonical_json(row) + "\n")
                streams[kind].flush()
                if kind == "runs":
                    print(
                        f"repeat={row['repeat']} plan={row['plan']} calls={row['model_calls']} runtime={row['runtime_seconds']:.3f}s",
                        flush=True,
                    )

            client = stack.enter_context(
                TypeSafeClient(model=args.model, retry=RetryPolicy(max_retries=0))
            )
            result = run_experiment(
                records,
                plans,
                client=client,
                repeats=args.repeats,
                threshold=args.threshold,
                pricing=pricing,
                on_event=save_event,
                prompts=prompts,
                requested_model=args.model,
            )
        summary = summarize_experiment(result)
        write_json(paths["summary"], summary)
        write_summary_csv(paths["csv"], summary)
        manifest.update(
            status="complete",
            resolved_models=sorted({c["model"] for c in result["calls"]}),
            result_sha256={
                paths[kind].name: digest_file(paths[kind]) for kind in result
            },
            summary_sha256=digest_file(paths["summary"]),
            csv_sha256=digest_file(paths["csv"]),
        )
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
    print(f"Saved {summary['total_model_calls']} model calls to {paths['decisions']}")


if __name__ == "__main__":
    main()
