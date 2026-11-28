"""Record the Jev calls needed to replay every plan for a set of AND queries.

For N rows, a union of F filters and Q queries this makes N × F one-question calls
(the independent plan over the union) plus N × Q combined calls, once. Every
sequential order, the combined plan, and the optimizer can then be replayed
offline with `jev-benchmark`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from jev_milestone_1.common import canonical_json, digest_file, read_jsonl, write_json
from jev_milestone_1.config import FILTER_PROMPTS, MODEL, THRESHOLD
from jev_milestone_2.datasets import check_sample_manifest, load_task, task_prompts
from jev_milestone_2.experiment import Plan, Pricing, run_experiment, validate


def parse_query(text: str) -> tuple[str, ...]:
    filters = tuple(f.strip() for f in text.split(",") if f.strip())
    if len(filters) < 2 or len(set(filters)) != len(filters):
        raise ValueError(f"Query {text!r} needs at least two distinct filters")
    return filters


def collection_plans(queries: list[tuple[str, ...]]) -> list[tuple[str, Plan]]:
    union = tuple(dict.fromkeys(f for q in queries for f in q))
    plans = [("independent", Plan("independent", union))]
    plans += [(f"combined-{'-'.join(q)}", Plan("combined", q)) for q in queries]
    return plans


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/sample.jsonl"))
    parser.add_argument("--task", type=Path)
    parser.add_argument(
        "--query",
        action="append",
        required=True,
        help="Comma-separated filters of one AND query; repeat for more queries",
    )
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--input-usd-per-million", type=float)
    parser.add_argument("--output-usd-per-million", type=float)
    parser.add_argument("--pricing-source")
    parser.add_argument("--max-api-calls", type=int, default=2000)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/milestone5/collect")
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    rates = (args.input_usd_per_million, args.output_usd_per_million)
    if (rates[0] is None) != (rates[1] is None):
        parser.error("Supply both input and output token rates, or neither")
    try:
        pricing = Pricing(*rates) if rates[0] is not None else None
        task = load_task(args.task) if args.task else None
        prompts = task_prompts(task) if task else FILTER_PROMPTS
        queries = list(dict.fromkeys(parse_query(q) for q in args.query))
        source_manifest = check_sample_manifest(args.input, task)
        records = read_jsonl(args.input)
        if args.max_records is not None:
            if args.max_records < 1:
                raise ValueError("--max-records must be positive")
            records = records[: args.max_records]
        plans = collection_plans(queries)
        for _, plan in plans:
            validate(records, [plan], 1, args.threshold, prompts)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    n_union = len(plans[0][1].filters)
    bounds = {
        "records": len(records),
        "queries": [",".join(q) for q in queries],
        "one_question_calls": len(records) * n_union,
        "combined_calls": len(records) * len(queries),
        "total_calls": len(records) * (n_union + len(queries)),
    }
    print(json.dumps(bounds, indent=2))
    if bounds["total_calls"] > args.max_api_calls:
        parser.error(
            f"{bounds['total_calls']} calls exceeds --max-api-calls {args.max_api_calls}; raise it explicitly"
        )
    if args.dry_run:
        return
    if args.output_dir.exists():
        parser.error(
            f"Output directory already exists: {args.output_dir}; choose a new --output-dir"
        )

    from dotenv import load_dotenv
    from typesafe_sdk import RetryPolicy, TypeSafeClient

    load_dotenv()
    if not os.getenv("TYPESAFE_API_KEY"):
        parser.error("TYPESAFE_API_KEY missing; export it or put it in .env")

    out = args.output_dir
    out.mkdir(parents=True)
    manifest = {
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "requested_model": args.model,
        "threshold": args.threshold,
        "filters": {f: prompts[f] for f in plans[0][1].filters},
        "runs": {name: plan.name for name, plan in plans},
        "input_file": str(args.input),
        "source_sample_sha256": digest_file(args.input),
        "input_manifest": source_manifest,
        "task_sha256": digest_file(args.task) if args.task else None,
        "n_records": len(records),
        "selected_records_sha256": hashlib.sha256(
            canonical_json(records).encode("utf-8")
        ).hexdigest(),
        "pricing": asdict(pricing) if pricing else None,
        "pricing_source": args.pricing_source,
        "call_bounds": bounds,
        "sdk_max_retries": 0,
        "typesafe_sdk_version": importlib.metadata.version("typesafe-sdk"),
        "python": sys.version.split()[0],
    }
    write_json(out / "manifest.json", manifest)
    models: set[str] = set()
    try:
        with TypeSafeClient(
            model=args.model, retry=RetryPolicy(max_retries=0)
        ) as client:
            for name, plan in plans:
                (out / name).mkdir()
                with ExitStack() as files:
                    streams = {
                        kind: files.enter_context(
                            (out / name / f"{kind}.jsonl").open("x", encoding="utf-8")
                        )
                        for kind in ("calls", "decisions", "runs")
                    }

                    def save(kind: str, row: dict, streams=streams) -> None:
                        streams[kind].write(canonical_json(row) + "\n")
                        streams[kind].flush()

                    result = run_experiment(
                        records,
                        [plan],
                        client=client,
                        repeats=1,
                        threshold=args.threshold,
                        pricing=pricing,
                        on_event=save,
                        prompts=prompts,
                        requested_model=args.model,
                    )
                models |= {c["model"] for c in result["calls"] if c.get("model")}
                print(f"[{name}] {plan.name} calls={len(result['calls'])}", flush=True)
        manifest.update(status="complete", resolved_models=sorted(models))
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        write_json(out / "manifest.json", manifest)
    print(f"Saved calls to {out}")


if __name__ == "__main__":
    main()
