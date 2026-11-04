"""Evaluate ONE Noul per record per request, preserving independent-filter measurement."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .common import canonical_json, digest_file, read_jsonl, summarize, write_json
from .config import FILTER_PROMPTS, MODEL, THRESHOLD


def run_individual_filters(
    records: list[dict], filters: list[str], repeats: int, threshold: float, client: Any
) -> list[dict]:
    """A single semantic question per API request; no caching or batching."""
    from typesafe_sdk import Noul

    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    output: list[dict] = []
    for repeat in range(1, repeats + 1):
        for record in records:
            for name in filters:
                start = time.perf_counter()
                response = client.system_one(
                    state=record["text"],
                    questions={name: Noul(instructions=FILTER_PROMPTS[name])},
                )
                elapsed = (time.perf_counter() - start) * 1000
                probability = float(response.nouls[name].noul)
                output.append(
                    {
                        "id": record["id"],
                        "filter": name,
                        "repeat": repeat,
                        "gold": int(record["gold"][name]),
                        "probability": probability,
                        "predicted": int(probability >= threshold),
                        "threshold": threshold,
                        "model": response.model,
                        "latency_ms": round(elapsed, 3),
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        "request_id": getattr(response, "request_id", None),
                    }
                )
                print(
                    f"repeat={repeat} id={record['id']} filter={name} p={probability:.3f}",
                    flush=True,
                )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/sample.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/pilot.jsonl"))
    parser.add_argument(
        "--filters",
        nargs="+",
        default=["gratitude", "admiration"],
        choices=sorted(FILTER_PROMPTS),
    )
    parser.add_argument(
        "--max-records", type=int, default=10, help="First n fixed records from sample"
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--overwrite", action="store_true", help="Permit replacing an existing output"
    )
    args = parser.parse_args()

    from dotenv import load_dotenv
    from typesafe_sdk import TypeSafeClient

    load_dotenv()
    if not os.getenv("TYPESAFE_API_KEY"):
        parser.error("TYPESAFE_API_KEY missing; export it or put it in .env")
    if not args.input.exists():
        parser.error(f"Missing {args.input}; run uv run jev-prepare first")
    if args.max_records < 1 or args.repeats < 1:
        parser.error("--max-records and --repeats must be positive")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if args.output.exists() and not args.overwrite:
        parser.error(
            f"Output {args.output} already exists; use another --output or --overwrite"
        )
    records = read_jsonl(args.input)[: args.max_records]
    if not records:
        parser.error("Empty input sample")
    with TypeSafeClient(model=args.model) as client:
        output = run_individual_filters(
            records, args.filters, args.repeats, args.threshold, client
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in output:
            stream.write(canonical_json(row) + "\n")
    meta = {
        "requested_model": args.model,
        "resolved_models": sorted(set(x["model"] for x in output)),
        "filters": {key: FILTER_PROMPTS[key] for key in args.filters},
        "threshold": args.threshold,
        "repeats": args.repeats,
        "n_records": len(records),
        "source_sample_sha256": digest_file(args.input),
        "input_file": str(args.input),
        "typesafe_sdk_version": importlib.metadata.version("typesafe-sdk"),
        "python": sys.version.split()[0],
        "execution": "One filter per API request. Serial evaluation. No application-level caching.",
        "result_file_sha256": digest_file(args.output),
    }
    write_json(args.output.with_suffix(".manifest.json"), meta)
    summary = summarize(output, args.filters, args.threshold)
    write_json(args.output.with_suffix(".summary.json"), summary)
    print("\nSummary (first repeat = quality; repeated calls = stability):")
    for name, m in summary["filters"].items():
        print(
            f"  {name}: gold selectivity={m['gold_selectivity']:.3f}, "
            f"pred selectivity={m['predicted_selectivity']:.3f}, "
            f"P={m['precision']:.3f}, R={m['recall']:.3f}, F1={m['f1']:.3f}, "
            f"repeat agreement={m['repeat_consistency']}"
        )
    print(f"Saved {len(output)} API results to {args.output}")


if __name__ == "__main__":
    main()
