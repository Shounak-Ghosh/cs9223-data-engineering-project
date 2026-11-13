"""Milestone 3 command-line entry points.

- `jev-filter-profile`: live single-filter run on all records or a seeded sample.
- `jev-sample-analysis`: offline resampling analysis of a full run (no SDK, no
  credentials, no API calls).
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from jev_milestone_1.common import canonical_json, digest_file, read_jsonl, write_json
from jev_milestone_1.config import MODEL, THRESHOLD
from jev_milestone_2.costs import QuestionPricing
from jev_milestone_2.datasets import check_sample_manifest, load_task, task_prompts
from jev_milestone_2.experiment import Pricing

from .profile import run_single_filter, validate_records
from .resample import (
    DEFAULT_DRAWS,
    DEFAULT_SEED,
    DEFAULT_SIZES,
    ordering_agreement,
    repeat_rows,
    resampling_analysis,
)
from .sampling import compare, draw_sample, estimate_filter

DEFAULT_SAMPLE_SEED = 42


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _project_version() -> str | None:
    try:
        return importlib.metadata.version("de-semantic-optimizer")
    except importlib.metadata.PackageNotFoundError:
        return None


def _write_new_json(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        stream.write("\n")


def _add_pricing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-usd-per-million",
        type=float,
        help="USD per million billable input tokens; verify at docs.typesafe.ai/models",
    )
    parser.add_argument(
        "--output-usd-per-million", type=float, help="USD per million output tokens"
    )
    parser.add_argument(
        "--pricing-source",
        help="Verified rate source/date or account contract used for the estimate",
    )
    parser.add_argument(
        "--credits-per-question-per-block",
        type=float,
        help="Custom-contract estimator only; not TypeSafe Jev token billing",
    )
    parser.add_argument(
        "--bytes-per-block",
        type=int,
        default=4096,
        help="Block size for custom-contract credit estimation only",
    )
    parser.add_argument(
        "--usd-per-credit", type=float, help="Optional custom-contract conversion"
    )


def _pricing_from_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> Pricing | QuestionPricing | None:
    """Same pricing-flag rules as Milestone 2."""
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
        if rates[0] is not None:
            return Pricing(*rates)
        if args.credits_per_question_per_block is not None:
            return QuestionPricing(
                args.credits_per_question_per_block,
                args.bytes_per_block,
                args.usd_per_credit,
            )
    except ValueError as error:
        parser.error(str(error))
    return None


def profile_main(argv: Sequence[str] | None = None) -> None:
    """Profile one semantic filter on all records or a seeded random sample."""
    parser = argparse.ArgumentParser(
        prog="jev-filter-profile", description=profile_main.__doc__
    )
    parser.add_argument(
        "--input", type=Path, default=Path("data/sembench_movie_verified.jsonl")
    )
    parser.add_argument("--task", type=Path, default=Path("data/verified_task.json"))
    parser.add_argument("--filter", required=True, help="Predicate name from the task")
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Records to sample; omit for all records (full-data reference)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help=f"Sample seed (default {DEFAULT_SAMPLE_SEED}); only with --sample-size",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--max-api-calls",
        type=int,
        default=100,
        help="Refuse runs whose call count exceeds this budget",
    )
    _add_pricing_args(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New directory; default results/milestone3/<filter>-full or -k<K>-seed<S>",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned calls and sampled IDs; no credentials, no API calls, no files",
    )
    args = parser.parse_args(argv)

    if args.repeats < 1 or args.max_api_calls < 1:
        parser.error("--repeats and --max-api-calls must be positive")
    if args.seed is not None and args.sample_size is None:
        parser.error("--seed is only used with --sample-size")
    pricing = _pricing_from_args(parser, args)

    # 1. Task and input-manifest validation.
    try:
        task = load_task(args.task)
        prompts = task_prompts(task)
        source_manifest = check_sample_manifest(args.input, task)
        records = read_jsonl(args.input)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    if args.filter not in prompts:
        parser.error(
            f"Unknown --filter {args.filter!r}; task predicates: {', '.join(prompts)}"
        )
    prompt = prompts[args.filter]
    # 2. Binary gold label for the filter on every input record.
    try:
        validate_records(records, args.filter)
    except ValueError as error:
        parser.error(str(error))
    # 3. Sample.
    seed = None
    if args.sample_size is not None:
        seed = DEFAULT_SAMPLE_SEED if args.seed is None else args.seed
        try:
            selected = draw_sample(records, args.sample_size, seed)
        except ValueError as error:
            parser.error(str(error))
    else:
        selected = records
    # 4. Budget.
    planned_calls = len(selected) * args.repeats
    if planned_calls > args.max_api_calls:
        parser.error(
            f"Planned {planned_calls} calls exceeds --max-api-calls {args.max_api_calls}; raise budget explicitly"
        )
    output_dir = args.output_dir or Path("results/milestone3") / (
        f"{args.filter}-full"
        if args.sample_size is None
        else f"{args.filter}-k{args.sample_size}-seed{seed}"
    )
    # 5. Plan.
    print(
        json.dumps(
            {
                "filter": args.filter,
                "prompt": prompt,
                "input_records": len(records),
                "sample_size": args.sample_size
                if args.sample_size is not None
                else "all",
                "seed": seed,
                "repeats": args.repeats,
                "planned_calls": planned_calls,
                "max_api_calls": args.max_api_calls,
                "model": args.model,
                "threshold": args.threshold,
                "output_dir": str(output_dir),
                "selected_ids": [r["id"] for r in selected],
            },
            indent=2,
        ),
        flush=True,
    )
    # 6. Dry run stops before credentials, SDK import, or any file.
    if args.dry_run:
        return
    if output_dir.exists():
        parser.error(
            f"Output directory already exists: {output_dir}; choose a new --output-dir"
        )

    # 7. Credentials and client.
    from dotenv import load_dotenv
    from typesafe_sdk import RetryPolicy, TypeSafeClient

    load_dotenv()
    if not os.getenv("TYPESAFE_API_KEY"):
        parser.error("TYPESAFE_API_KEY missing; export it or put it in .env")

    paths = {
        "calls": output_dir / "calls.jsonl",
        "estimate": output_dir / "estimate.json",
        "manifest": output_dir / "manifest.json",
    }
    manifest = {
        "status": "running",
        "started_at": _now(),
        "kind": "filter_profile",
        "filter": args.filter,
        "prompt": prompt,
        "threshold": args.threshold,
        "requested_model": args.model,
        "repeats": args.repeats,
        "sample_size": args.sample_size if args.sample_size is not None else "all",
        "seed": seed,
        "n_input_records": len(records),
        "n_selected_records": len(selected),
        "selected_ids": [r["id"] for r in selected],
        "planned_calls": planned_calls,
        "max_api_calls": args.max_api_calls,
        "input_file": str(args.input),
        "input_sha256": digest_file(args.input),
        "input_manifest": source_manifest,
        "task_file": str(args.task),
        "task_sha256": digest_file(args.task),
        "pricing": asdict(pricing) if pricing else None,
        "pricing_source": args.pricing_source,
        "pricing_basis": "question_request_blocks"
        if isinstance(pricing, QuestionPricing)
        else "tokens"
        if pricing
        else None,
        "selectivity_definition": "Jev pass rate: fraction of calls with probability >= threshold",
        "execution": "Serial, fresh requests, one Noul question per call, no application cache",
        "sdk_max_retries": 0,
        "typesafe_sdk_version": importlib.metadata.version("typesafe-sdk"),
        "python": sys.version.split()[0],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_new_json(paths["manifest"], manifest)
    rows: list[dict] = []
    try:
        with paths["calls"].open("x", encoding="utf-8") as stream:

            def save_row(row: dict) -> None:
                stream.write(canonical_json(row) + "\n")
                stream.flush()
                rows.append(row)
                if len(rows) % 10 == 0 or len(rows) == planned_calls:
                    print(f"calls {len(rows)}/{planned_calls}", flush=True)

            with TypeSafeClient(
                model=args.model, retry=RetryPolicy(max_retries=0)
            ) as client:
                run_single_filter(
                    selected,
                    filter_name=args.filter,
                    prompt=prompt,
                    client=client,
                    repeats=args.repeats,
                    threshold=args.threshold,
                    pricing=pricing,
                    model=args.model,
                    on_event=save_row,
                )
        estimate = estimate_filter(repeat_rows(rows), population_size=len(records))
        _write_new_json(paths["estimate"], estimate)
        manifest.update(
            status="complete",
            resolved_models=sorted({r["model"] for r in rows}),
            output_sha256={
                paths[kind].name: digest_file(paths[kind])
                for kind in ("calls", "estimate")
            },
        )
    except BaseException as error:
        manifest.update(
            status="failed",
            error_type=type(error).__name__,
            completed_calls=sum(r.get("status") == "ok" for r in rows),
        )
        raise
    finally:
        manifest["finished_at"] = _now()
        write_json(paths["manifest"], manifest)
    print(
        f"Saved {len(rows)} calls to {paths['calls']}; "
        f"selectivity {estimate['selectivity']:.3f} "
        f"[{estimate['selectivity_ci_low']:.3f}, {estimate['selectivity_ci_high']:.3f}], "
        f"gold {estimate['gold_selectivity']:.3f}"
    )


# ---------------------------------------------------------------------------
# Offline analysis


def _load_run(
    parser: argparse.ArgumentParser, run_dir: Path, label: str
) -> tuple[dict, list[dict]]:
    """Read a complete jev-filter-profile run and verify its output hashes."""
    manifest_path, calls_path = run_dir / "manifest.json", run_dir / "calls.jsonl"
    if not manifest_path.is_file() or not calls_path.is_file():
        parser.error(f"{label} {run_dir} is not a jev-filter-profile output directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "filter_profile":
        parser.error(f"{label} {run_dir} is not a jev-filter-profile output directory")
    if manifest.get("status") != "complete":
        parser.error(
            f"{label} status is {manifest.get('status')!r}, not 'complete'; rerun it"
        )
    for name, expected in (manifest.get("output_sha256") or {}).items():
        if digest_file(run_dir / name) != expected:
            parser.error(f"{label} file {name} does not match its manifest checksum")
    return manifest, read_jsonl(calls_path)


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def live_sample_comparison(
    live_manifest: dict,
    live_rows: list[dict],
    full_rows: list[dict],
    reference: dict,
) -> dict:
    """Live sample estimate vs. full reference, plus the same IDs from the full run.

    The same-ID comparison separates sampling error (which records were drawn)
    from run-to-run model noise (the live call vs. the full-run call).
    """
    population = reference["population_size"]
    live_estimate = estimate_filter(repeat_rows(live_rows), population_size=population)
    ids = set(live_manifest["selected_ids"])
    full_same = [r for r in repeat_rows(full_rows) if r["id"] in ids]
    if len(full_same) != len(ids):
        raise ValueError("Live sample IDs are missing from the full run")
    full_by_id = {r["id"]: r for r in full_same}
    live_first = repeat_rows(live_rows)
    return {
        "run": {
            "sample_size": live_manifest["sample_size"],
            "seed": live_manifest["seed"],
            "selected_ids": live_manifest["selected_ids"],
        },
        "estimate": live_estimate,
        "vs_full_reference": compare(live_estimate, reference),
        "full_run_same_ids_estimate": estimate_filter(
            full_same, population_size=population
        ),
        "decision_agreement_with_full_run": sum(
            r["predicted"] == full_by_id[r["id"]]["predicted"] for r in live_first
        )
        / len(live_first),
        "max_probability_difference_with_full_run": max(
            abs(r["probability"] - full_by_id[r["id"]]["probability"])
            for r in live_first
        ),
    }


CSV_COLUMNS = (
    "k",
    "draws",
    "mean_abs_error_selectivity",
    "p90_abs_error_selectivity",
    "max_abs_error_selectivity",
    "ci_coverage",
    "mean_ci_width",
    "zero_pass_fraction",
    "undefined_precision_fraction",
    "undefined_recall_fraction",
    "mean_rel_error_input_tokens",
    "mean_rel_error_output_tokens",
    "mean_rel_error_latency",
    "mean_rel_error_cost",
    "f1_defined_fraction",
    "mean_f1",
    "std_f1",
)


def analysis_main(argv: Sequence[str] | None = None) -> None:
    """Offline sampling-error analysis of a full jev-filter-profile run (0 API calls)."""
    parser = argparse.ArgumentParser(
        prog="jev-sample-analysis", description=analysis_main.__doc__
    )
    parser.add_argument(
        "--full-run",
        type=Path,
        required=True,
        help="jev-filter-profile output directory without --sample-size",
    )
    parser.add_argument(
        "--live-sample",
        type=Path,
        help="Optional jev-filter-profile sample run to compare with the full reference",
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New directory; default results/milestone3/<filter>-analysis",
    )
    args = parser.parse_args(argv)

    full_manifest, full_rows = _load_run(parser, args.full_run, "--full-run")
    if full_manifest.get("sample_size") != "all":
        parser.error("--full-run must be a full-data run (made without --sample-size)")
    live = None
    if args.live_sample is not None:
        live_manifest, live_rows = _load_run(parser, args.live_sample, "--live-sample")
        if live_manifest.get("sample_size") == "all":
            parser.error(
                "--live-sample must be a sampled run (made with --sample-size)"
            )
        for key in ("filter", "prompt", "threshold", "input_sha256"):
            if live_manifest.get(key) != full_manifest.get(key):
                parser.error(f"--live-sample and --full-run differ in {key}")
        live = (live_manifest, live_rows)
    output_dir = args.output_dir or Path("results/milestone3") / (
        f"{full_manifest['filter']}-analysis"
    )
    if output_dir.exists():
        parser.error(
            f"Output directory already exists: {output_dir}; choose a new --output-dir"
        )

    try:
        analysis = resampling_analysis(
            full_rows,
            sizes=args.sizes,
            draws=args.draws,
            seed=args.seed,
            population_size=full_manifest["n_input_records"],
        )
        if live is not None:
            analysis["live_sample"] = live_sample_comparison(
                *live, full_rows, analysis["reference"]
            )
    except ValueError as error:
        parser.error(str(error))

    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "analysis": output_dir / "analysis.json",
        "csv": output_dir / "analysis.csv",
        "manifest": output_dir / "manifest.json",
    }
    _write_new_json(paths["analysis"], analysis)
    with paths["csv"].open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in analysis["per_k"]:
            writer.writerow({key: row[key] for key in CSV_COLUMNS})

    def run_info(run_dir: Path) -> dict:
        return {
            "dir": str(run_dir),
            "manifest_sha256": digest_file(run_dir / "manifest.json"),
            "calls_sha256": digest_file(run_dir / "calls.jsonl"),
        }

    _write_new_json(
        paths["manifest"],
        {
            "kind": "sample_analysis",
            "status": "complete",
            "created_at": _now(),
            "filter": analysis["filter"],
            "full_run": run_info(args.full_run),
            "live_sample": run_info(args.live_sample) if live else None,
            "sizes": args.sizes,
            "draws": args.draws,
            "seed": args.seed,
            "population_size": analysis["population_size"],
            "api_calls": 0,
            "project_version": _project_version(),
            "python": sys.version.split()[0],
            "output_sha256": {
                paths[kind].name: digest_file(paths[kind])
                for kind in ("analysis", "csv")
            },
        },
    )

    ref = analysis["reference"]
    print(
        f"{analysis['filter']}: reference pass rate {_fmt(ref['selectivity'])} "
        f"(gold {_fmt(ref['gold_selectivity'])}), N={analysis['population_size']}, "
        f"{analysis['draws']} draws per k"
    )
    print(
        f"{'k':>4}  {'mean_abs_err':>12}  {'p90_err':>8}  {'ci_cov':>7}  {'zero_pass':>9}"
    )
    for row in analysis["per_k"]:
        print(
            f"{row['k']:>4}  {_fmt(row['mean_abs_error_selectivity']):>12}  "
            f"{_fmt(row['p90_abs_error_selectivity']):>8}  {_fmt(row['ci_coverage']):>7}  "
            f"{_fmt(row['zero_pass_fraction']):>9}"
        )
    if live is not None:
        quantity = analysis["live_sample"]["vs_full_reference"]["quantities"][
            "selectivity"
        ]
        print(
            f"live sample: pass rate {_fmt(quantity['estimate'])} vs reference "
            f"{_fmt(quantity['reference'])} (abs error {_fmt(quantity['abs_error'])})"
        )
    print(f"Saved analysis to {output_dir}")


ORDERING_CSV_COLUMNS = (
    "k",
    "draws",
    "selectivity_order_agreement",
    "selectivity_tie_fraction",
    "plan_agreement",
    "plan_tie_fraction",
    "mean_regret",
    "max_regret",
)


def ordering_main(argv: Sequence[str] | None = None) -> None:
    """Offline: how often k-record samples pick the same filter order as full data."""
    parser = argparse.ArgumentParser(
        prog="jev-ordering-analysis", description=ordering_main.__doc__
    )
    parser.add_argument(
        "--full-runs",
        type=Path,
        nargs=2,
        required=True,
        metavar="DIR",
        help="Two full jev-filter-profile runs of different filters",
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New directory; default results/milestone3/<a>-vs-<b>-ordering",
    )
    args = parser.parse_args(argv)

    runs = [_load_run(parser, d, f"--full-runs {d}") for d in args.full_runs]
    for manifest, _ in runs:
        if manifest.get("sample_size") != "all":
            parser.error(
                "--full-runs must be full-data runs (made without --sample-size)"
            )
    (manifest_a, rows_a), (manifest_b, rows_b) = runs
    for key in ("threshold", "input_sha256", "n_input_records"):
        if manifest_a.get(key) != manifest_b.get(key):
            parser.error(f"The two full runs differ in {key}")
    output_dir = args.output_dir or Path("results/milestone3") / (
        f"{manifest_a['filter']}-vs-{manifest_b['filter']}-ordering"
    )
    if output_dir.exists():
        parser.error(
            f"Output directory already exists: {output_dir}; choose a new --output-dir"
        )
    try:
        result = ordering_agreement(
            rows_a,
            rows_b,
            sizes=args.sizes,
            draws=args.draws,
            seed=args.seed,
            population_size=manifest_a["n_input_records"],
        )
    except ValueError as error:
        parser.error(str(error))

    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "ordering": output_dir / "ordering.json",
        "csv": output_dir / "ordering.csv",
        "manifest": output_dir / "manifest.json",
    }
    _write_new_json(paths["ordering"], result)
    with paths["csv"].open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ORDERING_CSV_COLUMNS)
        writer.writeheader()
        for row in result["per_k"]:
            writer.writerow({key: row[key] for key in ORDERING_CSV_COLUMNS})
    _write_new_json(
        paths["manifest"],
        {
            "kind": "ordering_analysis",
            "status": "complete",
            "created_at": _now(),
            "filters": result["filters"],
            "full_runs": [
                {
                    "dir": str(d),
                    "manifest_sha256": digest_file(d / "manifest.json"),
                    "calls_sha256": digest_file(d / "calls.jsonl"),
                }
                for d in args.full_runs
            ],
            "sizes": args.sizes,
            "draws": args.draws,
            "seed": args.seed,
            "api_calls": 0,
            "project_version": _project_version(),
            "python": sys.version.split()[0],
            "output_sha256": {
                paths[kind].name: digest_file(paths[kind])
                for kind in ("ordering", "csv")
            },
        },
    )

    ref = result["reference"]
    print(
        f"Full data: best order {' -> '.join(ref['best_order'])}; pass rates "
        + ", ".join(f"{n} {_fmt(v)}" for n, v in ref["selectivity"].items())
        + f"; wrong order costs {100 * ref['wrong_order_regret']:.1f}% more "
        f"(cost basis {result['cost_basis']})"
    )
    print(f"{'k':>4}  {'sel_order_agree':>15}  {'plan_agree':>10}  {'mean_regret':>11}")
    for row in result["per_k"]:
        print(
            f"{row['k']:>4}  {_fmt(row['selectivity_order_agreement']):>15}  "
            f"{_fmt(row['plan_agreement']):>10}  {_fmt(row['mean_regret'], 4):>11}"
        )
    print(f"Saved ordering analysis to {output_dir}")
