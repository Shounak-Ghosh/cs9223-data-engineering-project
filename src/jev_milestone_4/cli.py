"""Sample, estimate, choose the cheapest plan, then execute it on the remaining rows."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from jev_milestone_1.common import (
    binary_metrics,
    canonical_json,
    digest_file,
    read_jsonl,
    write_json,
)
from jev_milestone_1.config import FILTER_PROMPTS, MODEL, THRESHOLD
from jev_milestone_2.datasets import check_sample_manifest, load_task, task_prompts
from jev_milestone_2.experiment import (
    Plan,
    Pricing,
    run_experiment,
    summarize_experiment,
    validate,
)
from jev_milestone_3.sampling import draw_sample

from .optimizer import (
    MAX_FILTERS,
    OBJECTIVES,
    SAMPLE_MODES,
    choose_plan,
    estimate_from_cheap_sample,
    estimate_from_sample,
    plan_from_choice,
    predicted_vs_actual,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sum(rows: list[dict], field: str) -> float | None:
    values = [r[field] for r in rows]
    return sum(values) if all(v is not None for v in values) else None


def _cost_totals(calls: list[dict]) -> dict:
    return {
        "model_calls": len(calls),
        "input_tokens": _sum(calls, "input_tokens"),
        "estimated_cost_usd": _sum(calls, "estimated_cost_usd"),
        "api_latency_seconds": _sum(calls, "latency_seconds"),
    }


def _plan_decisions(result: dict, plan: Plan) -> list[dict]:
    return [d for d in result["decisions"] if d["plan"] == plan.name]


_E2E_NOTES = {
    "full": "Includes sampling overhead. Sample rows are answered by the independent sample run.",
    "cheap_combined": "Includes sampling overhead. Sample rows are answered by the combined sample calls.",
    "cheap_sequential": "Includes sampling overhead. Guard rows are answered by their one-question calls; other sample rows by the chosen plan (sample_fill).",
}


def _quality(rows: list[dict]) -> dict:
    metrics = binary_metrics([r["gold"] for r in rows], [r["predicted"] for r in rows])
    metrics["accuracy"] = (metrics["tp"] + metrics["tn"]) / metrics["n"]
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/sample.jsonl"))
    parser.add_argument(
        "--task",
        type=Path,
        help="Verified task definition (e.g. data/verified_task.json)",
    )
    parser.add_argument(
        "--filters", nargs="+", help="Predicates; defaults to all task predicates"
    )
    parser.add_argument(
        "--max-records", type=int, help="Use only the first n input rows"
    )
    parser.add_argument(
        "--k", type=int, default=20, help="Sample size (Milestone 3 recommends 20)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-mode",
        choices=SAMPLE_MODES,
        default="full",
        help="full: independent and combined on all k rows (Milestone 4); "
        "cheap: combined on all k rows, independent only on --guard-rows rows",
    )
    parser.add_argument(
        "--guard-rows",
        type=int,
        default=5,
        help="Cheap mode only: sample rows that also get one-question calls",
    )
    parser.add_argument(
        "--objective",
        choices=sorted(OBJECTIVES),
        default="input_tokens",
        help="What the optimizer minimizes; 'usd' needs token prices",
    )
    parser.add_argument(
        "--agreement-threshold",
        type=float,
        default=0.95,
        help="Combined plan is allowed only if its sample decisions agree with independent ones at least this often",
    )
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--input-usd-per-million", type=float)
    parser.add_argument("--output-usd-per-million", type=float)
    parser.add_argument("--pricing-source")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plan-only", action="store_true", help="Sample and choose; do not execute"
    )
    mode.add_argument(
        "--evaluate-all-plans",
        action="store_true",
        help="Execute every candidate plan on the remaining rows (predicted vs actual, regret)",
    )
    parser.add_argument(
        "--max-api-calls",
        type=int,
        default=1000,
        help="Reject runs whose worst-case call count exceeds this budget",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/milestone4/optimize")
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print call bounds without credentials, API calls, or files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    rates = (args.input_usd_per_million, args.output_usd_per_million)
    if (rates[0] is None) != (rates[1] is None):
        parser.error("Supply both input and output token rates, or neither")
    if args.objective == "usd" and rates[0] is None:
        parser.error(
            "--objective usd needs --input-usd-per-million and --output-usd-per-million"
        )
    if args.max_api_calls < 1 or (
        args.max_records is not None and args.max_records < 1
    ):
        parser.error("--max-api-calls and --max-records must be positive")
    try:
        pricing = Pricing(*rates) if rates[0] is not None else None
        task = load_task(args.task) if args.task else None
        prompts = task_prompts(task) if task else FILTER_PROMPTS
        filters = tuple(
            args.filters
            if args.filters
            else (list(prompts) if task else ["gratitude", "admiration"])
        )
        if len(filters) > MAX_FILTERS:
            raise ValueError(f"At most {MAX_FILTERS} filters are supported")
        source_manifest = check_sample_manifest(args.input, task)
        records = read_jsonl(args.input)
        if args.max_records is not None:
            records = records[: args.max_records]
        sample_plans = [Plan("independent", filters), Plan("combined", filters)]
        validate(records, sample_plans, 1, args.threshold, prompts)
        if not 0 <= args.agreement_threshold <= 1:
            raise ValueError("--agreement-threshold must be between 0 and 1")
        sample = draw_sample(records, args.k, args.seed)
        if not args.plan_only and len(sample) == len(records):
            raise ValueError(
                "--k must be smaller than the number of rows to leave rows to execute"
            )
        cheap = args.sample_mode == "cheap"
        if cheap and not 1 <= args.guard_rows <= len(sample):
            raise ValueError("--guard-rows must be between 1 and --k")
        guard = draw_sample(sample, args.guard_rows, args.seed) if cheap else []
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    sample_ids = {r["id"] for r in sample}
    rest = [r for r in records if r["id"] not in sample_ids]
    n_f = len(filters)
    if cheap:
        sample_calls = len(sample) + len(guard) * n_f
        # If a sequential plan wins, the non-guard sample rows still need answers.
        fill_max = 0 if args.plan_only else (len(sample) - len(guard)) * n_f
    else:
        sample_calls = len(sample) * (n_f + 1)
        fill_max = 0
    if args.plan_only:
        exec_max = 0
    elif args.evaluate_all_plans:
        exec_max = len(rest) * (math.factorial(n_f) * n_f + 1)
    else:
        exec_max = len(rest) * n_f
    worst_case = sample_calls + fill_max + exec_max
    bounds = {
        "records": len(records),
        "sample_rows": len(sample),
        "execution_rows": 0 if args.plan_only else len(rest),
        "sample_calls": sample_calls,
        "execution_max_calls": exec_max,
        "worst_case_calls": worst_case,
    }
    if cheap:
        bounds["guard_rows"] = len(guard)
        bounds["sample_fill_max_calls"] = fill_max
    print(json.dumps(bounds, indent=2))
    if worst_case > args.max_api_calls:
        parser.error(
            f"Worst case {worst_case} calls exceeds --max-api-calls {args.max_api_calls}; raise it explicitly"
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
    manifest = {
        "status": "running",
        "started_at": _now(),
        "requested_model": args.model,
        "filters": {name: prompts[name] for name in filters},
        "threshold": args.threshold,
        "objective": args.objective,
        "agreement_threshold": args.agreement_threshold,
        "k": args.k,
        "seed": args.seed,
        "mode": "plan_only"
        if args.plan_only
        else "evaluate_all_plans"
        if args.evaluate_all_plans
        else "execute_chosen",
        "input_file": str(args.input),
        "source_sample_sha256": digest_file(args.input),
        "input_manifest": source_manifest,
        "task_sha256": digest_file(args.task) if args.task else None,
        "n_records": len(records),
        "sample_mode": args.sample_mode,
        "sample_ids": [r["id"] for r in sample],
        "guard_ids": [r["id"] for r in guard] if cheap else None,
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
    (out / "sample").mkdir(parents=True)
    write_json(out / "manifest.json", manifest)
    try:
        with ExitStack() as stack:
            client = stack.enter_context(
                TypeSafeClient(model=args.model, retry=RetryPolicy(max_retries=0))
            )

            def run_phase(name: str, rows: list[dict], plans: list[Plan]) -> dict:
                (out / name).mkdir(exist_ok=True)
                with ExitStack() as files:
                    streams = {
                        kind: files.enter_context(
                            (out / name / f"{kind}.jsonl").open("x", encoding="utf-8")
                        )
                        for kind in ("calls", "decisions", "runs")
                    }

                    def save(kind: str, row: dict) -> None:
                        streams[kind].write(canonical_json(row) + "\n")
                        streams[kind].flush()
                        if kind == "runs":
                            print(
                                f"[{name}] plan={row['plan']} calls={row['model_calls']}",
                                flush=True,
                            )

                    return run_experiment(
                        rows,
                        plans,
                        client=client,
                        repeats=1,
                        threshold=args.threshold,
                        pricing=pricing,
                        on_event=save,
                        prompts=prompts,
                        requested_model=args.model,
                    )

            independent_plan, combined_plan = sample_plans
            if cheap:
                sample_result = run_phase("sample", sample, [combined_plan])
                guard_result = run_phase("guard", guard, [independent_plan])
                estimates = estimate_from_cheap_sample(
                    sample_result, guard_result, filters
                )
                sampling_calls = sample_result["calls"] + guard_result["calls"]
            else:
                sample_result = run_phase("sample", sample, sample_plans)
                estimates = estimate_from_sample(sample_result, filters)
                sampling_calls = sample_result["calls"]
            choice = choose_plan(
                estimates,
                objective=args.objective,
                agreement_threshold=args.agreement_threshold,
            )
            plan_doc = {"estimates": estimates, "choice": choice}
            write_json(out / "plan.json", plan_doc)
            print(f"Chosen plan: {choice['chosen']}", flush=True)

            summary: dict = {
                "chosen": choice["chosen"],
                "objective": args.objective,
                "sample_mode": args.sample_mode,
                "sampling_overhead": _cost_totals(sampling_calls),
            }
            all_calls = list(sampling_calls)
            if not args.plan_only:
                chosen = plan_from_choice(choice)
                # Answers for the sample rows, reusing paid calls where possible.
                fill_calls: list[dict] = []
                if not cheap:
                    sample_answers = _plan_decisions(sample_result, independent_plan)
                elif chosen.strategy == "combined":
                    sample_answers = _plan_decisions(sample_result, combined_plan)
                else:
                    sample_answers = _plan_decisions(guard_result, independent_plan)
                    guard_ids = {r["id"] for r in guard}
                    fill_rows = [r for r in sample if r["id"] not in guard_ids]
                    if fill_rows:
                        fill = run_phase("sample_fill", fill_rows, [chosen])
                        sample_answers += _plan_decisions(fill, chosen)
                        fill_calls = fill["calls"]
                if cheap:
                    summary["sample_fill"] = _cost_totals(fill_calls)
                plans = [chosen]
                if args.evaluate_all_plans:
                    plans += [
                        Plan(c["strategy"], tuple(c["filters"]))
                        for c in choice["candidates"]
                        if c["plan"] != chosen.name
                    ]
                execution = run_phase("execution", rest, plans)
                all_calls += fill_calls + execution["calls"]
                summary["execution"] = summarize_experiment(execution)
                summary["predicted_vs_actual"] = predicted_vs_actual(choice, execution)
                chosen_rows = _plan_decisions(execution, chosen)
                chosen_calls = [
                    c for c in execution["calls"] if c["plan"] == chosen.name
                ]
                e2e_calls = sampling_calls + fill_calls + chosen_calls
                summary["optimizer_end_to_end"] = {
                    "rows": len(sample_answers) + len(chosen_rows),
                    **_cost_totals(e2e_calls),
                    "quality": _quality(sample_answers + chosen_rows),
                    "note": _E2E_NOTES[
                        "full"
                        if not cheap
                        else "cheap_combined"
                        if chosen.strategy == "combined"
                        else "cheap_sequential"
                    ],
                }
            write_json(out / "summary.json", summary)
        manifest.update(
            status="complete",
            resolved_models=sorted({c["model"] for c in all_calls if c.get("model")}),
        )
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        manifest["finished_at"] = _now()
        write_json(out / "manifest.json", manifest)
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
