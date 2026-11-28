"""Replay the optimizer against always-combined and fixed-order baselines.

No API calls: every plan is rebuilt from recorded calls (`jev-collect`, or any
Milestone 2/4 run that covers every row). For each query it reports the baselines
over all rows and the optimizer across seeds, sample sizes, and sampling modes,
plus projections to larger inputs.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from jev_milestone_1.common import digest_file, read_jsonl, write_json
from jev_milestone_1.config import THRESHOLD
from jev_milestone_2.experiment import Pricing
from jev_milestone_4.optimizer import OBJECTIVES, SAMPLE_MODES

from .collect import parse_query
from .replay import (
    CallStore,
    baselines,
    check_repeats,
    project,
    simulate_optimizer,
    summarize_runs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Records with gold labels"
    )
    parser.add_argument(
        "--calls",
        type=Path,
        nargs="+",
        required=True,
        help="Recorded calls.jsonl files",
    )
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument("--label", help="Dataset label used in reports")
    parser.add_argument("--seeds", type=int, default=20, help="Seeds 0..n-1")
    parser.add_argument("--k", type=int, nargs="+", default=[10, 20])
    parser.add_argument(
        "--modes", nargs="+", choices=SAMPLE_MODES, default=list(SAMPLE_MODES)
    )
    parser.add_argument(
        "--guard-rows",
        type=int,
        nargs="+",
        default=[5],
        help="Cheap mode guard sizes to compare",
    )
    parser.add_argument(
        "--objective", choices=sorted(OBJECTIVES), default="input_tokens"
    )
    parser.add_argument("--agreement-threshold", type=float, default=0.95)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--input-usd-per-million", type=float)
    parser.add_argument("--output-usd-per-million", type=float)
    parser.add_argument(
        "--project-sizes", type=int, nargs="+", default=[100, 1000, 10000]
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/milestone5/benchmark")
    )
    return parser


def _fmt(value, pct=False, digits=2):
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isinf(value):
        return "never"
    return f"{value * 100:+.1f}%" if pct else f"{value:,.{digits}f}"


def _rows_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    rates = (args.input_usd_per_million, args.output_usd_per_million)
    if (rates[0] is None) != (rates[1] is None):
        parser.error("Supply both input and output token rates, or neither")
    if args.seeds < 1 or not 0 <= args.agreement_threshold <= 1:
        parser.error("--seeds must be positive and --agreement-threshold in [0, 1]")
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    try:
        pricing = Pricing(*rates) if rates[0] is not None else None
        queries = list(dict.fromkeys(parse_query(q) for q in args.query))
        records = read_jsonl(args.input)
        store = CallStore(threshold=args.threshold)
        for path in args.calls:
            for call in read_jsonl(path):
                store.add(call, pricing)
        ids = [r["id"] for r in records]
        for q in queries:
            if any(f not in records[0]["gold"] for f in q):
                raise ValueError(f"Records lack gold labels for query {q}")
            gaps = store.missing(ids, q)
            if gaps:
                raise ValueError(
                    f"Recorded calls do not cover query {','.join(q)}: missing {gaps}"
                )
        for k in args.k:
            if not all(1 <= g <= k < len(records) for g in args.guard_rows):
                raise ValueError("Need 1 <= --guard-rows <= k < number of records")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))

    label = args.label or args.input.stem
    objective = args.objective
    run_rows, summary_rows, baseline_rows, projection_rows = [], [], [], []
    report = {"queries": {}}
    for q in queries:
        qname = " AND ".join(q)
        gold = {r["id"]: int(all(r["gold"][f] for f in q)) for r in records}
        base = baselines(store, q, ids, gold)
        for name, b in base.items():
            baseline_rows.append(
                {
                    "dataset": label,
                    "query": qname,
                    "plan": name,
                    **{f"total_{m}": v for m, v in b["totals"].items()},
                    "f1": b["quality"]["f1"],
                    "precision": b["quality"]["precision"],
                    "recall": b["quality"]["recall"],
                }
            )
        qreport = {
            "gold_positives": sum(gold.values()),
            "baselines": {
                n: {kk: v for kk, v in b.items() if kk != "predicted"}
                for n, b in base.items()
            },
            "optimizer": {},
        }
        settings = [
            (mode, g)
            for mode in args.modes
            for g in ([None] if mode == "full" else args.guard_rows)
        ]
        for mode, g in settings:
            for k in args.k:
                runs = [
                    simulate_optimizer(
                        store,
                        q,
                        records,
                        gold,
                        k=k,
                        seed=seed,
                        mode=mode,
                        guard_rows=g or k,
                        objective=objective,
                        agreement_threshold=args.agreement_threshold,
                    )
                    for seed in range(args.seeds)
                ]
                for seed, r in enumerate(runs):
                    run_rows.append(
                        {
                            "dataset": label,
                            "query": qname,
                            "mode": mode,
                            "guard_rows": g or k,
                            "k": k,
                            "seed": seed,
                            "chosen": r["chosen"],
                            "combined_agreement": r["combined_agreement"],
                            "combined_allowed": r["combined_allowed"],
                            f"sampling_{objective}": r["sampling"][objective],
                            f"sample_fill_{objective}": r["sample_fill"][objective],
                            f"execution_{objective}": r["execution"][objective],
                            f"total_{objective}": r["total"][objective],
                            "total_calls": r["total"]["calls"],
                            "predicted_cost_per_row": r["predicted_cost_per_row"],
                            "actual_cost_per_row": r["actual_cost_per_row"],
                            "f1": r["quality"]["f1"],
                        }
                    )
                agg = summarize_runs(runs, base, q, objective)
                proj = project(runs, base, q, objective, args.project_sizes, k)
                key = f"{mode}-k{k}" if g is None else f"{mode}-g{g}-k{k}"
                qreport["optimizer"][key] = {"summary": agg, "projection": proj}
                summary_rows.append(
                    {
                        "dataset": label,
                        "query": qname,
                        "mode": mode,
                        "guard_rows": g or k,
                        "k": k,
                        "oracle_plan": agg["oracle_plan"],
                        "picked_oracle_plan": agg["picked_oracle_plan"],
                        "mean_plan_regret": agg["plan_regret"]["mean"],
                        "max_plan_regret": agg["plan_regret"]["max"],
                        "mean_sampling_cost": agg["sampling_cost"]["mean"],
                        "mean_total_cost": agg["total_cost"]["mean"],
                        "ratio_vs_always_combined": agg["ratio_vs_always_combined"][
                            "mean"
                        ],
                        "ratio_vs_always_separate": agg["ratio_vs_always_separate"][
                            "mean"
                        ],
                        "ratio_vs_best_fixed_order": agg["ratio_vs_best_fixed_order"][
                            "mean"
                        ],
                        "ratio_vs_worst_fixed_order": agg["ratio_vs_worst_fixed_order"][
                            "mean"
                        ],
                        "ratio_vs_oracle": agg["ratio_vs_oracle"]["mean"],
                        "mean_relative_prediction_error": agg[
                            "relative_prediction_error"
                        ]["mean"],
                        "guard_rejected_combined": agg["guard_rejected_combined"],
                        "mean_f1": agg["f1"]["mean"],
                        "mean_agreement_with_independent": agg[
                            "agreement_with_independent_answers"
                        ]["mean"],
                    }
                )
                projection_rows.append(
                    {
                        "dataset": label,
                        "query": qname,
                        "mode": mode,
                        "guard_rows": g or k,
                        "k": k,
                        **proj,
                    }
                )
        report["queries"][qname] = qreport

    out = args.output_dir
    out.mkdir(parents=True)
    _rows_csv(out / "baselines.csv", baseline_rows)
    _rows_csv(out / "optimizer_runs.csv", run_rows)
    _rows_csv(out / "optimizer_summary.csv", summary_rows)
    _rows_csv(out / "projection.csv", projection_rows)
    write_json(out / "summary.json", report)
    (out / "report.md").write_text(
        _markdown(label, objective, args, report), encoding="utf-8"
    )
    write_json(
        out / "manifest.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "dataset": label,
            "input_file": str(args.input),
            "input_sha256": digest_file(args.input),
            "calls_files": {str(p): digest_file(p) for p in args.calls},
            "queries": [",".join(q) for q in queries],
            "seeds": list(range(args.seeds)),
            "k": args.k,
            "modes": args.modes,
            "guard_rows": args.guard_rows,
            "objective": objective,
            "agreement_threshold": args.agreement_threshold,
            "threshold": args.threshold,
            "pricing": vars(pricing) if pricing else None,
            "repeat_check": check_repeats(store),
            "api_calls_made": 0,
        },
    )
    print((out / "report.md").read_text(encoding="utf-8"))
    print(f"Saved benchmark to {out}")


def _markdown(label: str, objective: str, args, report: dict) -> str:
    lines = [f"# Replay benchmark: {label}", ""]
    lines.append(
        f"Objective `{objective}`; {args.seeds} seeds; k in {args.k}; cheap-mode guard rows "
        f"{args.guard_rows} (`g`); agreement threshold {args.agreement_threshold}. No API calls."
    )
    for qname, q in report["queries"].items():
        lines += ["", f"## {qname}", "", f"Gold positives: {q['gold_positives']}", ""]
        lines += [
            f"| Plan (all rows) | Total {objective} | Calls | F1 |",
            "| --- | --- | --- | --- |",
        ]
        for name, b in q["baselines"].items():
            tag = " (always separate)" if b["strategy"] == "independent" else ""
            lines.append(
                f"| `{name}`{tag} | {_fmt(b['totals'][objective], digits=0)} | "
                f"{_fmt(b['totals']['calls'], digits=0)} | {_fmt(b['quality']['f1'], digits=3)} |"
            )
        lines += [
            "",
            "| Mode, k | Picked oracle plan | Mean sampling cost | Mean total | vs always-combined | vs always-separate | vs best order | vs worst order | Prediction error | F1 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for key, o in q["optimizer"].items():
            s = o["summary"]
            lines.append(
                f"| {key} | {s['picked_oracle_plan']:.0%} | {_fmt(s['sampling_cost']['mean'], digits=0)} | "
                f"{_fmt(s['total_cost']['mean'], digits=0)} | "
                f"{_fmt(s['ratio_vs_always_combined']['mean'] - 1, pct=True)} | "
                f"{_fmt(s['ratio_vs_always_separate']['mean'] - 1, pct=True)} | "
                f"{_fmt(s['ratio_vs_best_fixed_order']['mean'] - 1, pct=True)} | "
                f"{_fmt(s['ratio_vs_worst_fixed_order']['mean'] - 1, pct=True)} | "
                f"{_fmt(s['relative_prediction_error']['mean'], pct=True)} | {_fmt(s['f1']['mean'], digits=3)} |"
            )
        sizes = args.project_sizes
        lines += [
            "",
            "| Mode, k | "
            + " | ".join(f"vs combined @{n:,}" for n in sizes)
            + " | "
            + " | ".join(f"vs best order @{n:,}" for n in sizes)
            + " | Break-even vs best order | Break-even vs always-separate |",
            "| --- " * (2 * len(sizes) + 3) + "|",
        ]
        for key, o in q["optimizer"].items():
            p = o["projection"]
            lines.append(
                f"| {key} | "
                + " | ".join(
                    _fmt(p[f"ratio_vs_always_combined@{n}"] - 1, pct=True)
                    for n in sizes
                )
                + " | "
                + " | ".join(
                    _fmt(p[f"ratio_vs_best_fixed_order@{n}"] - 1, pct=True)
                    for n in sizes
                )
                + f" | {_fmt(p['break_even_rows_vs_best_fixed_order'], digits=0)} rows"
                + f" | {_fmt(p['break_even_rows_vs_always_separate'], digits=0)} rows |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
