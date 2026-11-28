import json
import math

import pytest
from test_optimizer import FakeClient, install_cli_fake, make_records, write_input

from jev_milestone_2.experiment import Plan, run_experiment
from jev_milestone_3.sampling import draw_sample
from jev_milestone_4.cli import main as optimize_main
from jev_milestone_4.optimizer import choose_plan, estimate_from_cheap_sample
from jev_milestone_5.benchmark import main as benchmark_main
from jev_milestone_5.collect import main as collect_main
from jev_milestone_5.collect import parse_query
from jev_milestone_5.replay import (
    CallStore,
    baselines,
    check_repeats,
    project,
    run_plan,
    simulate_optimizer,
    totals,
)

GA = ("gratitude", "admiration")
# gratitude passes 1 row in 10, admiration 5 in 10, so gratitude-first is cheap.
PASSES = {"gratitude": set(range(0, 30, 10)), "admiration": set(range(0, 30, 2))}
TOKENS = {"gratitude": 100, "admiration": 100}


def fake(combined_tokens=150, combined_flip=None):
    _, _, probs = make_records(PASSES, 30)
    return FakeClient(probs, TOKENS, combined_tokens, combined_flip)


def prompts():
    return {f: f"Does the text satisfy {f}?" for f in PASSES}


# ---- cheap estimates ----


def test_cheap_estimates_use_combined_pass_rates_and_guard_costs():
    filters, records, probs = make_records(PASSES, 30)
    client = FakeClient(probs, TOKENS, 150)
    p = {f: "q" for f in filters}
    sample = draw_sample(records, 20, 1)
    guard = draw_sample(sample, 5, 1)
    comb = run_experiment(sample, [Plan("combined", filters)], client=client, prompts=p)
    ind = run_experiment(
        guard, [Plan("independent", filters)], client=client, prompts=p
    )
    est = estimate_from_cheap_sample(comb, ind, filters)
    assert est["mode"] == "cheap" and est["k"] == 20 and est["guard_rows"] == 5
    assert len(est["decision_matrix"]) == 20
    assert est["per_filter"]["gratitude"]["n_calls"] == 5
    assert est["per_filter"]["gratitude"]["mean_cost"]["input_tokens"] == 100
    assert est["combined"]["mean_cost"]["input_tokens"] == 150
    assert est["combined"]["agreement_with_independent"] == 1
    with pytest.raises(ValueError, match="subset"):
        other = run_experiment(
            [r for r in records if r not in sample][:3],
            [Plan("independent", filters)],
            client=client,
            prompts=p,
        )
        estimate_from_cheap_sample(comb, other, filters)


# ---- live CLI, cheap mode ----


def run_optimize(tmp_path, monkeypatch, client, *extra):
    path, _, p, _ = write_input(tmp_path, PASSES, 30)
    install_cli_fake(monkeypatch, client, p)
    out = tmp_path / ("run" + "".join(extra).replace("-", ""))
    optimize_main(
        [
            "--input",
            str(path),
            "--k",
            "10",
            "--seed",
            "3",
            "--output-dir",
            str(out),
            *extra,
        ]
    )
    return (
        path,
        out,
        {
            n: json.loads((out / f"{n}.json").read_text())
            for n in ("manifest", "plan", "summary")
        },
    )


def test_cli_cheap_mode_combined_reuses_sample_answers(tmp_path, monkeypatch):
    client = fake(combined_tokens=101)  # combined almost as cheap as one call
    _, out, r = run_optimize(
        tmp_path, monkeypatch, client, "--sample-mode", "cheap", "--guard-rows", "3"
    )
    assert r["plan"]["choice"]["chosen"] == "combined:gratitude,admiration"
    assert (
        r["manifest"]["sample_mode"] == "cheap" and len(r["manifest"]["guard_ids"]) == 3
    )
    # 10 combined + 3 × 2 guard + 20 combined execution; no fill
    assert len(client.calls) == 10 + 6 + 20
    assert r["summary"]["sample_fill"]["model_calls"] == 0
    e2e = r["summary"]["optimizer_end_to_end"]
    assert e2e["rows"] == 30 and e2e["model_calls"] == 36
    assert "combined sample calls" in e2e["note"]
    assert not (out / "sample_fill").exists()


def test_cli_cheap_mode_sequential_fills_non_guard_rows(tmp_path, monkeypatch):
    client = fake(combined_tokens=190)
    _, out, r = run_optimize(
        tmp_path, monkeypatch, client, "--sample-mode", "cheap", "--guard-rows", "3"
    )
    assert r["plan"]["choice"]["chosen"] == "sequential:gratitude,admiration"
    fill = r["summary"]["sample_fill"]
    assert fill["model_calls"] >= 7  # 7 non-guard sample rows, ≥1 call each
    assert (out / "sample_fill" / "calls.jsonl").is_file()
    e2e = r["summary"]["optimizer_end_to_end"]
    assert e2e["rows"] == 30 and e2e["model_calls"] == len(client.calls)


def test_cli_cheap_mode_bounds_and_validation(tmp_path, capsys):
    path, *_ = write_input(tmp_path, PASSES, 30)
    optimize_main(
        [
            "--input",
            str(path),
            "--k",
            "10",
            "--sample-mode",
            "cheap",
            "--guard-rows",
            "4",
            "--dry-run",
        ]
    )
    bounds = json.loads(capsys.readouterr().out)
    assert bounds["sample_calls"] == 10 + 4 * 2
    assert bounds["sample_fill_max_calls"] == 6 * 2
    assert bounds["worst_case_calls"] == 18 + 12 + 20 * 2
    for bad in ("0", "11"):
        with pytest.raises(SystemExit):
            optimize_main(
                [
                    "--input",
                    str(path),
                    "--k",
                    "10",
                    "--sample-mode",
                    "cheap",
                    "--guard-rows",
                    bad,
                    "--dry-run",
                ]
            )


# ---- replay matches live execution ----


def collected_store(tmp_path, monkeypatch, client, queries=("gratitude,admiration",)):
    path, _, p, _ = write_input(tmp_path, PASSES, 30)
    install_cli_fake(monkeypatch, client, p)
    out = tmp_path / "collect"
    args = ["--input", str(path), "--output-dir", str(out)]
    for q in queries:
        args += ["--query", q]
    collect_main(args)
    store = CallStore()
    for f in sorted(out.glob("*/calls.jsonl")):
        for line in f.read_text().splitlines():
            store.add(json.loads(line))
    return path, out, store


def test_collect_makes_one_question_and_combined_calls_once(tmp_path, monkeypatch):
    client = fake()
    _, out, store = collected_store(tmp_path, monkeypatch, client)
    assert len(client.calls) == 30 * 2 + 30
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == [
        "combined-gratitude-admiration",
        "independent",
    ]
    assert len(store.single) == 60 and len(store.combined) == 30
    assert check_repeats(store)["repeated_requests"] == 0


def test_collect_dry_run_and_bad_queries(tmp_path, capsys):
    path, *_ = write_input(tmp_path, PASSES, 30)
    collect_main(["--input", str(path), "--query", "gratitude,admiration", "--dry-run"])
    assert json.loads(capsys.readouterr().out)["total_calls"] == 90
    for q in ("gratitude", "gratitude,gratitude", "gratitude,unknown"):
        with pytest.raises(SystemExit):
            collect_main(["--input", str(path), "--query", q, "--dry-run"])
    with pytest.raises(SystemExit):
        collect_main(
            [
                "--input",
                str(path),
                "--query",
                "gratitude,admiration",
                "--max-api-calls",
                "10",
                "--dry-run",
            ]
        )
    assert parse_query(" a , b ") == ("a", "b")


def test_replayed_plans_match_live_runs(tmp_path, monkeypatch):
    client = fake()
    _, _, store = collected_store(tmp_path, monkeypatch, client)
    _, records, probs = make_records(PASSES, 30)
    ids = [r["id"] for r in records]
    for plan in [
        Plan("sequential", GA),
        Plan("sequential", GA[::-1]),
        Plan("combined", GA),
    ]:
        live = run_experiment(
            records, [plan], client=FakeClient(probs, TOKENS, 150), prompts=prompts()
        )
        replay = run_plan(store, plan, ids)
        assert totals(replay.calls)["input_tokens"] == sum(
            c["input_tokens"] for c in live["calls"]
        )
        assert replay.predicted == {d["id"]: d["predicted"] for d in live["decisions"]}


@pytest.mark.parametrize("mode,extra", [("full", []), ("cheap", ["--guard-rows", "3"])])
@pytest.mark.parametrize("combined_tokens", [101, 190])
def test_replayed_optimizer_matches_live_optimizer(
    tmp_path, monkeypatch, mode, extra, combined_tokens
):
    _, _, store = collected_store(tmp_path, monkeypatch, fake(combined_tokens))
    _, records, _ = make_records(PASSES, 30)
    gold = {r["id"]: int(all(r["gold"].values())) for r in records}
    sim = simulate_optimizer(
        store,
        GA,
        records,
        gold,
        k=10,
        seed=3,
        mode=mode,
        guard_rows=3,
        objective="input_tokens",
        agreement_threshold=0.95,
    )
    live_dir = tmp_path / "live"
    live_dir.mkdir()
    _, _, live = run_optimize(
        live_dir, monkeypatch, fake(combined_tokens), "--sample-mode", mode, *extra
    )
    assert sim["chosen"] == live["plan"]["choice"]["chosen"]
    e2e = live["summary"]["optimizer_end_to_end"]
    assert sim["total"]["input_tokens"] == e2e["input_tokens"]
    assert sim["total"]["calls"] == e2e["model_calls"]
    assert sim["quality"]["f1"] == pytest.approx(e2e["quality"]["f1"])


# ---- benchmark ----


def test_benchmark_end_to_end(tmp_path, monkeypatch, capsys):
    path, coll, _ = collected_store(tmp_path, monkeypatch, fake(190))
    out = tmp_path / "bench"
    benchmark_main(
        [
            "--input",
            str(path),
            "--calls",
            *map(str, coll.glob("*/calls.jsonl")),
            "--query",
            "gratitude,admiration",
            "--seeds",
            "4",
            "--k",
            "10",
            "--guard-rows",
            "3",
            "5",
            "--output-dir",
            str(out),
        ]
    )
    assert "Replay benchmark" in capsys.readouterr().out
    for name in (
        "baselines.csv",
        "optimizer_runs.csv",
        "optimizer_summary.csv",
        "projection.csv",
        "summary.json",
        "report.md",
        "manifest.json",
    ):
        assert (out / name).is_file()
    summary = json.loads((out / "summary.json").read_text())["queries"][
        "gratitude AND admiration"
    ]
    assert set(summary["optimizer"]) == {"full-k10", "cheap-g3-k10", "cheap-g5-k10"}
    full = summary["optimizer"]["full-k10"]["summary"]
    assert full["oracle_plan"] == "sequential:gratitude,admiration"
    assert full["picked_oracle_plan"] == 1 and full["plan_regret"]["max"] == 0
    separate = summary["baselines"]["independent:gratitude,admiration"]["totals"]
    assert (
        full["reference_plans"]["always_separate"] == "independent:gratitude,admiration"
    )
    assert full["ratio_vs_always_separate"]["mean"] == pytest.approx(
        full["total_cost"]["mean"] / separate["input_tokens"]
    )
    assert "vs always-separate" in (out / "report.md").read_text()
    assert json.loads((out / "manifest.json").read_text())["api_calls_made"] == 0
    with pytest.raises(SystemExit):  # never overwrites
        benchmark_main(
            [
                "--input",
                str(path),
                "--calls",
                str(coll / "independent" / "calls.jsonl"),
                "--query",
                "gratitude,admiration",
                "--output-dir",
                str(out),
            ]
        )


def test_benchmark_rejects_incomplete_calls(tmp_path, monkeypatch):
    path, coll, _ = collected_store(tmp_path, monkeypatch, fake())
    with pytest.raises(SystemExit):  # combined calls missing
        benchmark_main(
            [
                "--input",
                str(path),
                "--calls",
                str(coll / "independent" / "calls.jsonl"),
                "--query",
                "gratitude,admiration",
                "--output-dir",
                str(tmp_path / "b"),
            ]
        )


def test_repeat_check_counts_disagreements():
    store = CallStore()
    a = {
        "status": "ok",
        "id": "1",
        "filters": ["x"],
        "probabilities": {"x": 0.9},
        "input_tokens": 10,
        "output_tokens": 0,
        "latency_seconds": 0.1,
        "estimated_cost_usd": None,
    }
    store.add(a)
    store.add({**a, "probabilities": {"x": 0.1}})
    store.add({**a, "input_tokens": 11})
    assert check_repeats(store) == {
        "repeated_requests": 2,
        "same_input_tokens": 1,
        "same_decisions": 1,
    }
    assert store.single[("1", "x")]["probabilities"]["x"] == 0.9  # first wins


def test_projection_break_even():
    base = {
        "combined:a,b": {"strategy": "combined", "per_row": {"input_tokens": 100.0}},
        "sequential:a,b": {
            "strategy": "sequential",
            "per_row": {"input_tokens": 120.0},
        },
        "sequential:b,a": {
            "strategy": "sequential",
            "per_row": {"input_tokens": 200.0},
        },
        "independent:a,b": {
            "strategy": "independent",
            "per_row": {"input_tokens": 250.0},
        },
    }
    runs = [
        {
            "chosen": "combined:a,b",
            "sampling": {"input_tokens": 1500.0},
            "sample_fill": {"input_tokens": 0.0},
        }
    ]
    p = project(runs, base, ("a", "b"), "input_tokens", [100, 1000], k=10)
    # 500 extra tokens beyond answering 10 rows; saves 20 per row vs best order
    # and 150 per row vs always separate
    assert p["break_even_rows_vs_always_separate"] == pytest.approx(500 / 150)
    assert p["break_even_rows_vs_best_fixed_order"] == pytest.approx(25)
    assert math.isinf(p["break_even_rows_vs_always_combined"])
    assert p["ratio_vs_always_combined@1000"] == pytest.approx(
        (1500 + 990 * 100) / 100_000
    )


def test_baselines_report_every_plan():
    _, records, probs = make_records(PASSES, 30)
    store = CallStore()
    client = FakeClient(probs, TOKENS, 150)
    for plan in [Plan("independent", GA), Plan("combined", GA)]:
        for call in run_experiment(records, [plan], client=client, prompts=prompts())[
            "calls"
        ]:
            store.add(call)
    gold = {r["id"]: int(all(r["gold"].values())) for r in records}
    base = baselines(store, GA, [r["id"] for r in records], gold)
    assert set(base) == {
        "sequential:gratitude,admiration",
        "sequential:admiration,gratitude",
        "combined:gratitude,admiration",
        "independent:gratitude,admiration",
    }
    assert base["combined:gratitude,admiration"]["totals"]["input_tokens"] == 30 * 150
    assert (
        base["sequential:gratitude,admiration"]["totals"]["input_tokens"]
        == 30 * 100 + 3 * 100
    )
    assert choose_plan  # imported API stays available
