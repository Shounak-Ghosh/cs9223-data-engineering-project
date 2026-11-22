import builtins
import json
import os
import sys
from types import SimpleNamespace

import pytest

from jev_milestone_2.experiment import Plan, Pricing, run_experiment
from jev_milestone_4.cli import main
from jev_milestone_4.optimizer import (
    choose_plan,
    estimate_from_sample,
    predicted_vs_actual,
    reach_fractions,
)


class FakeClient:
    """probs[text][filter] -> probability; tokens per single-filter call; combined tokens per row."""

    def __init__(self, probs, tokens, combined_tokens, combined_flip=None):
        self.probs, self.tokens, self.combined_tokens = probs, tokens, combined_tokens
        self.combined_flip = combined_flip
        self.calls = []

    def system_one(self, *, state, questions):
        names = tuple(questions)
        self.calls.append((state, names))
        answers = {n: self.probs[state][n] for n in names}
        if len(names) > 1 and self.combined_flip:
            answers[self.combined_flip] = 1 - answers[self.combined_flip]
        tokens = self.combined_tokens if len(names) > 1 else self.tokens[names[0]]
        return SimpleNamespace(
            model="fake-model",
            request_id=f"r{len(self.calls)}",
            nouls={n: SimpleNamespace(noul=p) for n, p in answers.items()},
            usage=SimpleNamespace(input_tokens=tokens, output_tokens=0),
        )


@pytest.fixture(autouse=True)
def stub_sdk(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "typesafe_sdk",
        SimpleNamespace(Noul=lambda **kw: kw, TypeSafeError=RuntimeError),
    )


def make_records(passes: dict[str, set[int]], n: int):
    """passes[filter] = set of row indexes where Jev says yes."""
    filters = tuple(passes)
    records = [
        {
            "id": f"r{i}",
            "text": f"r{i}",
            "gold": {f: int(i in passes[f]) for f in filters},
        }
        for i in range(n)
    ]
    probs = {
        f"r{i}": {f: (0.9 if i in passes[f] else 0.1) for f in filters}
        for i in range(n)
    }
    return filters, records, probs


def sample_estimates(
    passes, n, tokens, combined_tokens, combined_flip=None, pricing=None
):
    filters, records, probs = make_records(passes, n)
    fake = FakeClient(probs, tokens, combined_tokens, combined_flip)
    result = run_experiment(
        records,
        [Plan("independent", filters), Plan("combined", filters)],
        client=fake,
        pricing=pricing,
        prompts={f: f"q {f}" for f in filters},
    )
    return filters, records, probs, estimate_from_sample(result, filters)


def test_selective_filter_goes_first_when_costs_equal():
    _, _, _, est = sample_estimates(
        {"a": set(range(8)), "b": {0, 1}}, 10, {"a": 100, "b": 100}, 500
    )
    assert est["per_filter"]["a"]["selectivity"] == 0.8
    assert est["per_filter"]["b"]["selectivity"] == 0.2
    choice = choose_plan(est, objective="input_tokens", agreement_threshold=0.95)
    assert choice["chosen"] == "sequential:b,a"
    costs = {c["plan"]: c["predicted_cost_per_row"] for c in choice["candidates"]}
    assert costs["sequential:b,a"] == pytest.approx(100 + 0.2 * 100)
    assert costs["sequential:a,b"] == pytest.approx(100 + 0.8 * 100)


def test_cheaper_filter_goes_first_when_selectivity_equal():
    _, _, _, est = sample_estimates(
        {"a": set(range(5)), "b": set(range(5, 10))}, 10, {"a": 300, "b": 100}, 1000
    )
    choice = choose_plan(est, objective="input_tokens", agreement_threshold=0.95)
    assert choice["chosen"] == "sequential:b,a"


def test_combined_chosen_when_cheapest_and_quality_guard_passes():
    _, _, _, est = sample_estimates(
        {"a": set(range(8)), "b": set(range(8))}, 10, {"a": 100, "b": 100}, 120
    )
    choice = choose_plan(est, objective="input_tokens", agreement_threshold=0.95)
    assert est["combined"]["agreement_with_independent"] == 1
    assert choice["chosen"] == "combined:a,b"


def test_quality_guard_rejects_combined():
    _, _, _, est = sample_estimates(
        {"a": set(range(8)), "b": set(range(8))},
        10,
        {"a": 100, "b": 100},
        120,
        combined_flip="b",
    )
    assert est["combined"]["agreement_with_independent"] < 0.95
    choice = choose_plan(est, objective="input_tokens", agreement_threshold=0.95)
    assert choice["chosen_strategy"] == "sequential"
    combined = next(c for c in choice["candidates"] if c["strategy"] == "combined")
    assert not combined["allowed"] and "quality guard" in combined["reason"]
    # Threshold 0 disables the guard.
    assert (
        choose_plan(est, objective="input_tokens", agreement_threshold=0)["chosen"]
        == "combined:a,b"
    )


def test_correlated_filters_use_joint_reach_not_independence():
    # a and b never pass together; c passes a subset of a's rows.
    passes = {"a": set(range(5)), "b": set(range(5, 10)), "c": {0, 1, 2, 3}}
    _, _, _, est = sample_estimates(passes, 10, {"a": 100, "b": 100, "c": 100}, 10_000)
    assert reach_fractions(("c", "b", "a"), est["decision_matrix"]) == [1.0, 0.4, 0.0]
    choice = choose_plan(est, objective="input_tokens", agreement_threshold=0.95)
    assert choice["chosen"] == "sequential:c,b,a"
    costs = {c["plan"]: c for c in choice["candidates"]}
    assert costs["sequential:c,b,a"]["predicted_cost_per_row"] == pytest.approx(140)
    assert costs["sequential:c,a,b"]["predicted_cost_per_row"] == pytest.approx(180)
    # Multiplying selectivities would rate these two orders as equal.
    assert costs["sequential:c,a,b"][
        "independence_assumption_cost_per_row"
    ] == pytest.approx(
        costs["sequential:c,b,a"]["independence_assumption_cost_per_row"]
    )


def test_objectives_calls_and_usd():
    passes = {"a": set(range(8)), "b": {0, 1}}
    _, _, _, est = sample_estimates(passes, 10, {"a": 100, "b": 100}, 150)
    calls = choose_plan(est, objective="calls", agreement_threshold=0.95)
    assert calls["chosen"] == "combined:a,b"  # 1 call per row beats 1.2
    with pytest.raises(ValueError, match="pricing"):
        choose_plan(est, objective="usd", agreement_threshold=0.95)
    _, _, _, priced = sample_estimates(
        passes, 10, {"a": 100, "b": 100}, 150, pricing=Pricing(1_000_000, 0)
    )
    usd = choose_plan(priced, objective="usd", agreement_threshold=0.95)
    assert usd["chosen"] == "sequential:b,a"
    assert usd["candidates"][0]["predicted_cost_per_row"] == pytest.approx(120)
    with pytest.raises(ValueError):
        choose_plan(est, objective="nope", agreement_threshold=0.5)


def test_predicted_vs_actual_and_regret():
    passes = {"a": set(range(8)), "b": {0, 1}}
    filters, records, probs, est = sample_estimates(
        passes, 10, {"a": 100, "b": 100}, 500
    )
    choice = choose_plan(est, objective="input_tokens", agreement_threshold=0.95)
    fake = FakeClient(probs, {"a": 100, "b": 100}, 500)
    plans = [Plan(c["strategy"], tuple(c["filters"])) for c in choice["candidates"]]
    execution = run_experiment(
        records, plans, client=fake, prompts={f: f"q {f}" for f in filters}
    )
    report = predicted_vs_actual(choice, execution)
    assert report["cheapest_measured_plan"] == "sequential:b,a"
    assert report["chosen_was_cheapest"] is True and report["regret"] == 0
    row = report["plans"]["sequential:b,a"]
    assert row["actual_cost_per_row"] == pytest.approx(120)
    assert row["relative_error"] == pytest.approx(0)


# ---- CLI ----


def write_input(tmp_path, passes, n):
    _, records, probs = make_records(passes, n)
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    prompts = {f: f"Does the text satisfy {f}?" for f in passes}
    task = tmp_path / "task.json"
    return path, probs, prompts, task


def install_cli_fake(monkeypatch, fake, prompts):
    class Managed:
        def __init__(self, **kw):
            assert kw["retry"] == {"max_retries": 0}

        def __enter__(self):
            return fake

        def __exit__(self, *a):
            return False

    monkeypatch.setitem(
        sys.modules,
        "typesafe_sdk",
        SimpleNamespace(
            Noul=lambda **kw: kw,
            RetryPolicy=lambda **kw: kw,
            TypeSafeClient=Managed,
            TypeSafeError=RuntimeError,
        ),
    )
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake")
    monkeypatch.setattr("jev_milestone_4.cli.FILTER_PROMPTS", prompts)


def test_cli_dry_run_reads_no_credentials(tmp_path, monkeypatch, capsys):
    passes = {"gratitude": set(range(8)), "admiration": {0, 1}}
    path, *_ = write_input(tmp_path, passes, 30)
    out = tmp_path / "out"
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    real_import, real_getenv = builtins.__import__, os.getenv

    def guarded_import(name, *a, **kw):
        if name.split(".")[0] in {"dotenv", "typesafe_sdk"}:
            pytest.fail(f"Dry run imported {name}")
        return real_import(name, *a, **kw)

    def guarded_getenv(name, *a, **kw):
        if name == "TYPESAFE_API_KEY":
            pytest.fail("Dry run read credentials")
        return real_getenv(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(os, "getenv", guarded_getenv)
    main(["--input", str(path), "--k", "10", "--output-dir", str(out), "--dry-run"])
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "records": 30,
        "sample_rows": 10,
        "execution_rows": 20,
        "sample_calls": 30,
        "execution_max_calls": 40,
        "worst_case_calls": 70,
    }
    assert not out.exists()


def test_cli_rejects_budget_and_bad_settings(tmp_path):
    passes = {"gratitude": set(range(8)), "admiration": {0, 1}}
    path, *_ = write_input(tmp_path, passes, 30)
    base = ["--input", str(path), "--dry-run"]
    for extra in (
        ["--max-api-calls", "10"],
        ["--k", "30"],
        ["--k", "0"],
        ["--objective", "usd"],
        ["--agreement-threshold", "2"],
    ):
        with pytest.raises(SystemExit):
            main(base + extra)


def test_cli_end_to_end_with_fake_client(tmp_path, monkeypatch):
    passes = {"gratitude": set(range(0, 30, 2)), "admiration": set(range(0, 30, 5))}
    path, probs, prompts, _ = write_input(tmp_path, passes, 30)
    fake = FakeClient(probs, {"gratitude": 100, "admiration": 100}, 500)
    install_cli_fake(monkeypatch, fake, prompts)
    out = tmp_path / "run"
    main(
        [
            "--input",
            str(path),
            "--k",
            "10",
            "--output-dir",
            str(out),
            "--evaluate-all-plans",
            "--input-usd-per-million",
            "1",
            "--output-usd-per-million",
            "0",
        ]
    )
    manifest = json.loads((out / "manifest.json").read_text())
    plan = json.loads((out / "plan.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    assert manifest["status"] == "complete"
    assert len(manifest["sample_ids"]) == 10
    assert plan["choice"]["chosen"] == "sequential:admiration,gratitude"
    assert summary["predicted_vs_actual"]["chosen_was_cheapest"] is True
    e2e = summary["optimizer_end_to_end"]
    assert e2e["rows"] == 30 and e2e["quality"]["f1"] == 1
    assert e2e["model_calls"] == summary["sampling_overhead"]["model_calls"] + sum(
        1
        for line in (out / "execution" / "calls.jsonl").read_text().splitlines()
        if json.loads(line)["plan"] == "sequential:admiration,gratitude"
    )
    for phase in ("sample", "execution"):
        for kind in ("calls", "decisions", "runs"):
            assert (out / phase / f"{kind}.jsonl").is_file()
    with pytest.raises(SystemExit):  # never overwrites
        main(["--input", str(path), "--k", "10", "--output-dir", str(out)])


def test_cli_plan_only_makes_only_sample_calls(tmp_path, monkeypatch):
    passes = {"gratitude": set(range(8)), "admiration": {0, 1}}
    path, probs, prompts, _ = write_input(tmp_path, passes, 30)
    fake = FakeClient(probs, {"gratitude": 100, "admiration": 100}, 500)
    install_cli_fake(monkeypatch, fake, prompts)
    out = tmp_path / "plan"
    main(["--input", str(path), "--k", "10", "--output-dir", str(out), "--plan-only"])
    assert len(fake.calls) == 30
    assert not (out / "execution").exists()
    assert "execution" not in json.loads((out / "summary.json").read_text())


def test_cli_default_mode_executes_only_chosen_plan(tmp_path, monkeypatch):
    passes = {"gratitude": set(range(0, 30, 2)), "admiration": set(range(0, 30, 5))}
    path, probs, prompts, _ = write_input(tmp_path, passes, 30)
    fake = FakeClient(probs, {"gratitude": 100, "admiration": 100}, 500)
    install_cli_fake(monkeypatch, fake, prompts)
    out = tmp_path / "chosen"
    main(["--input", str(path), "--k", "10", "--output-dir", str(out)])
    summary = json.loads((out / "summary.json").read_text())
    report = summary["predicted_vs_actual"]
    assert list(report["plans"]) == ["sequential:admiration,gratitude"]
    assert report["chosen_was_cheapest"] is None and report["regret"] is None
    assert summary["optimizer_end_to_end"]["rows"] == 30
    assert json.loads((out / "manifest.json").read_text())["mode"] == "execute_chosen"
