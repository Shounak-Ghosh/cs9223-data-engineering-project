import builtins
import json
import os
import sys
from types import SimpleNamespace

import pytest

from jev_milestone_2.cli import main
from jev_milestone_2.experiment import Plan, Pricing, run_experiment, summarize_experiment


FILTERS = ("gratitude", "admiration")
PLANS = [Plan("independent", FILTERS), Plan("combined", FILTERS),
         Plan("sequential", FILTERS), Plan("sequential", FILTERS[::-1])]
RECORDS = [
    {"id": "both", "text": "both", "gold": {"gratitude": 1, "admiration": 1}},
    {"id": "g", "text": "g", "gold": {"gratitude": 1, "admiration": 0}},
    {"id": "neither", "text": "neither", "gold": {"gratitude": 0, "admiration": 0}},
]


class FakeClient:
    def __init__(self, *, combined_shift=False, missing_usage=False):
        self.calls = []
        self.combined_shift = combined_shift
        self.missing_usage = missing_usage

    def system_one(self, *, state, questions):
        self.calls.append((state, tuple(questions)))
        values = {"both": (.5, .9), "g": (.8, .1), "neither": (.1, .1)}[state]
        answers = dict(zip(FILTERS, values))
        if self.combined_shift and len(questions) > 1:
            answers["admiration"] = .1
        return SimpleNamespace(
            model="fake-model", request_id=f"r{len(self.calls)}",
            nouls={name: SimpleNamespace(noul=answers[name]) for name in questions},
            usage=None if self.missing_usage else SimpleNamespace(input_tokens=100, output_tokens=10),
        )


@pytest.fixture(autouse=True)
def stub_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "typesafe_sdk", SimpleNamespace(Noul=lambda **kwargs: kwargs, TypeSafeError=RuntimeError))


def test_plans_short_circuit_and_accounting():
    fake = FakeClient()
    result = run_experiment(RECORDS, PLANS, client=fake, repeats=2, pricing=Pricing(2, 3))
    summary = summarize_experiment(result)
    assert len(fake.calls) == summary["total_model_calls"] == 36  # 2 * (6 + 3 + 5 + 4)
    assert len(result["decisions"]) == 24
    assert [r["plan"] for r in result["runs"][4:]] == [p.name for p in PLANS[1:] + PLANS[:1]]
    for plan in PLANS:
        stats = summary["plans"][plan.name]
        assert stats["repeat_consistency"] == stats["agreement_with_independent"] == 1
        assert stats["repeats"][0]["quality"]["f1"] == 1
        assert stats["runtime_seconds"] >= stats["api_latency_seconds"] >= 0
    combined = summary["plans"][PLANS[1].name]
    assert combined["model_calls"] == 6
    assert combined["semantic_questions"] == 12
    assert combined["input_tokens"] == 600  # Not doubled for two questions.
    assert combined["estimated_cost_usd"] == pytest.approx(.00138)
    forward = summary["plans"][PLANS[2].name]["repeats"][0]
    reverse = summary["plans"][PLANS[3].name]["repeats"][0]
    assert forward["stages"][1]["input_rows"] == 2
    assert reverse["stages"][1]["input_rows"] == 1
    skipped = next(d for d in result["decisions"] if d["plan"] == PLANS[2].name and d["id"] == "neither")
    assert skipped["skipped_filters"] == ["admiration"]
    assert "admiration" not in skipped["probabilities"]
    assert skipped["predicted"] == 0


def test_combined_quality_can_differ_from_independent():
    result = run_experiment(RECORDS, PLANS, client=FakeClient(combined_shift=True))
    combined = summarize_experiment(result)["plans"][PLANS[1].name]
    assert combined["agreement_with_independent"] == pytest.approx(2 / 3)
    assert combined["repeats"][0]["quality"]["fn"] == 1
    assert combined["repeat_consistency"] is None


def test_unknown_cost_and_usage_are_not_zero():
    result = run_experiment(RECORDS, PLANS, client=FakeClient())
    assert summarize_experiment(result)["total_estimated_cost_usd"] is None
    result = run_experiment(RECORDS, PLANS, client=FakeClient(missing_usage=True), pricing=Pricing(2, 0))
    stats = summarize_experiment(result)["plans"][PLANS[0].name]
    assert stats["input_tokens"] is stats["output_tokens"] is stats["estimated_cost_usd"] is None


@pytest.mark.parametrize("records,plans,repeats,threshold", [
    ([], PLANS, 1, .5), (RECORDS, PLANS, 0, .5), (RECORDS, PLANS, 1, float("nan")),
    (RECORDS * 2, PLANS, 1, .5), (RECORDS, [Plan("combined", ("gratitude",) * 2)], 1, .5),
    (RECORDS, [PLANS[0], PLANS[0]], 1, .5),
    (RECORDS, [PLANS[0], Plan("sequential", ("gratitude", "anger"))], 1, .5),
    ([{"id": "x", "text": "x", "gold": {"gratitude": 1}}], PLANS, 1, .5),
])
def test_invalid_experiments_make_no_calls(records, plans, repeats, threshold):
    fake = FakeClient()
    with pytest.raises(ValueError):
        run_experiment(records, plans, repeats=repeats, threshold=threshold, client=fake)
    assert not fake.calls


def test_three_filters_stop_at_middle_failure():
    filters = FILTERS + ("optimism",)
    record = {**RECORDS[1], "gold": {**RECORDS[1]["gold"], "optimism": 1}}
    result = run_experiment([record], [Plan("sequential", filters)], client=FakeClient())
    assert len(result["calls"]) == 2
    assert result["decisions"][0]["skipped_filters"] == ["optimism"]
    quality = summarize_experiment(result)["plans"][f"sequential:{','.join(filters)}"]["repeats"][0]["quality"]
    assert quality["gold_positive_count"] == 0
    assert quality["quality_note"]


@pytest.mark.parametrize("rate", [-1, float("inf"), float("nan")])
def test_invalid_prices(rate):
    with pytest.raises(ValueError):
        Pricing(rate, 0)


def install_cli_fake(monkeypatch, fake):
    class ManagedClient:
        def __init__(self, **kwargs):
            assert kwargs["retry"] == {"max_retries": 0}

        def __enter__(self):
            return fake

        def __exit__(self, *args):
            return False

    monkeypatch.setitem(sys.modules, "typesafe_sdk", SimpleNamespace(
        Noul=lambda **kwargs: kwargs, RetryPolicy=lambda **kwargs: kwargs, TypeSafeClient=ManagedClient,
        TypeSafeError=RuntimeError))
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-test-key")


def cli_input(tmp_path):
    source = tmp_path / "sample.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in RECORDS))
    return source


@pytest.mark.parametrize("extra,filters,bounds", [
    ([], FILTERS, [(12, 12), (6, 6), (6, 12), (6, 12)]),
    (["--max-records", "1", "--repeats", "3"], FILTERS,
     [(6, 6), (3, 3), (3, 6), (3, 6)]),
    (["--filters", "gratitude", "admiration", "optimism",
      "--order", "optimism", "gratitude", "admiration"],
     FILTERS + ("optimism",), [(18, 18), (6, 6), (6, 18)]),
])
@pytest.mark.parametrize("existing_output", [False, True])
def test_cli_dry_run_bounds_without_credentials_or_sdk(
    tmp_path, monkeypatch, capsys, extra, filters, bounds, existing_output,
):
    source = tmp_path / "sample.jsonl"
    records = [{**r, "gold": {**r["gold"], "optimism": 1}} for r in RECORDS]
    source.write_text("\n".join(json.dumps(r) for r in records))
    output = tmp_path / "experiment"
    if existing_output:
        output.mkdir()
        (output / "sentinel.txt").write_text("preserve existing results")

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    original_import = builtins.__import__
    original_getenv = os.getenv

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"dotenv", "typesafe_sdk"}:
            pytest.fail(f"Dry run imported {name}")
        return original_import(name, *args, **kwargs)

    def guarded_getenv(name, *args, **kwargs):
        if name == "TYPESAFE_API_KEY":
            pytest.fail("Dry run read API credentials")
        return original_getenv(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(os, "getenv", guarded_getenv)
    main(["--input", str(source), "--output-dir", str(output), "--dry-run", *extra])

    report = json.loads(capsys.readouterr().out)
    assert report["records"] == (1 if "--max-records" in extra else 3)
    assert report["repeats"] == (3 if "--repeats" in extra else 2)
    orders = [("optimism", "gratitude", "admiration")] if "--order" in extra else [filters, filters[::-1]]
    plans = [Plan("independent", filters), Plan("combined", filters)]
    plans.extend(Plan("sequential", order) for order in orders)
    assert report["call_bounds"] == [
        {"plan": plan.name, "min_calls": lower, "max_calls": upper}
        for plan, (lower, upper) in zip(plans, bounds)
    ]
    if existing_output:
        assert list(output.iterdir()) == [output / "sentinel.txt"]
        assert (output / "sentinel.txt").read_text() == "preserve existing results"
    else:
        assert not output.exists()


def test_cli_dry_run_and_outputs(tmp_path, monkeypatch):
    fake = FakeClient()
    install_cli_fake(monkeypatch, fake)
    source = cli_input(tmp_path)
    output = tmp_path / "experiment"
    args = ["--input", str(source), "--output-dir", str(output), "--repeats", "1"]
    main(args + ["--dry-run"])
    assert not output.exists() and not fake.calls
    main(args + ["--input-usd-per-million", "2", "--output-usd-per-million", "0"])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["resolved_models"] == ["fake-model"]
    assert manifest["pricing"]["input_per_million"] == 2
    assert len((output / "calls.jsonl").read_text().splitlines()) == 18
    assert json.loads((output / "summary.json").read_text())["total_estimated_cost_usd"] == pytest.approx(.0036)
    with pytest.raises(SystemExit):
        main(args)
    assert len(fake.calls) == 18


def test_cli_preserves_partial_calls_on_failure(tmp_path, monkeypatch):
    class FailingClient(FakeClient):
        def system_one(self, **kwargs):
            if self.calls:
                raise RuntimeError("offline failure")
            return super().system_one(**kwargs)

    install_cli_fake(monkeypatch, FailingClient())
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError):
        main(["--input", str(cli_input(tmp_path)), "--output-dir", str(output)])
    events = [json.loads(line) for line in (output / "calls.jsonl").read_text().splitlines()]
    assert [event["status"] for event in events] == ["ok", "error"]
    assert events[-1]["error_type"] == "RuntimeError"
    assert "offline failure" not in json.dumps(events[-1])
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
    assert not (output / "summary.json").exists()


@pytest.mark.parametrize("extra", [
    ["--order", "gratitude", "gratitude"], ["--order", "gratitude", "anger"],
    ["--input-usd-per-million", "1"], ["--output-usd-per-million", "nan", "--input-usd-per-million", "1"],
])
def test_cli_rejects_invalid_settings(tmp_path, extra):
    with pytest.raises(SystemExit):
        main(["--input", str(cli_input(tmp_path)), "--dry-run", *extra])
