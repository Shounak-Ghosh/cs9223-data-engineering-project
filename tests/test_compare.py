import csv
import json

import pytest

from jev_milestone_2.costs import QuestionPricing
from jev_milestone_2.experiment import Plan, run_experiment, summarize_experiment
from jev_milestone_2.runner import main
from test_experiment import FakeClient, FILTERS, RECORDS, install_cli_fake
from test_datasets import task, source_rows, write_task


def test_overlay_three_plans_export_csv_and_credit_costs(tmp_path, monkeypatch):
    fake = FakeClient()
    install_cli_fake(monkeypatch, fake)
    source = tmp_path / "sample.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in RECORDS))
    output = tmp_path / "compare.jsonl"
    main(["--input", str(source), "--output", str(output),
          "--credits-per-question-per-block", ".25", "--usd-per-credit", ".02",
          "--pricing-source", "synthetic unit-test rate; not a provider price"])
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert len(summary["plans"]) == 3
    assert all(not name.startswith("independent") for name in summary["plans"])
    assert len(fake.calls) == 12  # 3 combined + 5 forward + 4 reverse
    assert summary["plans"]["combined:gratitude,admiration"]["input_tokens"] == 300
    assert summary["plans"]["combined:gratitude,admiration"]["estimated_credits"] == 1.5
    assert summary["total_estimated_cost_usd"] == pytest.approx(.075)
    with output.with_suffix(".summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert all(row["n"] == "3" for row in rows)
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["pricing_basis"] == "question_request_blocks"
    assert manifest["worst_case_model_calls"] == 15
    assert manifest["requested_call_budget"] == 100


def test_question_billing_block_boundary_and_utf8():
    pricing = QuestionPricing(.25, bytes_per_block=10**6, usd_per_credit=.02)
    prompts = {"a": "question A", "b": "question B"}
    kwargs = dict(state="café", names=("a", "b"), prompts=prompts, model="fake")
    cost = pricing.estimate_request(**kwargs)
    size = cost["estimated_request_bytes"]
    assert QuestionPricing(.25, size).estimate_request(**kwargs)["estimated_credits"] == .5
    assert QuestionPricing(.25, size - 1).estimate_request(**kwargs)["estimated_credits"] == 1
    assert cost["estimated_cost_usd"] == .01


@pytest.mark.parametrize("rate", [-1, float("nan"), float("inf")])
def test_nonfinite_credit_prices_rejected(rate):
    with pytest.raises(ValueError):
        QuestionPricing(rate)
    with pytest.raises(ValueError):
        QuestionPricing(1, usd_per_credit=rate)


@pytest.mark.parametrize("extra", [
    ["--max-api-calls", "1"], ["--usd-per-credit", "1"],
    ["--credits-per-question-per-block", "nan"],
    ["--credits-per-question-per-block", "1", "--input-usd-per-million", "1", "--output-usd-per-million", "0"],
])
def test_bad_budget_or_pricing_makes_no_calls(tmp_path, monkeypatch, extra):
    fake = FakeClient()
    install_cli_fake(monkeypatch, fake)
    source = tmp_path / "sample.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in RECORDS))
    with pytest.raises(SystemExit):
        main(["--input", str(source), "--dry-run", *extra])
    assert not fake.calls


def test_custom_benchmark_prompts_reach_model_without_labels(tmp_path, monkeypatch, task, source_rows):
    class BenchmarkFake:
        def __init__(self):
            self.calls = []

        def system_one(self, *, state, questions):
            from types import SimpleNamespace
            self.calls.append((state, questions))
            return SimpleNamespace(model="offline", request_id=None,
                                   usage=SimpleNamespace(input_tokens=10, output_tokens=2),
                                   nouls={name: SimpleNamespace(noul=.9) for name in questions})

    fake = BenchmarkFake()
    install_cli_fake(monkeypatch, fake)
    source = tmp_path / "sample.jsonl"
    source.write_text(json.dumps({"id": "x", "text": "Great acting!", "gold": {"positive": 1, "mentions_acting": 1}}))
    task_path = write_task(tmp_path, task)
    output = tmp_path / "benchmark"
    main(["--input", str(source), "--task", str(task_path), "--output-dir", str(output)])
    assert len(fake.calls) == 5
    assert all(state == "Great acting!" for state, _ in fake.calls)
    for _, questions in fake.calls:
        for name, question in questions.items():
            assert question["instructions"] == task["predicates"][name]["prompt"]
    assert json.loads((output / "manifest.json").read_text())["task"] == task


def test_credit_estimates_remain_available_without_usage(monkeypatch):
    install_cli_fake(monkeypatch, FakeClient())
    result = run_experiment(RECORDS, [Plan("combined", FILTERS)], client=FakeClient(missing_usage=True),
                            pricing=QuestionPricing(.5))
    stats = summarize_experiment(result)["plans"]["combined:gratitude,admiration"]
    assert stats["input_tokens"] is None
    assert stats["estimated_credits"] == 3
    assert stats["estimated_cost_usd"] is None
