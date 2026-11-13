import builtins
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from jev_milestone_2.experiment import Pricing
from jev_milestone_3.cli import profile_main
from jev_milestone_3.profile import run_single_filter
from jev_milestone_3.sampling import draw_sample

ROOT = Path(__file__).resolve().parents[1]
REAL_INPUT = ROOT / "data" / "sembench_movie_verified.jsonl"
REAL_TASK = ROOT / "data" / "verified_task.json"
ACTING_PROMPT = json.loads(REAL_TASK.read_text())["predicates"]["acting_quality"][
    "prompt"
]
RECORDS = [
    {
        "id": f"r{i:02d}",
        "text": f"review text {i}",
        "gold": {"acting_quality": int(i % 5 == 0), "positive": int(i % 2 == 0)},
    }
    for i in range(20)
]


class FakeClient:
    def __init__(self, *, fail_on=None, missing_usage=False):
        self.calls = []
        self.fail_on = fail_on
        self.missing_usage = missing_usage

    def system_one(self, *, state, questions):
        self.calls.append((state, questions))
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise RuntimeError("secret message that must not be stored")
        index = int(state.rsplit(" ", 1)[1])
        name = next(iter(questions))
        return SimpleNamespace(
            model="fake-model",
            request_id=f"req{len(self.calls)}",
            nouls={name: SimpleNamespace(noul=0.9 if index % 4 == 0 else 0.1)},
            usage=None
            if self.missing_usage
            else SimpleNamespace(input_tokens=40 + index, output_tokens=1),
        )


@pytest.fixture(autouse=True)
def stub_sdk(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "typesafe_sdk",
        SimpleNamespace(Noul=lambda **kwargs: kwargs, TypeSafeError=RuntimeError),
    )


def run(client, **kwargs):
    options = dict(
        filter_name="acting_quality", prompt=ACTING_PROMPT, client=client, repeats=1
    )
    options.update(kwargs)
    return run_single_filter(RECORDS, **options)


# run_single_filter


def test_one_question_per_call_and_records_times_repeats():
    client = FakeClient()
    rows = run(client, repeats=3)
    assert len(client.calls) == len(rows) == len(RECORDS) * 3
    for _, questions in client.calls:
        assert len(questions) == 1
        assert list(questions) == ["acting_quality"]
    assert [r["repeat"] for r in rows] == [1] * 20 + [2] * 20 + [3] * 20


def test_task_prompt_reaches_client_and_gold_never_in_state():
    client = FakeClient()
    run(client)
    for (state, questions), record in zip(client.calls, RECORDS):
        assert questions["acting_quality"] == {"instructions": ACTING_PROMPT}
        assert state == record["text"]
        assert isinstance(state, str)
        assert "gold" not in state and "acting_quality" not in state


def test_row_fields_and_token_pricing():
    rows = run(FakeClient(), pricing=Pricing(1_000_000, 0), threshold=0.5)
    row = rows[0]
    for key in (
        "id",
        "filter",
        "repeat",
        "gold",
        "probability",
        "predicted",
        "threshold",
        "model",
        "request_id",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "estimated_cost_usd",
    ):
        assert key in row, key
    assert row["id"] == "r00" and row["gold"] == 1 and row["predicted"] == 1
    assert row["model"] == "fake-model" and row["latency_ms"] >= 0
    assert row["estimated_cost_usd"] == pytest.approx(40)
    assert sum(r["predicted"] for r in rows) == 5


def test_question_pricing_uses_single_filter_name():
    from jev_milestone_2.costs import QuestionPricing

    rows = run(FakeClient(), pricing=QuestionPricing(2.0, 4096, 0.5))
    assert rows[0]["estimated_credits"] == 2.0
    assert rows[0]["estimated_cost_usd"] == 1.0


def test_missing_usage_gives_null_tokens_and_cost():
    rows = run(FakeClient(missing_usage=True), pricing=Pricing(1, 0))
    assert rows[0]["input_tokens"] is None
    assert rows[0]["estimated_cost_usd"] is None


def test_error_row_recorded_then_reraised():
    events = []
    with pytest.raises(RuntimeError):
        run(FakeClient(fail_on=3), on_event=events.append)
    assert len(events) == 3
    error = events[-1]
    assert error["status"] == "error"
    assert error["error_type"] == "RuntimeError"
    assert error["id"] == "r02"
    assert "secret" not in json.dumps(error)


def test_rejects_missing_gold():
    records = [*RECORDS, {"id": "x", "text": "t", "gold": {"positive": 1}}]
    with pytest.raises(ValueError):
        run_single_filter(
            records, filter_name="acting_quality", prompt="p", client=FakeClient()
        )


# CLI


def install_cli_fake(monkeypatch, fake):
    class ManagedClient:
        def __init__(self, **kwargs):
            assert kwargs["retry"] == {"max_retries": 0}

        def __enter__(self):
            return fake

        def __exit__(self, *args):
            return False

    monkeypatch.setitem(
        sys.modules,
        "typesafe_sdk",
        SimpleNamespace(
            Noul=lambda **kwargs: kwargs,
            RetryPolicy=lambda **kwargs: kwargs,
            TypeSafeClient=ManagedClient,
            TypeSafeError=RuntimeError,
        ),
    )
    monkeypatch.setitem(
        sys.modules, "dotenv", SimpleNamespace(load_dotenv=lambda: None)
    )
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-test-key")


def small_input(tmp_path):
    source = tmp_path / "records.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n")
    return source


def guard_no_credentials(monkeypatch):
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


def real_args(output, *extra):
    return [
        "--input",
        str(REAL_INPUT),
        "--task",
        str(REAL_TASK),
        "--filter",
        "acting_quality",
        "--output-dir",
        str(output),
        *extra,
    ]


def test_dry_run_full_reads_no_credentials_and_writes_nothing(
    tmp_path, monkeypatch, capsys
):
    output = tmp_path / "full"
    guard_no_credentials(monkeypatch)
    profile_main(
        real_args(output, "--repeats", "2", "--max-api-calls", "200", "--dry-run")
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["planned_calls"] == 200
    assert plan["sample_size"] == "all"
    assert len(plan["selected_ids"]) == 100
    assert plan["prompt"] == ACTING_PROMPT
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_dry_run_sample_prints_seeded_ids(tmp_path, monkeypatch, capsys):
    output = tmp_path / "sample"
    guard_no_credentials(monkeypatch)
    profile_main(real_args(output, "--sample-size", "10", "--seed", "42", "--dry-run"))
    plan = json.loads(capsys.readouterr().out)
    records = [json.loads(line) for line in REAL_INPUT.read_text().splitlines()]
    assert plan["selected_ids"] == [r["id"] for r in draw_sample(records, 10, 42)]
    assert plan["planned_calls"] == 10
    assert not output.exists()


def test_budget_exceeded_errors_before_any_call(tmp_path, monkeypatch):
    fake = FakeClient()
    install_cli_fake(monkeypatch, fake)
    output = tmp_path / "full"
    with pytest.raises(SystemExit):
        profile_main(real_args(output, "--repeats", "2"))  # 200 > default 100
    assert fake.calls == []
    assert not output.exists()


def test_existing_output_dir_refused(tmp_path, monkeypatch):
    fake = FakeClient()
    install_cli_fake(monkeypatch, fake)
    output = tmp_path / "full"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    with pytest.raises(SystemExit):
        profile_main(
            [
                "--input",
                str(small_input(tmp_path)),
                "--task",
                str(REAL_TASK),
                "--filter",
                "acting_quality",
                "--output-dir",
                str(output),
            ]
        )
    assert fake.calls == []
    assert [p.name for p in output.iterdir()] == ["keep.txt"]


def test_unknown_filter_refused(tmp_path, monkeypatch, capsys):
    guard_no_credentials(monkeypatch)
    with pytest.raises(SystemExit):
        profile_main(
            [
                "--input",
                str(REAL_INPUT),
                "--task",
                str(REAL_TASK),
                "--filter",
                "nonexistent",
                "--dry-run",
            ]
        )
    assert "Unknown --filter" in capsys.readouterr().err


def test_seed_without_sample_size_refused(tmp_path):
    with pytest.raises(SystemExit):
        profile_main(real_args(tmp_path / "x", "--seed", "1", "--dry-run"))


def test_cli_full_run_writes_outputs(tmp_path, monkeypatch):
    fake = FakeClient()
    install_cli_fake(monkeypatch, fake)
    output = tmp_path / "full"
    profile_main(
        [
            "--input",
            str(small_input(tmp_path)),
            "--task",
            str(REAL_TASK),
            "--filter",
            "acting_quality",
            "--repeats",
            "2",
            "--output-dir",
            str(output),
            "--input-usd-per-million",
            "0.042",
            "--output-usd-per-million",
            "0",
        ]
    )
    assert len(fake.calls) == 40
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["sample_size"] == "all" and manifest["seed"] is None
    assert manifest["sdk_max_retries"] == 0
    assert manifest["resolved_models"] == ["fake-model"]
    assert manifest["prompt"] == ACTING_PROMPT
    assert set(manifest["output_sha256"]) == {"calls.jsonl", "estimate.json"}
    assert len((output / "calls.jsonl").read_text().splitlines()) == 40
    estimate = json.loads((output / "estimate.json").read_text())
    assert estimate["k"] == 20 and estimate["population_size"] == 20
    assert estimate["selectivity"] == pytest.approx(0.25)
    assert estimate["gold_selectivity"] == pytest.approx(0.2)
    assert estimate["mean_cost_usd"] is not None


def test_cli_sample_estimate_uses_full_population(tmp_path, monkeypatch):
    fake = FakeClient()
    install_cli_fake(monkeypatch, fake)
    output = tmp_path / "k5"
    profile_main(
        [
            "--input",
            str(small_input(tmp_path)),
            "--task",
            str(REAL_TASK),
            "--filter",
            "acting_quality",
            "--sample-size",
            "5",
            "--output-dir",
            str(output),
        ]
    )
    assert len(fake.calls) == 5
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["sample_size"] == 5 and manifest["seed"] == 42
    assert manifest["selected_ids"] == [r["id"] for r in draw_sample(RECORDS, 5, 42)]
    estimate = json.loads((output / "estimate.json").read_text())
    assert estimate["k"] == 5 and estimate["population_size"] == 20


def test_cli_preserves_partial_calls_on_failure(tmp_path, monkeypatch):
    fake = FakeClient(fail_on=4)
    install_cli_fake(monkeypatch, fake)
    output = tmp_path / "full"
    with pytest.raises(RuntimeError):
        profile_main(
            [
                "--input",
                str(small_input(tmp_path)),
                "--task",
                str(REAL_TASK),
                "--filter",
                "acting_quality",
                "--output-dir",
                str(output),
            ]
        )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "RuntimeError"
    assert manifest["completed_calls"] == 3
    rows = [
        json.loads(line) for line in (output / "calls.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 4 and rows[-1]["error_type"] == "RuntimeError"
    assert not (output / "estimate.json").exists()
