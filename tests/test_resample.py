import builtins
import json
import sys

import pytest

from jev_milestone_3.cli import analysis_main
from jev_milestone_3.resample import repeat_consistency, resampling_analysis


def make_full_rows(n=40, repeats=2, flip_ids=()):
    rows = []
    for repeat in range(1, repeats + 1):
        for i in range(n):
            passed = i % 5 == 0
            if repeat > 1 and f"r{i:03d}" in flip_ids:
                passed = not passed
            rows.append(
                {
                    "id": f"r{i:03d}",
                    "filter": "acting_quality",
                    "repeat": repeat,
                    "status": "ok",
                    "gold": int(i % 4 == 0),
                    "probability": (0.8 if passed else 0.2) + 0.01 * (repeat - 1),
                    "predicted": int(passed),
                    "threshold": 0.5,
                    "latency_ms": 100.0 + (i % 7) * 10,
                    "input_tokens": 50 + i,
                    "output_tokens": 1,
                    "estimated_cost_usd": None,
                }
            )
    return rows


def test_deterministic_for_fixed_seed():
    rows = make_full_rows()
    a = resampling_analysis(rows, sizes=[5, 10], draws=200, seed=3, population_size=40)
    b = resampling_analysis(rows, sizes=[5, 10], draws=200, seed=3, population_size=40)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    c = resampling_analysis(rows, sizes=[5, 10], draws=200, seed=4, population_size=40)
    assert a["per_k"] != c["per_k"]


def test_full_population_sample_has_zero_error_and_full_coverage():
    rows = make_full_rows()
    result = resampling_analysis(rows, sizes=[40], draws=20, seed=0, population_size=40)
    row = result["per_k"][0]
    assert row["mean_abs_error_selectivity"] == 0
    assert row["max_abs_error_selectivity"] == 0
    assert row["ci_coverage"] == 1
    assert row["mean_rel_error_input_tokens"] == 0
    assert row["mean_rel_error_latency"] == 0
    assert row["mean_rel_error_cost"] is None  # no pricing -> unknown, not 0
    assert row["std_f1"] == 0


def test_aggregates_are_valid_fractions():
    result = resampling_analysis(
        make_full_rows(), sizes=[5, 10, 20], draws=300, seed=0, population_size=40
    )
    assert result["reference"]["selectivity"] == pytest.approx(0.2)
    for row in result["per_k"]:
        for key in (
            "ci_coverage",
            "zero_pass_fraction",
            "undefined_precision_fraction",
            "undefined_recall_fraction",
            "f1_defined_fraction",
        ):
            assert 0 <= row[key] <= 1, key
        assert row["mean_abs_error_selectivity"] <= row["max_abs_error_selectivity"]
        assert row["p90_abs_error_selectivity"] <= row["max_abs_error_selectivity"]
    small, large = result["per_k"][0], result["per_k"][-1]
    assert small["zero_pass_fraction"] > large["zero_pass_fraction"]
    assert small["mean_ci_width"] > large["mean_ci_width"]


def test_uses_repeat_one_only_for_reference_and_draws():
    rows = make_full_rows(flip_ids={"r001", "r002"})
    result = resampling_analysis(rows, sizes=[40], draws=1, seed=0, population_size=40)
    assert result["reference"]["passes"] == 8
    assert result["reference"]["k"] == 40


def test_repeat_consistency():
    rows = make_full_rows(n=10, flip_ids={"r001"})
    consistency = repeat_consistency(rows)
    assert consistency["decision_consistency"] == pytest.approx(0.9)
    assert consistency["n_inconsistent_records"] == 1
    assert consistency["max_probability_range"] == pytest.approx(0.61)
    assert repeat_consistency(make_full_rows(repeats=1)) is None


def test_rejects_bad_sizes():
    with pytest.raises(ValueError):
        resampling_analysis(make_full_rows(), sizes=[41], draws=1, population_size=40)
    with pytest.raises(ValueError):
        resampling_analysis(make_full_rows(), sizes=[0], draws=1, population_size=40)


# CLI


def write_run(directory, rows, *, status="complete", sample_size="all", seed=None):
    from jev_milestone_1.common import canonical_json, digest_file

    directory.mkdir(parents=True)
    calls = directory / "calls.jsonl"
    calls.write_text("".join(canonical_json(r) + "\n" for r in rows))
    ids = sorted({r["id"] for r in rows})
    manifest = {
        "kind": "filter_profile",
        "status": status,
        "filter": "acting_quality",
        "prompt": "Does this movie review comment on the quality of the acting?",
        "threshold": 0.5,
        "input_sha256": "abc",
        "sample_size": sample_size,
        "seed": seed,
        "selected_ids": ids,
        "n_input_records": 40,
        "output_sha256": {"calls.jsonl": digest_file(calls)},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def analysis_args(full, output, *extra):
    return [
        "--full-run",
        str(full),
        "--output-dir",
        str(output),
        "--sizes",
        "5",
        "10",
        "40",
        "--draws",
        "50",
        *extra,
    ]


def test_analysis_cli_writes_outputs_without_sdk(tmp_path, monkeypatch, capsys):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"dotenv", "typesafe_sdk"}:
            pytest.fail(f"Offline analysis imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "typesafe_sdk", raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    full_rows = make_full_rows(flip_ids={"r003"})
    full = write_run(tmp_path / "full", full_rows)
    sample_ids = {"r000", "r001", "r005", "r010", "r020"}
    live_rows = [
        {**r, "probability": 0.55, "predicted": 1} if r["id"] == "r001" else r
        for r in full_rows
        if r["repeat"] == 1 and r["id"] in sample_ids
    ]
    live = write_run(tmp_path / "k5", live_rows, sample_size=5, seed=42)
    output = tmp_path / "analysis"
    analysis_main(analysis_args(full, output, "--live-sample", str(live)))

    analysis = json.loads((output / "analysis.json").read_text())
    assert [row["k"] for row in analysis["per_k"]] == [5, 10, 40]
    assert analysis["per_k"][-1]["mean_abs_error_selectivity"] == 0
    assert analysis["repeat_consistency"]["decision_consistency"] == pytest.approx(
        39 / 40
    )
    live_result = analysis["live_sample"]
    assert live_result["estimate"]["passes"] == 5
    assert live_result["full_run_same_ids_estimate"]["passes"] == 4
    assert live_result["decision_agreement_with_full_run"] == pytest.approx(0.8)
    assert live_result["vs_full_reference"]["quantities"]["selectivity"][
        "reference"
    ] == pytest.approx(0.2)
    csv_lines = (output / "analysis.csv").read_text().splitlines()
    assert csv_lines[0].startswith("k,draws,mean_abs_error_selectivity")
    assert len(csv_lines) == 4
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["api_calls"] == 0 and manifest["draws"] == 50
    assert set(manifest["output_sha256"]) == {"analysis.json", "analysis.csv"}
    out = capsys.readouterr().out
    assert "ci_cov" in out and "zero_pass" in out


def test_analysis_refuses_incomplete_full_run(tmp_path):
    full = write_run(tmp_path / "full", make_full_rows(), status="failed")
    with pytest.raises(SystemExit):
        analysis_main(analysis_args(full, tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_analysis_refuses_sampled_full_run(tmp_path):
    full = write_run(tmp_path / "full", make_full_rows(), sample_size=10, seed=42)
    with pytest.raises(SystemExit):
        analysis_main(analysis_args(full, tmp_path / "out"))


def test_analysis_refuses_tampered_calls(tmp_path):
    full = write_run(tmp_path / "full", make_full_rows())
    with (full / "calls.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(SystemExit):
        analysis_main(analysis_args(full, tmp_path / "out"))


def test_analysis_refuses_mismatched_live_sample(tmp_path):
    full = write_run(tmp_path / "full", make_full_rows())
    other = [{**r, "filter": "positive"} for r in make_full_rows(repeats=1)[:5]]
    live = write_run(tmp_path / "k5", other, sample_size=5, seed=1)
    manifest = json.loads((live / "manifest.json").read_text())
    manifest["filter"] = "positive"
    (live / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit):
        analysis_main(analysis_args(full, tmp_path / "out", "--live-sample", str(live)))


def test_analysis_refuses_existing_output(tmp_path):
    full = write_run(tmp_path / "full", make_full_rows())
    (tmp_path / "out").mkdir()
    with pytest.raises(SystemExit):
        analysis_main(analysis_args(full, tmp_path / "out"))


# Ordering agreement


def make_b_rows(n=40):
    """A second, less selective filter (pass rate 0.5) on the same IDs."""
    return [
        {**r, "filter": "positive", "predicted": int(i % 2 == 0)}
        for i, r in enumerate(make_full_rows(n=n, repeats=1))
    ]


def test_ordering_full_population_agrees_with_zero_regret():
    from jev_milestone_3.resample import ordering_agreement

    a = [r for r in make_full_rows() if r["repeat"] == 1]
    result = ordering_agreement(
        a, make_b_rows(), sizes=[40], draws=5, seed=0, population_size=40
    )
    assert result["reference"]["best_order"] == ["acting_quality", "positive"]
    assert result["cost_basis"] == "mean_input_tokens"  # no pricing in fixtures
    row = result["per_k"][0]
    assert row["selectivity_order_agreement"] == row["plan_agreement"] == 1
    assert row["mean_regret"] == row["max_regret"] == 0
    assert result["reference"]["wrong_order_regret"] > 0


def test_ordering_is_deterministic_and_bounded():
    from jev_milestone_3.resample import ordering_agreement

    args = (make_full_rows(), make_b_rows())
    kwargs = {"sizes": [5, 10], "draws": 200, "seed": 1, "population_size": 40}
    first = ordering_agreement(*args, **kwargs)
    assert first == ordering_agreement(*args, **kwargs)
    for row in first["per_k"]:
        for key in ("selectivity_order_agreement", "plan_agreement"):
            assert 0 <= row[key] <= 1
        assert 0 <= row["mean_regret"] <= row["max_regret"]
    assert first["per_k"][0]["plan_agreement"] < 1  # k=5 sometimes misorders


def test_ordering_rejects_same_filter_or_different_ids():
    from jev_milestone_3.resample import ordering_agreement

    with pytest.raises(ValueError):
        ordering_agreement(
            make_full_rows(), make_full_rows(), sizes=[5], draws=1, population_size=40
        )
    with pytest.raises(ValueError):
        ordering_agreement(
            make_full_rows(), make_b_rows()[:-1], sizes=[5], draws=1, population_size=40
        )


def test_ordering_cli(tmp_path, capsys):
    from jev_milestone_3.cli import ordering_main

    full_a = write_run(tmp_path / "a", make_full_rows())
    full_b = write_run(tmp_path / "b", make_b_rows())
    manifest = json.loads((full_b / "manifest.json").read_text())
    (full_b / "manifest.json").write_text(
        json.dumps({**manifest, "filter": "positive"})
    )
    output = tmp_path / "ordering"
    ordering_main(
        [
            "--full-runs",
            str(full_a),
            str(full_b),
            "--sizes",
            "5",
            "40",
            "--draws",
            "20",
            "--output-dir",
            str(output),
        ]
    )
    result = json.loads((output / "ordering.json").read_text())
    assert result["per_k"][-1]["plan_agreement"] == 1
    assert len((output / "ordering.csv").read_text().splitlines()) == 3
    assert json.loads((output / "manifest.json").read_text())["api_calls"] == 0
    assert "plan_agree" in capsys.readouterr().out
    sampled = write_run(tmp_path / "s", make_b_rows(), sample_size=10, seed=1)
    with pytest.raises(SystemExit):
        ordering_main(
            [
                "--full-runs",
                str(full_a),
                str(sampled),
                "--output-dir",
                str(tmp_path / "x"),
            ]
        )
