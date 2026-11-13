import pytest

from jev_milestone_3.sampling import (
    compare,
    draw_sample,
    estimate_filter,
    safe_binary_metrics,
    wilson_interval,
)


def make_rows(n=100, passes=20, gold_every=5, usage=True, cost=True, repeat=1):
    return [
        {
            "id": f"r{i:03d}",
            "filter": "acting_quality",
            "repeat": repeat,
            "status": "ok",
            "gold": int(i % gold_every == 0),
            "probability": 0.9 if i < passes else 0.1,
            "predicted": int(i < passes),
            "threshold": 0.5,
            "latency_ms": 100.0 + i,
            "input_tokens": 50 + i if usage else None,
            "output_tokens": 1 if usage else None,
            "estimated_cost_usd": (50 + i) * 1e-6 if usage and cost else None,
        }
        for i in range(n)
    ]


# draw_sample


def test_draw_sample_same_seed_same_ids_and_different_seeds_differ():
    records = [{"id": str(i)} for i in range(100)]
    first = [r["id"] for r in draw_sample(records, 10, 42)]
    assert first == [r["id"] for r in draw_sample(records, 10, 42)]
    assert first != [r["id"] for r in draw_sample(records, 10, 43)]


def test_draw_sample_no_duplicates_and_original_order():
    records = list(range(100))
    sample = draw_sample(records, 30, 7)
    assert len(sample) == len(set(sample)) == 30
    assert sample == sorted(sample)
    assert draw_sample(records, 100, 1) == records


@pytest.mark.parametrize("k", [0, -1, 101, 2.5])
def test_draw_sample_bad_k_raises(k):
    with pytest.raises(ValueError):
        draw_sample(list(range(100)), k, 0)


# wilson_interval


def test_wilson_hand_computed_values():
    low, high = wilson_interval(2, 10)
    assert low == pytest.approx(0.0567, abs=5e-4)
    assert high == pytest.approx(0.5098, abs=5e-4)
    low, high = wilson_interval(0, 10)
    assert low == 0.0
    assert high == pytest.approx(0.2775, abs=5e-4)
    assert wilson_interval(10, 10)[1] == 1.0


def test_wilson_bounds_within_unit_interval():
    for n in (1, 2, 5, 10, 37, 100):
        for s in range(n + 1):
            low, high = wilson_interval(s, n)
            assert 0.0 <= low <= s / n <= high <= 1.0


def test_wilson_rejects_bad_inputs():
    with pytest.raises(ValueError):
        wilson_interval(0, 0)
    with pytest.raises(ValueError):
        wilson_interval(3, 2)


# safe_binary_metrics


def test_safe_metrics_no_gold_positives_gives_null_recall():
    metrics = safe_binary_metrics([0, 0, 0], [1, 0, 0])
    assert metrics["recall"] is None
    assert metrics["precision"] == 0.0
    assert metrics["f1"] is None
    assert metrics["accuracy"] == pytest.approx(2 / 3)


def test_safe_metrics_no_predicted_positives_gives_null_precision():
    metrics = safe_binary_metrics([1, 0, 0], [0, 0, 0])
    assert metrics["precision"] is None
    assert metrics["recall"] == 0.0
    assert metrics["f1"] is None


def test_safe_metrics_defined_case_matches_binary_metrics():
    metrics = safe_binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0.5
    assert metrics["accuracy"] == 0.5


# estimate_filter


def test_estimate_selectivity_and_extrapolation():
    rows = make_rows()
    estimate = estimate_filter(rows, population_size=100)
    assert estimate["k"] == 100
    assert estimate["passes"] == 20
    assert estimate["selectivity"] == pytest.approx(0.2)
    assert estimate["gold_selectivity"] == pytest.approx(0.2)
    assert estimate["selectivity_ci_low"] < 0.2 < estimate["selectivity_ci_high"]
    mean_input = sum(r["input_tokens"] for r in rows) / 100
    mean_latency = sum(r["latency_ms"] for r in rows) / 100
    assert estimate["mean_input_tokens"] == pytest.approx(mean_input)
    assert estimate["mean_output_tokens"] == 1
    assert estimate["predicted_total_input_tokens"] == pytest.approx(100 * mean_input)
    assert estimate["predicted_total_latency_s"] == pytest.approx(
        100 * mean_latency / 1000
    )
    assert estimate["predicted_total_cost_usd"] == pytest.approx(
        100 * estimate["mean_cost_usd"]
    )


def test_estimate_extrapolates_sample_to_population():
    sample = make_rows()[:10]
    estimate = estimate_filter(sample, population_size=100)
    assert estimate["k"] == 10
    assert estimate["population_size"] == 100
    assert estimate["predicted_total_input_tokens"] == pytest.approx(
        100 * estimate["mean_input_tokens"]
    )


def test_estimate_missing_usage_gives_null_tokens_and_cost():
    rows = make_rows()
    rows[3] = {**rows[3], "input_tokens": None, "estimated_cost_usd": None}
    estimate = estimate_filter(rows, population_size=100)
    for key in (
        "mean_input_tokens",
        "mean_output_tokens",
        "mean_cost_usd",
        "predicted_total_input_tokens",
        "predicted_total_output_tokens",
        "predicted_total_cost_usd",
    ):
        assert estimate[key] is None, key
    assert estimate["mean_latency_ms"] is not None


def test_estimate_without_pricing_gives_null_cost_but_tokens():
    estimate = estimate_filter(make_rows(cost=False), population_size=100)
    assert estimate["mean_cost_usd"] is None
    assert estimate["predicted_total_cost_usd"] is None
    assert estimate["mean_input_tokens"] is not None


def test_estimate_zero_pass_sample_has_null_precision():
    rows = make_rows()[50:60]
    estimate = estimate_filter(rows, population_size=100)
    assert estimate["passes"] == 0
    assert estimate["selectivity_ci_low"] == 0.0
    assert estimate["precision"] is None


def test_estimate_rejects_mixed_or_error_rows():
    rows = make_rows(n=10)
    with pytest.raises(ValueError):
        estimate_filter(rows + make_rows(n=10, repeat=2), population_size=10)
    with pytest.raises(ValueError):
        estimate_filter(rows[:5] + rows[:1], population_size=10)
    with pytest.raises(ValueError):
        estimate_filter(
            rows[:9] + [{**rows[9], "status": "error", "error_type": "X"}],
            population_size=10,
        )
    with pytest.raises(ValueError):
        estimate_filter([], population_size=10)


# compare


def test_compare_errors_and_ci_containment():
    reference = estimate_filter(make_rows(), population_size=100)
    estimate = estimate_filter(make_rows()[:30], population_size=100)
    result = compare(estimate, reference)
    selectivity = result["quantities"]["selectivity"]
    assert selectivity["estimate"] == pytest.approx(20 / 30)
    assert selectivity["abs_error"] == pytest.approx(20 / 30 - 0.2)
    assert selectivity["rel_error"] == pytest.approx((20 / 30 - 0.2) / 0.2)
    assert result["ci_contains_reference"] is False
    assert compare(reference, reference)["ci_contains_reference"] is True


def test_compare_relative_error_null_when_reference_zero_or_missing():
    reference = estimate_filter(make_rows(passes=0), population_size=100)
    estimate = estimate_filter(make_rows(passes=0)[:10], population_size=100)
    estimate["selectivity"] = 0.1
    quantities = compare(estimate, reference)["quantities"]
    assert quantities["selectivity"]["abs_error"] == pytest.approx(0.1)
    assert quantities["selectivity"]["rel_error"] is None
    assert quantities["precision"]["abs_error"] is None  # undefined on both sides
