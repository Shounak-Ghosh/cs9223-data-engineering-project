from jev_milestone_1.prepare import sample_records


def test_sampling_is_stable_and_binary():
    rows = [{"text": f"text {i}", "gratitude": i % 2, "admiration": i % 3 == 0} for i in range(30)]
    a = sample_records(rows, limit=8, seed=42, labels=["gratitude", "admiration"])
    b = sample_records(rows, limit=8, seed=42, labels=["gratitude", "admiration"])
    assert a == b
    assert len({r["id"] for r in a}) == 8
    assert all(set(r["gold"]) == {"gratitude", "admiration"} for r in a)
