from jev_milestone_1.common import binary_metrics, summarize


def test_confusion_metrics():
    got = binary_metrics([1, 0, 1, 0], [1, 1, 0, 0])
    assert (got["tp"], got["fp"], got["fn"], got["tn"]) == (1, 1, 1, 1)
    assert got["precision"] == got["recall"] == got["f1"] == 0.5


def test_repeat_agreement():
    rows = [
        {"filter": "gratitude", "id": "a", "repeat": 1, "gold": 1, "predicted": 1, "probability": .9, "latency_ms": 12, "input_tokens": 10},
        {"filter": "gratitude", "id": "b", "repeat": 1, "gold": 0, "predicted": 0, "probability": .2, "latency_ms": 10, "input_tokens": 11},
        {"filter": "gratitude", "id": "a", "repeat": 2, "gold": 1, "predicted": 1, "probability": .8, "latency_ms": 10, "input_tokens": 12},
        {"filter": "gratitude", "id": "b", "repeat": 2, "gold": 0, "predicted": 1, "probability": .6, "latency_ms": 10, "input_tokens": 13},
    ]
    got = summarize(rows, ["gratitude"], .5)["filters"]["gratitude"]
    assert got["repeat_consistency"] == .5
    assert got["f1"] == 1.0  # First pass was perfect.
    assert got["api_calls"] == 4
