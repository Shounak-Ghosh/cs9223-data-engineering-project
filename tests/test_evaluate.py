from types import SimpleNamespace
import sys

from jev_milestone_1.evaluate import run_individual_filters


class FakeClient:
    def __init__(self):
        self.calls = []

    def system_one(self, *, state, questions):
        assert len(questions) == 1, "Milestone-1 calls must contain a single filter"
        name = next(iter(questions))
        self.calls.append((state, name))
        probability = 0.8 if "Thank" in state else 0.1
        return SimpleNamespace(
            model="jev-1.14.0",
            nouls={name: SimpleNamespace(noul=probability)},
            usage=SimpleNamespace(input_tokens=15, output_tokens=3),
            request_id=None,
        )


def test_individual_filter_calls_are_independent(monkeypatch):
    # Inject a lightweight SDK stub: the test makes zero network calls.
    monkeypatch.setitem(sys.modules, "typesafe_sdk", SimpleNamespace(Noul=lambda **kwargs: kwargs))
    recs = [{"id": "x", "text": "Thank you!", "gold": {"gratitude": 1}},
            {"id": "y", "text": "Fine.", "gold": {"gratitude": 0}}]
    fake = FakeClient()
    result = run_individual_filters(recs, ["gratitude"], repeats=2, threshold=.5, client=fake)
    assert len(fake.calls) == 4
    assert [r["predicted"] for r in result] == [1, 0, 1, 0]
