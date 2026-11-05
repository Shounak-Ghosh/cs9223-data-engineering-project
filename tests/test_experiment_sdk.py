"""Exercise real SDK serialization over an offline HTTP transport."""

import json

import httpx2
import pytest
from typesafe_sdk import RetryPolicy, TypeSafeClient, TypeSafeInternalServerError

from jev_milestone_1.config import FILTER_PROMPTS, MODEL
from jev_milestone_2.experiment import Plan, Pricing, run_experiment


@pytest.mark.parametrize("request_id", [None, "offline-request-1"])
def test_combined_real_sdk_request_and_usage(request_id):
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        return httpx2.Response(200, headers={"x-typesafe-request-id": request_id} if request_id else {}, json={
            "model": MODEL, "usage": {"input_tokens": 123, "output_tokens": 4},
            "answers": {name: {"type": "noul", "noul": .9} for name in body["questions"]},
        })

    filters = ("gratitude", "admiration")
    with TypeSafeClient(api_key="offline-test-key", model=MODEL,
                        base_url="https://offline.invalid", compression=False,
                        transport=httpx2.MockTransport(respond), retry=RetryPolicy(max_retries=0)) as client:
        result = run_experiment(
            [{"id": "x", "text": "Thank you, wonderful work!", "gold": dict.fromkeys(filters, 1)}],
            [Plan("combined", filters)], client=client, pricing=Pricing(2, 0),
        )
    assert len(requests) == 1
    assert requests[0]["state"] == "Thank you, wonderful work!"
    assert requests[0]["model"] == MODEL
    for name in filters:
        assert requests[0]["questions"][name]["type"] == "noul"
        assert requests[0]["questions"][name]["instructions"] == FILTER_PROMPTS[name]
    assert result["calls"][0]["input_tokens"] == 123
    assert result["calls"][0]["request_id"] == request_id
    assert result["calls"][0]["estimated_cost_usd"] == pytest.approx(.000246)
    assert result["decisions"][0]["predicted"] == 1


def test_sdk_does_not_retry_failed_model_calls():
    requests = []

    def fail(request):
        requests.append(request)
        return httpx2.Response(500, json={"error": "offline simulated failure"})

    with TypeSafeClient(api_key="offline-test-key", model=MODEL,
                        base_url="https://offline.invalid", compression=False,
                        transport=httpx2.MockTransport(fail), retry=RetryPolicy(max_retries=0)) as client:
        with pytest.raises(TypeSafeInternalServerError):
            run_experiment(
                [{"id": "x", "text": "x", "gold": {"gratitude": 0, "admiration": 0}}],
                [Plan("combined", ("gratitude", "admiration"))], client=client,
            )
    assert len(requests) == 1
