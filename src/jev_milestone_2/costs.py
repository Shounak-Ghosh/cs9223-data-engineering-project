"""Optional request-byte/question credit estimates, independent of token usage."""

from dataclasses import dataclass
import math

from jev_milestone_1.common import canonical_json


@dataclass(frozen=True)
class QuestionPricing:
    credits_per_question_per_block: float
    bytes_per_block: int = 4096
    usd_per_credit: float | None = None

    def __post_init__(self) -> None:
        if type(self.bytes_per_block) is not int or self.bytes_per_block < 1:
            raise ValueError("bytes_per_block must be a positive integer")
        for value in (self.credits_per_question_per_block, self.usd_per_credit):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("Credit rates must be finite and nonnegative")

    def estimate_request(
        self, *, state: str, names: tuple[str, ...], prompts: dict[str, str], model: str
    ) -> dict:
        body = {
            "model": model,
            "state": state,
            "questions": {
                name: {"type": "noul", "instructions": prompts[name]} for name in names
            },
        }
        size = len(canonical_json(body).encode("utf-8"))
        blocks = max(1, math.ceil(size / self.bytes_per_block))
        credits = len(names) * blocks * self.credits_per_question_per_block
        return {
            "estimated_request_bytes": size,
            "estimated_billing_blocks": blocks,
            "estimated_credits": credits,
            "estimated_cost_usd": credits * self.usd_per_credit
            if self.usd_per_credit is not None
            else None,
        }
