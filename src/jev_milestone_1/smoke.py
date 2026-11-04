"""Minimal live key/API connectivity check: exactly one Jev request."""

from __future__ import annotations

import os


def main() -> None:
    from dotenv import load_dotenv
    from typesafe_sdk import Noul, TypeSafeClient
    from .config import MODEL

    load_dotenv()
    if not os.getenv("TYPESAFE_API_KEY"):
        raise SystemExit(
            "TYPESAFE_API_KEY missing; export it or put it in a local .env file"
        )
    with TypeSafeClient(model=MODEL) as client:
        response = client.system_one(
            state="Thank you so much for your help!",
            questions={
                "gratitude": Noul(
                    instructions="Does this text express gratitude or thanks?"
                )
            },
        )
    print(f"Model: {response.model}")
    print(f"P(gratitude): {response.nouls['gratitude'].noul:.4f}")
    print(f"Input tokens: {response.usage.input_tokens}")
    print("API smoke test passed (request completed and a valid Noul was returned).")


if __name__ == "__main__":
    main()
