"""Overlay entry point, sharing the existing experiment engine and artifacts."""

from typing import Sequence

from .cli import main as experiment_main


def main(argv: Sequence[str] | None = None) -> None:
    experiment_main(argv, comparison_only=True)


if __name__ == "__main__":
    main()
