"""Shared file helpers for the annotation tools."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from jev_milestone_1.common import read_jsonl


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    raise ValueError(f"{path} must be a .csv or .jsonl file")


def write_csv_new(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Write a CSV, refusing to overwrite an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )


def write_text_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def require_new_dir(path: Path) -> None:
    if path.exists():
        raise ValueError(f"{path} already exists; outputs are never overwritten")


def unique_ids(rows: list[dict], id_column: str) -> list[str]:
    ids = []
    for index, row in enumerate(rows):
        value = row.get(id_column)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Row {index} needs a nonempty string {id_column}")
        ids.append(value)
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate {id_column} values found")
    return ids
