"""Validated task definitions and provenance for text-only benchmark imports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import re

from jev_milestone_1.common import canonical_json, digest_file, read_jsonl, write_json


def load_task(path: Path) -> dict:
    task = json.loads(path.read_text(encoding="utf-8"))
    return validate_task(task)


def validate_task(task: dict) -> dict:
    """Validate an in-memory definition without creating or modifying files."""
    if not isinstance(task, dict):
        raise ValueError("Task must be a JSON object")
    dataset = task.get("dataset")
    if not isinstance(dataset, dict) or any(
        not isinstance(dataset.get(k), str) or not dataset[k].strip()
        for k in ("name", "scenario", "revision", "source_url")
    ):
        raise ValueError(
            "Task dataset needs name, scenario, revision, and source_url provenance"
        )
    for key in ("id_column", "text_column"):
        if not isinstance(task.get(key), str) or not task[key].strip():
            raise ValueError(f"Task requires {key}")
    predicates = task.get("predicates")
    if not isinstance(predicates, dict) or len(predicates) < 2:
        raise ValueError(
            "Task requires at least two individually labeled semantic predicates"
        )
    for name, definition in predicates.items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", name) or not isinstance(
            definition, dict
        ):
            raise ValueError("Invalid predicate name or definition")
        if definition.get("kind") != "semantic":
            raise ValueError(
                "This comparison requires semantic predicates; structured filters need a separate baseline"
            )
        for key in ("prompt", "label_column", "label_source"):
            if not isinstance(definition.get(key), str) or not definition[key].strip():
                raise ValueError(
                    f"Predicate {name} requires {key}; unverified labels are not ground truth"
                )
        positive, negative = (
            definition.get("positive_values"),
            definition.get("negative_values"),
        )
        if any(
            not isinstance(values, list)
            or not values
            or any(not isinstance(v, str) for v in values)
            for values in (positive, negative)
        ) or set(positive) & set(negative):
            raise ValueError(
                f"Predicate {name} requires nonempty, disjoint string label mappings"
            )
    label_columns = {p["label_column"] for p in predicates.values()}
    if task["text_column"] in label_columns:
        raise ValueError("Text column cannot be a gold-label column")
    if len(label_columns) != len(predicates):
        raise ValueError("Distinct semantic predicates require distinct label columns")
    return task


def task_prompts(task: dict) -> dict[str, str]:
    return {
        name: definition["prompt"] for name, definition in task["predicates"].items()
    }


def check_sample_manifest(path: Path, task: dict | None = None) -> dict | None:
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Sample manifest must be a JSON object")
    if manifest.get("sample_sha256") and manifest["sample_sha256"] != digest_file(path):
        raise ValueError("Input checksum differs from source sample manifest")
    if "task" in manifest and manifest["task"] != task:
        raise ValueError("Pass the matching --task definition for this imported sample")
    return manifest


def convert_records(rows: list[dict], task: dict) -> list[dict]:
    """Map verified annotations; send only the text field to the model."""
    records, seen = [], set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Source row {index} must be an object")
        identity, text = row.get(task["id_column"]), row.get(task["text_column"])
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError(f"Source row {index} needs a unique nonempty string id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Source row {index} needs nonempty text")
        seen.add(identity)
        gold = {}
        for name, definition in task["predicates"].items():
            value = row.get(definition["label_column"])
            if value is None:
                raise ValueError(f"Source row {index} has no verified label for {name}")
            value = str(value)
            if (
                value
                not in definition["positive_values"] + definition["negative_values"]
            ):
                raise ValueError(f"Source row {index} has an unmapped label for {name}")
            gold[name] = int(value in definition["positive_values"])
        records.append(
            {"id": identity, "text": text, "gold": gold, "source_row": index}
        )
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Import a verified SemBench-derived text task from local CSV/JSONL"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.output.suffix != ".jsonl":
        parser.error("--output must end in .jsonl")
    manifest_path = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest_path.exists():
        parser.error("Choose new output paths; existing samples are never overwritten")
    try:
        task = load_task(args.task)
        if task["dataset"]["name"] != "SemBench":
            raise ValueError("This importer requires dataset.name = SemBench")
        if args.source.suffix == ".csv":
            with args.source.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
        elif args.source.suffix == ".jsonl":
            rows = read_jsonl(args.source)
        else:
            raise ValueError(
                "Source must be a local .csv or .jsonl export with verified label columns"
            )
        records = convert_records(rows, task)
        if not 1 <= args.limit <= len(records):
            raise ValueError(f"--limit must be between 1 and {len(records)}")
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    indices = sorted(random.Random(args.seed).sample(range(len(records)), args.limit))
    selected = [records[i] for i in indices]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.writelines(canonical_json(record) + "\n" for record in selected)
    write_json(
        manifest_path,
        {
            "dataset_repo": task["dataset"]["source_url"],
            "dataset": task["dataset"],
            "role": "research benchmark candidate; label provenance supplied by task author",
            "task": task,
            "task_sha256": digest_file(args.task),
            "source_file": str(args.source),
            "source_sha256": digest_file(args.source),
            "source_rows": len(records),
            "sample_rows": len(selected),
            "sample_seed": args.seed,
            "sampling_method": "uniform without replacement, sorted by source row",
            "sample_sha256": digest_file(args.output),
            "gold_positive_counts": {
                name: sum(r["gold"][name] for r in selected)
                for name in task["predicates"]
            },
            "conjunction_positive_count": sum(
                all(r["gold"].values()) for r in selected
            ),
        },
    )
    print(f"Imported {len(selected)} labeled records to {args.output}")


if __name__ == "__main__":
    main()
