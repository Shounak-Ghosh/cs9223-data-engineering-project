"""Synthetic fixtures only: no downloaded benchmark rows or generated gold labels."""

import copy
import json

import pytest

from jev_milestone_1.common import digest_file
from jev_milestone_2.datasets import check_sample_manifest, convert_records, load_task, main


@pytest.fixture
def task():
    return {
        "dataset": {"name": "SemBench", "scenario": "synthetic_adapter_test",
                    "revision": "offline-fixture-v1", "source_url": "https://example.invalid/test-fixture"},
        "id_column": "reviewId", "text_column": "reviewText",
        "predicates": {
            "positive": {"kind": "semantic", "prompt": "Is this a positive review?",
                         "label_column": "scoreSentiment", "positive_values": ["POSITIVE"],
                         "negative_values": ["NEGATIVE"], "label_source": "Synthetic test fixture sentiment"},
            "mentions_acting": {"kind": "semantic", "prompt": "Does this review discuss acting?",
                                "label_column": "acting_label", "positive_values": ["1"],
                                "negative_values": ["0"], "label_source": "Synthetic fixture manual labels"},
        },
    }


@pytest.fixture
def source_rows():
    return [
        {"reviewId": str(i), "reviewText": text, "scoreSentiment": sentiment, "acting_label": acting}
        for i, (text, sentiment, acting) in enumerate([
            ("Great acting!", "POSITIVE", "1"), ("Wonderful music!", "POSITIVE", "0"),
            ("Poor acting!", "NEGATIVE", "1"), ("Boring movie.", "NEGATIVE", "0"),
        ])
    ]


def write_task(tmp_path, task):
    path = tmp_path / "task.json"
    path.write_text(json.dumps(task))
    return path


def test_import_csv_reproducible_with_provenance_and_no_label_leakage(tmp_path, task, source_rows):
    import csv

    source = tmp_path / "reviews.csv"
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)
    task_path = write_task(tmp_path, task)
    outputs = [tmp_path / "a.jsonl", tmp_path / "b.jsonl"]
    for output in outputs:
        main(["--source", str(source), "--task", str(task_path), "--output", str(output), "--limit", "4"])
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    records = [json.loads(line) for line in outputs[0].read_text().splitlines()]
    assert [r["gold"] for r in records] == [
        {"positive": 1, "mentions_acting": 1}, {"positive": 1, "mentions_acting": 0},
        {"positive": 0, "mentions_acting": 1}, {"positive": 0, "mentions_acting": 0},
    ]
    assert all(set(r) == {"id", "text", "gold", "source_row"} for r in records)
    assert [r["text"] for r in records] == [r["reviewText"] for r in source_rows]
    manifest = check_sample_manifest(outputs[0], task)
    assert manifest["source_sha256"] == digest_file(source)
    assert manifest["conjunction_positive_count"] == 1
    assert manifest["task"] == task
    with pytest.raises(ValueError, match="matching --task"):
        check_sample_manifest(outputs[0])
    outputs[0].write_text(outputs[0].read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        check_sample_manifest(outputs[0], task)


@pytest.mark.parametrize("mutation", [
    lambda t: t["predicates"].pop("mentions_acting"),
    lambda t: t["predicates"]["mentions_acting"].update(label_source=""),
    lambda t: t["predicates"]["mentions_acting"].update(kind="structured"),
    lambda t: t["predicates"]["mentions_acting"].update(negative_values=["1"]),
    lambda t: t.update(text_column="scoreSentiment"),
    lambda t: t["dataset"].update(revision=""),
])
def test_unverified_or_invalid_tasks_rejected(tmp_path, task, mutation):
    mutation(task)
    with pytest.raises(ValueError):
        load_task(write_task(tmp_path, task))


def test_missing_second_label_never_inferred(task, source_rows):
    source_rows[0].pop("acting_label")
    with pytest.raises(ValueError, match="no verified label"):
        convert_records(source_rows, task)


def test_reject_unmapped_labels_and_duplicate_ids(task, source_rows):
    original = copy.deepcopy(source_rows)
    source_rows[0]["scoreSentiment"] = "UNKNOWN"
    with pytest.raises(ValueError, match="unmapped"):
        convert_records(source_rows, task)
    with pytest.raises(ValueError, match="unique"):
        convert_records(original + original, task)


def test_import_jsonl_and_preserve_existing_outputs(tmp_path, task, source_rows):
    source = tmp_path / "reviews.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in source_rows))
    output = tmp_path / "sample.jsonl"
    args = ["--source", str(source), "--task", str(write_task(tmp_path, task)),
            "--output", str(output), "--limit", "2"]
    main(args)
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        main(args)
    assert output.read_bytes() == before
