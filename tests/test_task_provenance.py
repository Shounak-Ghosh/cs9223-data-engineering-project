"""Bind task drafts to real workflow artifacts; never label human examples here."""

import json
from argparse import Namespace

import pytest

from jev_milestone_1.common import digest_file
from jev_milestone_2.tools import agreement, build_task
from test_tools import TEMPLATE, fill, read_csv, run_sheets, source, write_csv


@pytest.fixture
def artifacts(tmp_path, source):
    sheet_dir = tmp_path / "sheets"
    run_sheets(source, sheet_dir)
    paths = [fill(sheet_dir, name, lambda r: "1" if "acting" in r["reviewText"] else "0") for name in ("A", "B")]
    output = tmp_path / "agreement"
    agreement.main(["--sheets", *map(str, paths), "--source", str(source), "--output-dir", str(output)])
    template = tmp_path / "template.json"
    template.write_text(json.dumps(TEMPLATE))
    fetch = tmp_path / "fetch.json"
    fetch.write_text(json.dumps({"revision": "fixture-revision", "source_url": "https://example.invalid/actual-source",
                                 "output_sha256": digest_file(source), "rows": 40,
                                 "id_column": "reviewId", "label_column": "scoreSentiment"}))
    return Namespace(template=template, fetch_json=fetch, agreement_json=output / "agreement.json",
                     sheet_manifest=sheet_dir / "sheet_manifest.json", dataset=output / "annotated.csv",
                     output=tmp_path / "new-dir" / "task.json", predicate_name="acting_quality",
                     prompt="Does this movie review comment on the quality of the acting or performances?",
                     scenario="movie_with_additional_annotations")


def change(path, key, value):
    payload = json.loads(path.read_text())
    payload[key] = value
    path.write_text(json.dumps(payload))


def argv(args):
    return ["--template", str(args.template), "--fetch-json", str(args.fetch_json),
            "--agreement-json", str(args.agreement_json), "--sheet-manifest", str(args.sheet_manifest),
            "--dataset", str(args.dataset), "--output", str(args.output)]


def test_build_is_pure_and_preserves_preexisting_validation_file(artifacts):
    # The old implementation would overwrite and delete this unrelated file.
    artifacts.output.parent.mkdir()
    scratch = artifacts.output.with_suffix(".validating.json")
    scratch.write_bytes(b"existing file must survive")
    task = build_task.build(artifacts)
    assert scratch.read_bytes() == b"existing file must survive"
    assert not artifacts.output.exists()
    assert task["dataset"]["source_url"] == "https://example.invalid/actual-source"
    assert task["provenance"]["annotated_sha256"] == digest_file(artifacts.dataset)
    assert task["provenance"]["agreement_report_sha256"] == digest_file(artifacts.agreement_json)
    build_task.main(argv(artifacts))
    assert artifacts.output.exists()
    assert scratch.read_bytes() == b"existing file must survive"


def test_build_creates_output_parent_only_after_validation(artifacts):
    build_task.main(argv(artifacts))
    before = artifacts.output.read_bytes()
    with pytest.raises(SystemExit):
        build_task.main(argv(artifacts))
    assert artifacts.output.read_bytes() == before


@pytest.mark.parametrize("report,key,value,match", [
    ("agreement_json", "annotated_sha256", "0" * 64, "Annotated dataset checksum"),
    ("agreement_json", "source_sha256", "0" * 64, "Agreement source checksum"),
    ("fetch_json", "output_sha256", "0" * 64, "Agreement source checksum"),
    ("agreement_json", "rows", 999, "row count"),
    ("agreement_json", "selected_ids", ["unrelated-id"], "Agreement selected IDs"),
    ("sheet_manifest", "selected_ids", ["unrelated-id"], "Sheet selected IDs"),
    ("sheet_manifest", "source_sha256", "0" * 64, "Sheet source checksum"),
    ("sheet_manifest", "annotators", 3, "annotator counts"),
    ("sheet_manifest", "prompt", "Different predicate?", "annotation sheet protocol"),
    ("fetch_json", "rows", 1, "Fetch row count"),
    ("fetch_json", "source_url", "", "requires source_url"),
    ("agreement_json", "id_column", "other", "ID columns"),
    ("agreement_json", "final_positive", 999, "positive-label count"),
    ("agreement_json", "disagreement_ids", ["unrelated-id"], "adjudication counts or IDs"),
    ("agreement_json", "adjudicated", 999, "adjudication counts or IDs"),
])
def test_provenance_mismatches_are_rejected_before_writes(artifacts, report, key, value, match):
    change(getattr(artifacts, report), key, value)
    with pytest.raises(ValueError, match=match):
        build_task.build(artifacts)
    assert not artifacts.output.parent.exists()


def test_modified_csv_rejected(artifacts):
    artifacts.dataset.write_text(artifacts.dataset.read_text() + "\n")
    with pytest.raises(ValueError, match="Annotated dataset checksum"):
        build_task.build(artifacts)


def test_legacy_agreement_needs_sheet_ids(artifacts):
    report = json.loads(artifacts.agreement_json.read_text())
    del report["selected_ids"]
    artifacts.agreement_json.write_text(json.dumps(report))
    assert build_task.build(artifacts)
    artifacts.sheet_manifest = None
    with pytest.raises(ValueError, match="lacks selected_ids"):
        build_task.build(artifacts)


def test_report_with_selected_ids_works_without_sheet_manifest(artifacts):
    artifacts.sheet_manifest = None
    assert build_task.build(artifacts)


def test_duplicate_annotated_ids_rejected_even_with_matching_file_checksum(artifacts):
    rows = read_csv(artifacts.dataset)
    rows[1]["reviewId"] = rows[0]["reviewId"]
    write_csv(artifacts.dataset, rows)
    change(artifacts.agreement_json, "annotated_sha256", digest_file(artifacts.dataset))
    with pytest.raises(ValueError, match="Duplicate"):
        build_task.build(artifacts)
