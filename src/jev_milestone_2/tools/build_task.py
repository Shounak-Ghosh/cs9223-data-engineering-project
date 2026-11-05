"""Fill the Movie task template from fetch and agreement outputs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re

from jev_milestone_1.common import digest_file
from jev_milestone_2.datasets import convert_records, validate_task
from jev_milestone_2.tools.common import read_rows, unique_ids, write_json_new

PLACEHOLDER = "second_semantic_predicate"


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def describe_protocol(agreement: dict, sheet_manifest: dict | None) -> str:
    kappas = [p["cohen_kappa"] for p in agreement["pairwise"].values()]
    parts = [
        f"Human annotation by {len(agreement['annotators'])} independent annotators "
        f"({', '.join(agreement['annotators'])}) of {agreement['rows']} reviews",
        "pairwise Cohen kappa " + ", ".join(fmt(k) for k in kappas),
    ]
    if agreement.get("fleiss_kappa") is not None:
        parts.append(f"Fleiss kappa {fmt(agreement['fleiss_kappa'])}")
    parts.append(
        f"{agreement['adjudicated']} disagreements adjudicated"
        if agreement["adjudicated"]
        else "no disagreements"
    )
    if sheet_manifest:
        sampling = sheet_manifest["sampling"]
        parts.append(f"sampling: {sampling['mode']} (seed {sheet_manifest['seed']})")
        if sampling.get("mode") == "keyword-oversample":
            parts.append(f"keyword fraction {sampling['keyword_fraction']}; sample is not representative of the source")
    return "; ".join(parts) + ". Derived annotation, not part of the official SemBench query."


def require_digest(value: str, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Missing or invalid {field}; regenerate the provenance report")
    return value


def verify_ids(values: list[str], expected: set[str], field: str) -> None:
    if (not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values)
            or len(set(values)) != len(values) or set(values) != expected):
        raise ValueError(f"{field} do not match the annotated dataset IDs")


def verify_provenance(rows: list[dict], dataset: Path, fetch: dict, agreement: dict,
                      sheets: dict | None, predicate_name: str, prompt: str) -> str:
    """Bind source, selected sample and agreement evidence to the actual output."""
    if require_digest(agreement.get("annotated_sha256"), "agreement.annotated_sha256") != digest_file(dataset):
        raise ValueError("Annotated dataset checksum differs from agreement.annotated_sha256")
    source_sha = require_digest(fetch.get("output_sha256"), "fetch.output_sha256")
    if require_digest(agreement.get("source_sha256"), "agreement.source_sha256") != source_sha:
        raise ValueError("Agreement source checksum differs from fetched output checksum")
    if type(agreement.get("rows")) is not int or agreement["rows"] != len(rows) or not rows:
        raise ValueError("Agreement row count differs from annotated dataset row count")
    if type(fetch.get("rows")) is not int or fetch["rows"] < len(rows):
        raise ValueError("Fetch row count is smaller than the annotated sample")
    for field in ("revision", "source_url", "id_column", "label_column"):
        if not isinstance(fetch.get(field), str) or not fetch[field].strip():
            raise ValueError(f"Fetch report requires {field}")
    id_column = fetch["id_column"]
    if agreement.get("id_column", id_column) != id_column:
        raise ValueError("Agreement and fetch ID columns differ")
    ids = set(unique_ids(rows, id_column))
    if "selected_ids" in agreement:
        verify_ids(agreement["selected_ids"], ids, "Agreement selected IDs")
    elif sheets is None:
        raise ValueError("Agreement lacks selected_ids; regenerate agreement or supply --sheet-manifest")
    annotators = agreement.get("annotators")
    if (not isinstance(annotators, list) or len(annotators) < 2
            or any(not isinstance(a, str) or not a for a in annotators)
            or len(set(annotators)) != len(annotators)):
        raise ValueError("Agreement requires at least two distinct annotators")
    if sheets is not None:
        if require_digest(sheets.get("source_sha256"), "sheet_manifest.source_sha256") != source_sha:
            raise ValueError("Sheet source checksum differs from fetched output checksum")
        verify_ids(sheets.get("selected_ids"), ids, "Sheet selected IDs")
        if sheets.get("annotators") != len(annotators):
            raise ValueError("Sheet and agreement annotator counts differ")
        if sheets.get("id_column", id_column) != id_column:
            raise ValueError("Sheet and fetch ID columns differ")
        if sheets.get("predicate") != predicate_name or sheets.get("prompt") != prompt:
            raise ValueError("Requested predicate/prompt differs from annotation sheet protocol")
    return id_column


def build(args: argparse.Namespace) -> dict:
    task = copy.deepcopy(json.loads(args.template.read_text(encoding="utf-8")))
    fetch = json.loads(args.fetch_json.read_text(encoding="utf-8"))
    agreement = json.loads(args.agreement_json.read_text(encoding="utf-8"))
    sheet_manifest = (
        json.loads(args.sheet_manifest.read_text(encoding="utf-8"))
        if args.sheet_manifest
        else None
    )
    label_column = agreement["label_column"]
    rows = read_rows(args.dataset)
    id_column = verify_provenance(rows, args.dataset, fetch, agreement, sheet_manifest,
                                  args.predicate_name, args.prompt)
    if not rows or label_column not in rows[0]:
        raise ValueError(f"{args.dataset} has no {label_column} column")
    if {r[label_column] for r in rows} - {"0", "1"}:
        raise ValueError(f"{label_column} must contain only 0 and 1")
    if agreement.get("final_positive") != sum(r[label_column] == "1" for r in rows):
        raise ValueError("Agreement positive-label count differs from annotated dataset")
    disagreements = agreement.get("disagreement_ids")
    if (not isinstance(disagreements, list) or any(not isinstance(i, str) for i in disagreements)
            or len(set(disagreements)) != len(disagreements)
            or not set(disagreements) <= {r[id_column] for r in rows}
            or agreement.get("adjudicated") != len(disagreements)):
        raise ValueError("Agreement disagreement/adjudication counts or IDs are inconsistent")
    if PLACEHOLDER not in task["predicates"]:
        raise ValueError(f"Template has no {PLACEHOLDER} entry")
    if args.predicate_name in task["predicates"]:
        raise ValueError(f"Predicate {args.predicate_name} already exists in template")
    task["dataset"]["revision"] = fetch["revision"]
    task["dataset"]["source_url"] = fetch["source_url"]
    task["dataset"]["scenario"] = args.scenario
    task["id_column"] = id_column
    if sheet_manifest and "text_column" in sheet_manifest:
        task["text_column"] = sheet_manifest["text_column"]
    task["predicates"]["positive"]["label_column"] = fetch["label_column"]
    predicates = {}
    for name, definition in task["predicates"].items():
        if name == PLACEHOLDER:
            predicates[args.predicate_name] = {
                **definition,
                "prompt": args.prompt,
                "label_column": label_column,
                "label_source": describe_protocol(agreement, sheet_manifest),
            }
        else:
            predicates[name] = definition
    task["predicates"] = predicates
    task["provenance"] = {
        "fetch_report_sha256": digest_file(args.fetch_json),
        "agreement_report_sha256": digest_file(args.agreement_json),
        "fetched_source_sha256": fetch["output_sha256"],
        "annotated_sha256": agreement["annotated_sha256"],
        "selected_ids": sorted(r[id_column] for r in rows),
    }
    if args.sheet_manifest:
        task["provenance"]["sheet_manifest_sha256"] = digest_file(args.sheet_manifest)
    # Pure validation: no predictable scratch path and no filesystem mutations.
    validate_task(task)
    convert_records(rows, task)
    return task


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--fetch-json", type=Path, required=True)
    parser.add_argument("--agreement-json", type=Path, required=True)
    parser.add_argument("--sheet-manifest", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, required=True, help="annotated.csv")
    parser.add_argument("--predicate-name", default="acting_quality")
    parser.add_argument("--prompt", default="Does this movie review comment on the quality of the acting or performances?")
    parser.add_argument("--scenario", default="movie_with_additional_annotations")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Choose a new --output; existing files are never overwritten")
    try:
        task = build(args)
        write_json_new(args.output, task)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(f"Wrote {args.output}; review label_source before use")


if __name__ == "__main__":
    main()
