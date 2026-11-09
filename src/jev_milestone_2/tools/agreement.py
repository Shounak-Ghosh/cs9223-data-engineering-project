"""Validate completed annotation sheets, measure agreement, and merge final labels."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

from jev_milestone_1.common import digest_file
from jev_milestone_2.tools.common import (
    read_rows,
    require_new_dir,
    unique_ids,
    write_csv_new,
    write_json_new,
)

VALID = {"0", "1"}


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((a.count(c) / n) * (b.count(c) / n) for c in VALID)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def fleiss_kappa(columns: list[list[str]]) -> float | None:
    raters, n = len(columns), len(columns[0])
    counts = [[sum(col[i] == c for col in columns) for c in sorted(VALID)] for i in range(n)]
    p_i = [(sum(k * k for k in row) - raters) / (raters * (raters - 1)) for row in counts]
    p_bar = sum(p_i) / n
    p_j = [sum(row[j] for row in counts) / (n * raters) for j in range(2)]
    expected = sum(p * p for p in p_j)
    return None if expected == 1 else (p_bar - expected) / (1 - expected)


def load_sheet(path: Path) -> dict[str, str]:
    rows = read_rows(path)
    ids = unique_ids(rows, "reviewId")
    labels = {}
    for identity, row in zip(ids, rows):
        value = (row.get("label") or "").strip()
        if value not in VALID:
            raise ValueError(f"{path.name}: {identity} has label {value!r}; expected 0 or 1")
        labels[identity] = value
    return labels


def load_adjudication(path: Path) -> dict[str, str]:
    rows = read_rows(path)
    ids = unique_ids(rows, "reviewId")
    result = {}
    for identity, row in zip(ids, rows):
        value = (row.get("label") or "").strip()
        if value not in VALID:
            raise ValueError(f"Adjudication {identity} has label {value!r}")
        result[identity] = value
    return result


def analyze(sheets: dict[str, dict[str, str]], adjudication: dict[str, str]) -> dict:
    names = sorted(sheets)
    id_sets = [set(sheets[n]) for n in names]
    if any(s != id_sets[0] for s in id_sets):
        raise ValueError("Annotator sheets do not contain the same reviewIds")
    ids = sorted(id_sets[0])
    columns = [[sheets[n][i] for i in ids] for n in names]
    disagreements = [i for i in ids if len({sheets[n][i] for n in names}) > 1]
    stray = set(adjudication) - set(disagreements)
    if stray:
        raise ValueError(f"Adjudication given for rows without disagreement: {sorted(stray)[:5]}")
    unresolved = [i for i in disagreements if i not in adjudication]
    if unresolved:
        raise ValueError(f"{len(unresolved)} disagreements need adjudication: {unresolved}")
    final = {i: adjudication.get(i, sheets[names[0]][i]) for i in ids}
    pairwise = {
        f"{x}-{y}": {
            "agreement": sum(p == q for p, q in zip(columns[a], columns[b])) / len(ids),
            "cohen_kappa": cohen_kappa(columns[a], columns[b]),
        }
        for (a, x), (b, y) in combinations(enumerate(names), 2)
    }
    return {
        "annotators": names,
        "rows": len(ids),
        "pairwise": pairwise,
        "fleiss_kappa": fleiss_kappa(columns) if len(names) >= 3 else None,
        "disagreement_ids": disagreements,
        "adjudicated": len(disagreements),
        "final_positive": sum(v == "1" for v in final.values()),
        "final_positive_rate": sum(v == "1" for v in final.values()) / len(ids),
        "_final": final,
    }


def run(args: argparse.Namespace) -> None:
    if not 2 <= len(args.sheets) <= 3:
        raise ValueError("Provide 2 or 3 completed sheets")
    require_new_dir(args.output_dir)
    sheets = {p.stem.removeprefix("annotator_"): load_sheet(p) for p in args.sheets}
    if len(sheets) != len(args.sheets):
        raise ValueError("Sheet names must be distinct")
    adjudication = load_adjudication(args.adjudication) if args.adjudication else {}
    report = analyze(sheets, adjudication)
    final = report.pop("_final")
    source = read_rows(args.source)
    unique_ids(source, args.id_column)
    by_id = {r[args.id_column]: r for r in source}
    if set(final) - set(by_id):
        raise ValueError("Annotated reviewIds are missing from --source")
    if args.label_column in source[0]:
        raise ValueError(f"--source already has a {args.label_column} column")
    merged = [{**by_id[i], args.label_column: final[i]} for i in sorted(final)]
    out_csv = args.output_dir / "annotated.csv"
    write_csv_new(out_csv, list(merged[0]), merged)
    write_json_new(
        args.output_dir / "agreement.json",
        {
            **report,
            "label_column": args.label_column,
            "id_column": args.id_column,
            "selected_ids": sorted(final),
            "adjudication_file": str(args.adjudication) if args.adjudication else None,
            "sheet_sha256": {p.name: digest_file(p) for p in args.sheets},
            "source_sha256": digest_file(args.source),
            "annotated_sha256": digest_file(out_csv),
        },
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheets", type=Path, nargs="+", required=True)
    parser.add_argument("--adjudication", type=Path, default=None)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--id-column", default="reviewId")
    parser.add_argument("--label-column", default="acting_quality")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.error(str(error))
    print(f"Wrote annotated.csv and agreement.json to {args.output_dir}")


if __name__ == "__main__":
    main()
