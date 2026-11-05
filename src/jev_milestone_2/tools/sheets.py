"""Generate blind annotation sheets for the acting_quality predicate."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import re
import string

from jev_milestone_1.common import digest_file
from jev_milestone_2.tools.common import (
    read_rows,
    require_new_dir,
    unique_ids,
    write_csv_new,
    write_json_new,
    write_text_new,
)

PROMPT = "Does this movie review comment on the quality of the acting or performances?"
KEYWORDS = re.compile(r"\b(act(?:or|ors|ress|ing|ed)?|cast|performances?|perform(?:ed|s)?)\b", re.I)

GUIDELINES = f"""# Annotation guidelines: acting_quality

Question: **{PROMPT}**

Fill the `label` column with `1` (yes) or `0` (no) for every row. Work alone.

- `1`: the review evaluates or comments on the acting, cast, or performances,
  whether praise or criticism ("wooden delivery", "the lead was superb").
- `0`: no such comment. Plot, directing, effects, music, or writing alone do not
  count, and neither does a bare actor name with no assessment.
- Judge only the text shown. Do not infer from the overall sentiment of the review.
"""


def select_rows(
    rows: list[dict], text_column: str, n: int, seed: int, keyword_fraction: float
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    if keyword_fraction == 0:
        return rng.sample(rows, n), {"mode": "uniform"}
    matching = [r for r in rows if KEYWORDS.search(r[text_column])]
    other = [r for r in rows if not KEYWORDS.search(r[text_column])]
    n_match = round(n * keyword_fraction)
    if len(matching) < n_match or len(other) < n - n_match:
        raise ValueError("Not enough rows to satisfy --keyword-fraction")
    picked = rng.sample(matching, n_match) + rng.sample(other, n - n_match)
    return picked, {
        "mode": "keyword-oversample",
        "keyword_pattern": KEYWORDS.pattern,
        "keyword_fraction": keyword_fraction,
        "keyword_matched_in_source": len(matching),
        "note": "Not representative of the source; disclose in label_source.",
    }


def make_sheets(args: argparse.Namespace) -> list[Path]:
    if not 2 <= args.annotators <= 3:
        raise ValueError("--annotators must be 2 or 3")
    if not 0 <= args.keyword_fraction <= 1:
        raise ValueError("--keyword-fraction must be in [0, 1]")
    rows = read_rows(args.source)
    unique_ids(rows, args.id_column)
    if any(not isinstance(r.get(args.text_column), str) or not r[args.text_column].strip() for r in rows):
        raise ValueError(f"Every row needs nonempty {args.text_column}")
    if not 1 <= args.n <= len(rows):
        raise ValueError(f"--n must be between 1 and {len(rows)}")
    require_new_dir(args.output_dir)
    chosen, sampling = select_rows(rows, args.text_column, args.n, args.seed, args.keyword_fraction)
    chosen.sort(key=lambda r: r[args.id_column])
    paths, checksums = [], {}
    for index in range(args.annotators):
        name = string.ascii_uppercase[index]
        order = random.Random(args.seed + 1 + index).sample(chosen, len(chosen))
        sheet = [
            {"reviewId": r[args.id_column], "reviewText": r[args.text_column], "label": ""}
            for r in order
        ]
        path = args.output_dir / f"annotator_{name}.csv"
        write_csv_new(path, ["reviewId", "reviewText", "label"], sheet)
        paths.append(path)
        checksums[path.name] = digest_file(path)
    write_text_new(args.output_dir / "GUIDELINES.md", GUIDELINES)
    write_json_new(
        args.output_dir / "sheet_manifest.json",
        {
            "predicate": "acting_quality",
            "prompt": PROMPT,
            "source_file": str(args.source),
            "source_sha256": digest_file(args.source),
            "id_column": args.id_column,
            "text_column": args.text_column,
            "seed": args.seed,
            "sampling": sampling,
            "annotators": args.annotators,
            "selected_ids": [r[args.id_column] for r in chosen],
            "blank_sheet_sha256": checksums,
        },
    )
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--annotators", type=int, default=2)
    parser.add_argument("--keyword-fraction", type=float, default=0.0,
                        help="Fraction of rows drawn from keyword matches (0 = uniform sample)")
    parser.add_argument("--id-column", default="reviewId")
    parser.add_argument("--text-column", default="reviewText")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        paths = make_sheets(args)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"Wrote {len(paths)} sheets to {args.output_dir}")


if __name__ == "__main__":
    main()
