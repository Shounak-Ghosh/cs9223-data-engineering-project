"""Fetch SemBench at a pinned ref and join reviews with withheld gold labels.

Standalone and networked by design; the Milestone 2 commands never download data.
Only the files named on the command line are used, so no SemBench layout is assumed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import tempfile

from jev_milestone_1.common import digest_file
from jev_milestone_2.tools.common import (
    read_rows,
    require_new_dir,
    unique_ids,
    write_csv_new,
    write_json_new,
)

DEFAULT_REPO = "https://github.com/SemBench/SemBench"


def git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def drop_exact_duplicates(rows: list[dict]) -> tuple[list[dict], int]:
    """Remove rows identical in every column to an earlier row (first kept)."""
    seen, kept = set(), []
    for row in rows:
        key = tuple(sorted(row.items()))
        if key not in seen:
            seen.add(key)
            kept.append(row)
    return kept, len(rows) - len(kept)


def join_reviews(
    reviews: list[dict],
    labels: list[dict] | None,
    id_column: str,
    label_column: str,
) -> list[dict]:
    """Attach the withheld label to each review; fail on any gap or duplicate."""
    review_ids = unique_ids(reviews, id_column)
    if labels is None:
        by_id = {
            row[id_column]: row.get(label_column) for row in reviews
        }
    else:
        label_ids = unique_ids(labels, id_column)
        by_id = {
            row[id_column]: row.get(label_column) for row in labels
        }
        missing = set(review_ids) - set(label_ids)
        if missing:
            raise ValueError(f"{len(missing)} reviews have no withheld label")
    joined = []
    for row in reviews:
        value = by_id.get(row[id_column])
        if value is None or str(value).strip() == "":
            raise ValueError(f"Review {row[id_column]} has no {label_column}")
        joined.append({**row, label_column: str(value)})
    return joined


def fetch(args: argparse.Namespace) -> Path:
    require_new_dir(args.output_dir)
    with tempfile.TemporaryDirectory() as scratch:
        checkout = Path(scratch) / "sembench"
        git("clone", "--quiet", args.repo, str(checkout))
        git("checkout", "--quiet", args.ref, cwd=checkout)
        revision = git("rev-parse", "HEAD", cwd=checkout)
        reviews_path = checkout / args.reviews_file
        labels_path = checkout / args.labels_file if args.labels_file else None
        for path in (reviews_path, labels_path):
            if path is not None and not path.is_file():
                raise ValueError(f"{path.relative_to(checkout)} not found at {revision}")
        reviews = read_rows(reviews_path)
        labels = read_rows(labels_path) if labels_path else None
        dropped = 0
        if args.drop_exact_duplicates:
            reviews, dropped = drop_exact_duplicates(reviews)
        joined = join_reviews(reviews, labels, args.id_column, args.label_column)
        sources = {
            str(args.reviews_file): digest_file(reviews_path),
            **({str(args.labels_file): digest_file(labels_path)} if labels_path else {}),
        }
    fieldnames = list(joined[0])
    out_csv = args.output_dir / "verified_reviews.csv"
    write_csv_new(out_csv, fieldnames, joined)
    write_json_new(
        args.output_dir / "fetch.json",
        {
            "source_url": args.repo,
            "ref": args.ref,
            "revision": revision,
            "reviews_file": str(args.reviews_file),
            "labels_file": str(args.labels_file) if args.labels_file else None,
            "id_column": args.id_column,
            "label_column": args.label_column,
            "source_sha256": sources,
            "rows": len(joined),
            "exact_duplicate_rows_dropped": dropped,
            "output_sha256": digest_file(out_csv),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return out_csv


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--ref", required=True, help="Pinned commit, tag, or branch")
    parser.add_argument("--reviews-file", type=Path, required=True,
                        help="Reviews CSV/JSONL path inside the checkout")
    parser.add_argument("--labels-file", type=Path, default=None,
                        help="Separate withheld-label file; omit if labels are in the reviews file")
    parser.add_argument("--drop-exact-duplicates", action="store_true",
                        help="Drop rows identical in every column to an earlier row "
                             "(count recorded in fetch.json); other duplicate IDs still fail")
    parser.add_argument("--id-column", default="reviewId")
    parser.add_argument("--label-column", default="scoreSentiment")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        out = fetch(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(f"Wrote {out} and fetch.json")


if __name__ == "__main__":
    main()
