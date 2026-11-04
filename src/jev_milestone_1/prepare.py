"""Download a labeled benchmark test split and create a fixed-seed sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from .common import canonical_json, digest_file, write_json
from .config import DATASET_FILE, DATASET_REPO, FILTER_PROMPTS


def sample_records(
    source_rows: list[dict], *, limit: int, seed: int, labels: list[str]
) -> list[dict]:
    if limit <= 0 or limit > len(source_rows):
        raise ValueError(f"--limit must be between 1 and {len(source_rows)}")
    indices = sorted(random.Random(seed).sample(range(len(source_rows)), limit))
    result = []
    for i in indices:
        row = source_rows[i]
        if not isinstance(row.get("text"), str):
            raise ValueError(f"Bad text at dataset row {i}")
        gold = {label: int(row[label]) for label in labels}
        if not all(v in (0, 1) for v in gold.values()):
            raise ValueError(f"Non-binary label at dataset row {i}")
        result.append(
            {"id": f"test-{i:05d}", "source_row": i, "text": row["text"], "gold": gold}
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Sample size; 100 is reasonable for later experiments",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/sample.jsonl"))
    parser.add_argument(
        "--revision", help="Optional exact Hugging Face dataset commit SHA"
    )
    args = parser.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    repo_info = HfApi().repo_info(
        repo_id=DATASET_REPO, repo_type="dataset", revision=args.revision or "main"
    )
    revision = repo_info.sha  # Resolve the moving 'main' to an immutable commit.
    source = Path(
        hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            filename=DATASET_FILE,
            revision=revision,
        )
    )
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sample = sample_records(
        rows, limit=args.limit, seed=args.seed, labels=list(FILTER_PROMPTS)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in sample:
            f.write(canonical_json(record) + "\n")
    manifest = {
        "dataset_repo": DATASET_REPO,
        "dataset_file": DATASET_FILE,
        "dataset_revision": revision,
        "source_sha256": digest_file(source),
        "source_rows": len(rows),
        "sample_rows": len(sample),
        "sample_seed": args.seed,
        "labels": list(FILTER_PROMPTS),
        "sample_sha256": digest_file(args.output),
        "sampling_method": "uniform without replacement, sorted by source row",
        "notes": "Ground-truth emotion annotations can be subjective and imperfect.",
    }
    write_json(args.output.with_suffix(".manifest.json"), manifest)
    print(f"Wrote {len(sample)} records to {args.output}")
    print(f"Dataset revision: {revision}")
    print(f"Sample SHA256: {manifest['sample_sha256']}")
    print("Ground-truth positive counts:")
    for label in FILTER_PROMPTS:
        print(f"  {label}: {sum(x['gold'][label] for x in sample)}/{len(sample)}")


if __name__ == "__main__":
    main()
