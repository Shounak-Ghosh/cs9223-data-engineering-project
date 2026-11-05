"""Offline tests for the annotation tools; synthetic data and a local git repo only."""

import csv
import json
import subprocess

import pytest

from jev_milestone_1.common import digest_file
from jev_milestone_2.datasets import main as prepare_main
from jev_milestone_2.runner import main as compare_main
from jev_milestone_2.tools import agreement, build_task, fetch, sheets
from jev_milestone_2.tools.agreement import analyze, cohen_kappa, fleiss_kappa

TEMPLATE = {
    "dataset": {"name": "SemBench", "scenario": "movie_with_additional_annotations",
                "revision": None, "source_url": "https://example.invalid/sembench"},
    "id_column": "reviewId", "text_column": "reviewText",
    "predicates": {
        "positive": {"kind": "semantic", "prompt": "Positive?", "label_column": "scoreSentiment",
                     "positive_values": ["POSITIVE"], "negative_values": ["NEGATIVE"],
                     "label_source": "Synthetic sentiment"},
        "second_semantic_predicate": {"kind": "semantic", "prompt": None, "label_column": None,
                                      "positive_values": ["1"], "negative_values": ["0"],
                                      "label_source": None},
    },
}


def write_csv(path, rows):
    with path.open("w", newline="") as s:
        w = csv.DictWriter(s, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with path.open(newline="") as s:
        return list(csv.DictReader(s))


@pytest.fixture
def source(tmp_path):
    rows = [{"reviewId": f"r{i:03d}",
             "reviewText": ("The acting was superb" if i % 4 == 0 else "A fine plot") + f" {i}",
             "scoreSentiment": "POSITIVE" if i % 2 else "NEGATIVE"} for i in range(40)]
    path = tmp_path / "reviews.csv"
    write_csv(path, rows)
    return path


# ---- fetch ----

def make_repo(tmp_path, separate_labels):
    repo = tmp_path / "repo"
    repo.mkdir()
    reviews = [{"reviewId": "a", "reviewText": "x"}, {"reviewId": "b", "reviewText": "y"}]
    labels = [{"reviewId": "a", "scoreSentiment": "POSITIVE"}, {"reviewId": "b", "scoreSentiment": "NEGATIVE"}]
    if separate_labels:
        write_csv(repo / "reviews.csv", reviews)
        write_csv(repo / "labels.csv", labels)
    else:
        write_csv(repo / "reviews.csv", [{**r, **l} for r, l in zip(reviews, labels)])
    env = ["-c", "user.email=t@example.invalid", "-c", "user.name=t"]
    for cmd in (["init", "-q", "-b", "main"], ["add", "."], [*env, "commit", "-q", "-m", "x"]):
        subprocess.run(["git", *cmd], cwd=repo, check=True, capture_output=True)
    return repo


@pytest.mark.parametrize("separate", [True, False])
def test_fetch_joins_labels_and_records_revision(tmp_path, separate):
    repo = make_repo(tmp_path, separate)
    out = tmp_path / "out"
    argv = ["--repo", str(repo), "--ref", "main", "--reviews-file", "reviews.csv",
            "--output-dir", str(out)]
    if separate:
        argv += ["--labels-file", "labels.csv"]
    fetch.main(argv)
    rows = read_csv(out / "verified_reviews.csv")
    assert [r["scoreSentiment"] for r in rows] == ["POSITIVE", "NEGATIVE"]
    meta = json.loads((out / "fetch.json").read_text())
    assert len(meta["revision"]) == 40 and meta["rows"] == 2
    with pytest.raises(SystemExit):
        fetch.main(argv)  # existing output is never overwritten


def test_join_rejects_gaps_and_duplicates():
    reviews = [{"reviewId": "a"}, {"reviewId": "b"}]
    with pytest.raises(ValueError, match="no withheld label"):
        fetch.join_reviews(reviews, [{"reviewId": "a", "s": "P"}], "reviewId", "s")
    with pytest.raises(ValueError, match="Duplicate"):
        fetch.join_reviews([{"reviewId": "a"}, {"reviewId": "a"}], None, "reviewId", "s")
    with pytest.raises(ValueError, match="has no s"):
        fetch.join_reviews(reviews, [{"reviewId": "a", "s": "P"}, {"reviewId": "b", "s": ""}], "reviewId", "s")


# ---- sheets ----

def run_sheets(source, out, *extra):
    sheets.main(["--source", str(source), "--n", "10", "--seed", "7", "--output-dir", str(out), *extra])


def test_sheets_deterministic_blind_and_same_rows(tmp_path, source):
    run_sheets(source, tmp_path / "a", "--annotators", "3")
    run_sheets(source, tmp_path / "b", "--annotators", "3")
    for name in ("annotator_A.csv", "annotator_B.csv", "annotator_C.csv", "sheet_manifest.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    a, b = read_csv(tmp_path / "a/annotator_A.csv"), read_csv(tmp_path / "a/annotator_B.csv")
    assert set(a[0]) == {"reviewId", "reviewText", "label"}  # no sentiment leakage
    assert all(r["label"] == "" for r in a)
    assert {r["reviewId"] for r in a} == {r["reviewId"] for r in b} and len(a) == 10
    assert [r["reviewId"] for r in a] != [r["reviewId"] for r in b]


def test_sheets_keyword_oversample_recorded(tmp_path, source):
    run_sheets(source, tmp_path / "k", "--keyword-fraction", "0.5")
    manifest = json.loads((tmp_path / "k/sheet_manifest.json").read_text())
    assert manifest["sampling"]["mode"] == "keyword-oversample"
    texts = [r["reviewText"] for r in read_csv(tmp_path / "k/annotator_A.csv")]
    assert sum("acting" in t for t in texts) == 5


@pytest.mark.parametrize("extra", [["--annotators", "1"], ["--annotators", "4"], ["--n", "99"],
                                   ["--keyword-fraction", "1.5"]])
def test_sheets_rejects_bad_args(tmp_path, source, extra):
    with pytest.raises(SystemExit):
        sheets.main(["--source", str(source), "--n", "10", "--output-dir", str(tmp_path / "o"), *extra])


# ---- agreement ----

def test_kappa_known_values():
    assert cohen_kappa(list("0101"), list("0101")) == 1
    assert cohen_kappa(list("0011"), list("0101")) == 0
    assert cohen_kappa(list("0000"), list("0000")) is None
    assert fleiss_kappa([list("0101"), list("0101"), list("0101")]) == 1
    # Hand-computed: 4 subjects, 3 raters
    cols = [list("0011"), list("0011"), list("0101")]
    assert fleiss_kappa(cols) == pytest.approx(0.333333, abs=1e-5)


def test_analyze_adjudication_rules():
    sheets_ = {"A": {"1": "1", "2": "0"}, "B": {"1": "1", "2": "1"}}
    with pytest.raises(ValueError, match="need adjudication"):
        analyze(sheets_, {})
    with pytest.raises(ValueError, match="without disagreement"):
        analyze(sheets_, {"1": "1", "2": "0"})
    report = analyze(sheets_, {"2": "0"})
    assert report["_final"] == {"1": "1", "2": "0"} and report["adjudicated"] == 1
    with pytest.raises(ValueError, match="same reviewIds"):
        analyze({"A": {"1": "1"}, "B": {"2": "1"}}, {})


def fill(sheet_dir, name, labeler):
    path = sheet_dir / f"annotator_{name}.csv"
    rows = read_csv(path)
    for r in rows:
        r["label"] = labeler(r)
    write_csv(path, rows)
    return path


def test_agreement_rejects_blank_and_invalid(tmp_path, source):
    run_sheets(source, tmp_path / "s")
    a = tmp_path / "s/annotator_A.csv"
    fill(tmp_path / "s", "B", lambda r: "1")
    with pytest.raises(SystemExit):  # blanks
        agreement.main(["--sheets", str(a), str(tmp_path / "s/annotator_B.csv"),
                        "--source", str(source), "--output-dir", str(tmp_path / "o")])
    fill(tmp_path / "s", "A", lambda r: "maybe")
    with pytest.raises(SystemExit):
        agreement.main(["--sheets", str(a), str(tmp_path / "s/annotator_B.csv"),
                        "--source", str(source), "--output-dir", str(tmp_path / "o2")])


# ---- end to end through the existing importer ----

def test_end_to_end_with_importer(tmp_path, source):
    s = tmp_path / "s"
    run_sheets(source, s)
    has_acting = lambda r: "1" if "acting" in r["reviewText"] else "0"
    a = fill(s, "A", has_acting)
    disputed = read_csv(a)[0]["reviewId"]
    b = fill(s, "B", lambda r: ("1" if has_acting(r) == "0" else "0") if r["reviewId"] == disputed else has_acting(r))
    adjud = tmp_path / "adj.csv"
    write_csv(adjud, [{"reviewId": disputed, "label": "1"}])
    out = tmp_path / "agree"
    agreement.main(["--sheets", str(a), str(b), "--adjudication", str(adjud),
                    "--source", str(source), "--output-dir", str(out)])
    report = json.loads((out / "agreement.json").read_text())
    assert report["rows"] == 10 and report["adjudicated"] == 1

    fetch_json = tmp_path / "fetch.json"
    fetch_json.write_text(json.dumps({"revision": "abc123", "source_url": "https://example.invalid/fetched-repo",
                                      "id_column": "reviewId", "label_column": "scoreSentiment",
                                      "output_sha256": digest_file(source), "rows": 40}))
    template = tmp_path / "template.json"
    template.write_text(json.dumps(TEMPLATE))
    task_path = tmp_path / "task.json"
    build_task.main(["--template", str(template), "--fetch-json", str(fetch_json),
                     "--agreement-json", str(out / "agreement.json"),
                     "--sheet-manifest", str(s / "sheet_manifest.json"),
                     "--dataset", str(out / "annotated.csv"), "--output", str(task_path)])
    task = json.loads(task_path.read_text())
    assert task["dataset"]["revision"] == "abc123"
    assert task["dataset"]["source_url"] == "https://example.invalid/fetched-repo"
    assert "acting_quality" in task["predicates"] and "second_semantic_predicate" not in task["predicates"]
    assert "Cohen kappa" in task["predicates"]["acting_quality"]["label_source"]

    sample = tmp_path / "sample.jsonl"
    prepare_main(["--source", str(out / "annotated.csv"), "--task", str(task_path),
                  "--output", str(sample), "--limit", "10"])
    records = [json.loads(l) for l in sample.read_text().splitlines()]
    assert len(records) == 10 and all(set(r["gold"]) == {"positive", "acting_quality"} for r in records)
    compare_main(["--input", str(sample), "--task", str(task_path), "--max-records", "10", "--dry-run"])


def test_build_task_refuses_unfilled_or_bad_inputs(tmp_path, source):
    template = tmp_path / "t.json"
    template.write_text(json.dumps(TEMPLATE))
    fetch_json = tmp_path / "f.json"
    fetch_json.write_text(json.dumps({"revision": "r"}))
    agree = tmp_path / "a.json"
    agree.write_text(json.dumps({"label_column": "scoreSentiment", "annotators": ["A", "B"], "rows": 40,
                                 "pairwise": {}, "adjudicated": 0}))
    # label column collides with the sentiment column -> distinct-column check / 0-1 check fails
    with pytest.raises(SystemExit):
        build_task.main(["--template", str(template), "--fetch-json", str(fetch_json),
                         "--agreement-json", str(agree), "--dataset", str(source),
                         "--output", str(tmp_path / "out.json")])
    assert not (tmp_path / "out.json").exists()


def test_drop_exact_duplicates_only_removes_identical_rows():
    rows = [{"reviewId": "a", "t": "x"}, {"reviewId": "a", "t": "x"}, {"reviewId": "b", "t": "y"}]
    kept, dropped = fetch.drop_exact_duplicates(rows)
    assert kept == [rows[0], rows[2]] and dropped == 1
    conflicting = [{"reviewId": "a", "t": "x"}, {"reviewId": "a", "t": "z"}]
    kept, dropped = fetch.drop_exact_duplicates(conflicting)
    assert dropped == 0
    with pytest.raises(ValueError, match="Duplicate"):
        fetch.join_reviews(kept, None, "reviewId", "t")
