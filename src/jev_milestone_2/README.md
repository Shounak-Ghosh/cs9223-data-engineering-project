# Milestone 2: combined and fixed-order semantic filters

Run commands from the repository root with the existing uv project. Milestone 1
commands and the frozen GoEmotions sample remain available. Milestone 2 compares
execution plans for the **same logical conjunction**, using TypeSafe Jev Noul
questions. Model decisions, call counts, runtime, and cost can differ across plans.

## Setup and offline checks

```bash
uv sync --locked
uv run pytest -q
uv run jev-compare --dry-run
uv run jev-experiment --dry-run
```

Dry runs validate data, task definitions, prices, orders, and any sample-manifest
checksum. They print request bounds without reading credentials, submitting Jev
requests, downloading data, or writing result files.

For live runs, put `TYPESAFE_API_KEY` in the local gitignored `.env` or export it.
The comparison commands use the installed TypeSafe SDK and do not acquire data.
Only the standalone `jev-fetch-sembench` command performs repository downloads.

## Model and pricing: TypeSafe

Use the official [TypeSafe model documentation](https://docs.typesafe.ai/models)
as the model/pricing reference. Checked on **2026-11-09**:

| Setting | Value used in the examples |
| --- | --- |
| Versioned model | `jev-1.14.0` |
| Input price | USD 0.042 per million input tokens (USD 42 per billion) |
| Output price | Free; pass `--output-usd-per-million 0` explicitly |
| Default context | 64k per request; 32k per question |
| Endpoint | `https://api.typesafe.ai/v1/systemone` |

The project keeps the versioned model instead of a moving alias. Each successful
call records the resolved model. TypeSafe documents that versioned model IDs can
be accepted even when its model-list endpoint lists only aliases. Recheck pricing
before a new run and use your actual account contract if it differs. Thresholds
should be selected before evaluating the reporting sample; the default is 0.5.

The CLI requires **both** token-price flags when pricing is configured:

```text
--model jev-1.14.0
--input-usd-per-million 0.042
--output-usd-per-million 0
--pricing-source "https://docs.typesafe.ai/models; checked 2026-11-09; Jev 1.14"
```

Estimated USD = (SDK input tokens × input rate + SDK output tokens × output rate)
/ 1,000,000. Usage is counted **once per request**, including combined requests.
Jev shares a record's state across questions in a combined request; do not multiply
its returned input-token usage by the number of questions.

Raw usage is retained independently of pricing. Missing rates or required usage
produce JSON `null`, never an assumed zero. Prices remain explicit run parameters;
the code does not hardcode or automatically fetch a provider price. Estimates are
not invoices and exclude credits, discounts, taxes, and ambiguous failed-request
charges. SDK retries are disabled, so a failed request aborts the experiment
instead of silently adding retries; TypeSafe rate-limit errors may require a new
run after waiting.

The optional `--credits-per-question-per-block`, `--bytes-per-block`, and
`--usd-per-credit` flags remain for explicitly different custom billing contracts.
They estimate question count × locally reconstructed request-size blocks × rate.
**They do not describe TypeSafe Jev pricing and should not be used for these
experiments.** They cannot be combined with the token-price flags.

## Plans, defaults, and budgets

| Plan | Behavior |
| --- | --- |
| Combined | One `system_one()` request per record, containing all selected Noul questions; AND the thresholded decisions |
| Sequential A → B | Ask A, then ask B only if A passes |
| Sequential B → A | Ask B, then ask A only if B passes |
| Independent baseline | Ask each predicate separately for every record, including rows already rejected by another predicate |

Combined execution uses multiple named questions, not a rewritten conjunctive
question or multiplied probabilities. Sequential requests use the same original
text and predicate prompts. Every plan gets fresh calls without application-level
caching or result reuse. Plan execution order rotates between repetitions while
each sequential plan's internal filter order stays fixed. Requests execute serially.

`jev-compare` defaults to the three required plans, 10 rows, one repetition, and
`gratitude admiration`. Its worst case is 50 calls against a default 100-call
budget. `jev-experiment` adds the independent baseline and defaults to two repeats:
100–140 calls for the same 10 rows, with no default budget. Pass `--max-api-calls`
to either command to enforce a conservative worst-case bound before live requests.

For more than two filters, default sequential orders are the selected order and
its reverse. Repeated `--order` flags can replace them with explicit permutations;
each must contain every selected filter exactly once. Such overrides can change
the number of plans. Example using the independent baseline:

```bash
uv run jev-experiment --input data/sample.jsonl --max-records 100 --repeats 2 \
  --filters gratitude admiration optimism \
  --order gratitude admiration optimism \
  --order optimism admiration gratitude \
  --max-api-calls 2000 --dry-run
```

## Dataset roles and label requirements

| Dataset | Role |
| --- | --- |
| `data/sample.jsonl` (GoEmotions) | Frozen development, debugging, and regression comparisons |
| Verified SemBench-derived text sample | Research evaluation of realistic semantic predicates |
| Synthetic labeled fixtures | Offline tests and known-selectivity cases |

The default input remains GoEmotions. SemBench is imported explicitly with its task
JSON. The [SemBench Movie scenario](https://github.com/SemBench/SemBench/blob/main/src/scenario/movie/README.md)
provides `scoreSentiment` labels. Its positive-review/movie-ID query has one
semantic and one structured predicate. The additional `acting_quality` predicate
requires separate human annotations and creates a **derived task**, not an
unchanged official SemBench query.

The importer requires individual verified labels for every selected predicate and
derives conjunction gold with logical AND. Only `reviewText` is sent to Jev;
sentiment and human labels remain outside the model input. It validates IDs,
label mappings, data/task checksums, and declared provenance, but cannot certify
human annotation accuracy.

## Movie workflow: fetch, annotate, import, compare

Outputs must be new files/directories. If the local artifacts already exist,
continue from the relevant step or choose new paths; these commands do not resume
or overwrite previous outputs.

### 1. Fetch the pinned Movie snapshot

```bash
uv run jev-fetch-sembench \
  --ref c4b2bcff3a5907148f504cfb6faea4c6d3f90e8f \
  --reviews-file files/movie/data/sf_2000/Reviews.csv \
  --drop-exact-duplicates --output-dir data/sembench_fetch
```

This revision contains both the reviews and `scoreSentiment` in the committed CSV;
no separate label download is needed for this snapshot. `fetch.json` records the
resolved revision, URL, raw-file hashes, output hash, duplicate count, and timestamp.
Only identical rows are dropped; conflicting duplicate IDs fail. Other upstream
revisions may require external data acquisition. Record an external source's
version and checksum separately if changing to such a workflow.

### 2. Create blind human annotation sheets

```bash
uv run jev-make-annotation-sheets --source data/sembench_fetch/verified_reviews.csv \
  --n 100 --seed 42 --annotators 3 --output-dir data/annotation
```

Give each annotator only their CSV and `GUIDELINES.md`. Sheets contain ID, text,
and an empty 0/1 label column, with no sentiment labels or model predictions.
Annotators work independently. The predicate asks whether the review comments on
acting or performance quality; praise and criticism both count.

Optional `--keyword-fraction 0.5` changes sampling to oversample acting-related
keywords. It does not assign labels or guarantee semantic positives. It is
recorded in the manifest and `label_source`; disclose the resulting sampling bias.

### 3. Check agreement and resolve every disagreement

After the annotators complete their sheets:

```bash
uv run jev-annotation-agreement --sheets data/annotation/annotator_{A,B,C}.csv \
  --source data/sembench_fetch/verified_reviews.csv \
  --output-dir data/annotation_result
```

The tool validates matching IDs and complete binary labels. `agreement.json`
contains pairwise raw agreement and Cohen's kappa, Fleiss' kappa for three raters,
raw unanimous agreement, per-pair 2×2 class confusion tables, and disagreement IDs.
Table rows belong to `row_annotator`, columns to `column_annotator`; keys `0` and
`1` are annotation classes, not model-vs-gold confusion counts. Undefined kappa
(e.g. everyone labels every item identically) is `null`.

Older saved agreement reports remain usable by the existing task/provenance
checks, but lack the new confusion tables. To include those tables in a submission,
regenerate agreement and subsequent task/import artifacts using new output paths;
keep the completed human sheets as the annotation inputs.

If there are disagreements, the command prints all unresolved IDs and fails
without a final merged dataset. Create a separate `adjudicated.csv` with
`reviewId,label` for **every** disagreeing ID, then rerun the command with
`--adjudication adjudicated.csv`. Unknown IDs or adjudications for unanimous rows
fail. Statistics have no automatic pass/fail threshold; review them yourself.

### 4. Build and review the verified task definition

```bash
uv run jev-build-task --template examples/sembench_movie.task.template.json \
  --fetch-json data/sembench_fetch/fetch.json \
  --agreement-json data/annotation_result/agreement.json \
  --sheet-manifest data/annotation/sheet_manifest.json \
  --dataset data/annotation_result/annotated.csv --output data/verified_task.json
```

Review `label_source` before treating this as a final research task. The builder
checks dataset/source hashes, selected IDs, row and annotator counts, prompt,
positive support, and adjudication metadata. It copies the fetched revision and
source URL, retains report hashes, and validates the task in memory without
creating a temporary validation file. Existing output files are preserved.

The template intentionally contains nulls until filled. Each predicate needs a
prompt, distinct label column, label-source description, and disjoint positive and
negative string mappings. New agreement reports carry selected IDs; older reports
without them require the matching sheet manifest. Reports lacking required
checksums must be regenerated. Do not modify annotations after producing their
report without regenerating the report and subsequent artifacts.

### 5. Freeze the annotated sample for experiment input

```bash
uv run jev-prepare-sembench \
  --source data/annotation_result/annotated.csv \
  --task data/verified_task.json \
  --limit 100 --seed 42 \
  --output data/sembench_movie_verified.jsonl
```

This produces the canonical `{id, text, gold}` JSONL records and a hashed manifest
with dataset/task provenance, annotation sources, sampling parameters, positive
support, and conjunction-positive count. `--limit` cannot exceed the annotated
row count. When there are exactly 100 annotated rows, this command retains all of
them; it does not add a new sample or generate missing labels.

### 6. Validate the priced comparison, then run it

First validate the exact intended settings without sending requests:

```bash
uv run jev-compare \
  --input data/sembench_movie_verified.jsonl --task data/verified_task.json \
  --max-records 100 --repeats 1 --max-api-calls 500 \
  --model jev-1.14.0 \
  --input-usd-per-million 0.042 --output-usd-per-million 0 \
  --pricing-source "https://docs.typesafe.ai/models; checked 2026-11-09; Jev 1.14" \
  --output-dir results/milestone2 --dry-run
```

After reviewing the task, budget, and current TypeSafe price, the following submits
live requests and saves the three-plan measurements:

```bash
uv run jev-compare \
  --input data/sembench_movie_verified.jsonl --task data/verified_task.json \
  --max-records 100 --repeats 1 --max-api-calls 500 \
  --model jev-1.14.0 \
  --input-usd-per-million 0.042 --output-usd-per-million 0 \
  --pricing-source "https://docs.typesafe.ai/models; checked 2026-11-09; Jev 1.14" \
  --output-dir results/milestone2
```

For 100 rows and two predicates, one repeat makes 300–500 calls. To measure repeat
stability, use `--repeats 2 --max-api-calls 1000` and a new output directory.

### 7. Inspect and report the comparison

Use `summary.csv` for a plan-by-plan table: calls, input/output tokens, runtime,
precision/recall/F1, positive support, and estimated USD. `summary.json` retains
quality per repetition, repeat agreement, and prediction disagreements against
combined execution. CSV quality uses only the first repeat; calls, runtime, and
cost aggregate all repeats. The independent baseline, if enabled, also has its own
agreement comparison.

Inspect changed record IDs/probabilities in `decisions.jsonl` and `calls.jsonl`.
Report the sample/revision, annotation protocol, model, threshold, filter orders,
number of repeats, pricing source/date, conjunction-positive support, and any
sampling bias alongside the measurements. Include the annotation agreement report.
Do not pool repeated predictions into a larger nominal quality sample.

## Artifacts and interpretation

| Directory artifact | Contents |
| --- | --- |
| `calls.jsonl` | One row per attempted SDK call; successful probabilities, usage, model, request ID, latency, and cost; failed-call type without sensitive exception strings |
| `decisions.jsonl` | One row per input/plan/repeat; final AND gold/prediction, evaluated probabilities, skipped decisions, calls, usage, runtime, cost |
| `runs.jsonl` | Per-plan/repeat wall time, call count, execution position, stage input/pass counts |
| `summary.json` | Quality per repeat; all-repeat calls, questions, tokens, runtime, cost; repeat and cross-plan agreement |
| `summary.csv` | First-repeat quality and all-repeat execution measurements, one row per plan |
| `manifest.json` | Completion status, settings, versions, task/data provenance, selected IDs, prices, timestamps, result hashes |

`--output results/name.jsonl` instead produces `name.jsonl`, `name.calls.jsonl`,
`name.runs.jsonl`, `name.summary.json`, `name.summary.csv`, and `name.manifest.json`.

Skipped predicates are unobserved, not negative labels. Every input row counts in
final-query quality even if execution stops early. Stage pass counts in sequential
plans describe conditional selectivity; use Milestone 1 for full-dataset individual
filter quality. Check conjunction support: small/imbalanced samples can have no
positives, and summaries flag that case. The frozen GoEmotions sample has no
positive anger labels. Accuracy alone can conceal poor rare-class performance.

Runtime includes serial request and local processing time; API latency wraps the
SDK invocation and is not server-only inference time. Combined requests need not
match separate-request decisions. Fewer requests do not alone establish lower
billable tokens or better quality; compare measured usage and saved estimates.

Raw calls/decisions are flushed incrementally. Interrupted runs keep partial
artifacts and a failed manifest; the exit status and manifest determine whether
results are complete. Start a new output path to rerun. There is no automatic
resume, application cache, download, or model-generated annotation in the
experiment commands.
