# Milestone 2: fixed semantic-filter plans

Run from the repository root with the existing uv environment. Milestone 1 and
`jev-evaluate` remain the independent per-filter reference. The existing pinned
model, Milestone 1 commands, and frozen GoEmotions snapshot are preserved.

This milestone adds two commands that share one engine for execution, metrics, and
failure handling:

- `jev-experiment`: three or more plans, including an independent baseline.
- `jev-compare`: a lighter three-plan comparison (combined, sequential A → B,
  sequential B → A), defaulting to 10 rows, one repetition, and the gratitude and
  admiration predicates.

Both evaluate an **AND query** over the same input rows using fresh calls for every
plan and repeat, and both disable SDK retries so one invocation is one attempted
request. Plans do not share model results.

## Dataset roles

| Dataset | Role |
| --- | --- |
| Frozen GoEmotions sample (`data/sample.jsonl`) | Development, debugging, and regression comparisons |
| Verified SemBench-derived text tasks | Research benchmark candidates and final evaluation |
| Synthetic labeled examples | Offline tests with known conjunction labels and selectivities |

SemBench integration is an **explicit local import path**, not a claim that the
default sample has become SemBench or that final benchmark evaluation has run. No
benchmark data is downloaded by these commands, and no model-generated labels are
used as ground truth.

## Execution plans

`jev-experiment` supports:

- **Independent:** one question per call, evaluate every selected filter on every row.
- **Combined:** all selected Noul questions in one call per row; AND their individual
  threshold decisions. This does not ask a new, paraphrased conjunction question
  and does not multiply probabilities.
- **Sequential:** one question per call, in a fixed order; stop a row at its first
  failed filter. Defaults to the selected filter order and its reverse.

No optimizer, learned ordering, batching across records, or application cache is
used. Each call passes the same text and the configured question prompts (Milestone 1 defaults or an explicit task definition).
Plan execution order rotates across repetitions to reduce timing-order bias; each
sequential plan's internal filter order stays fixed.

## Validate before running

```bash
uv sync --locked
uv run jev-experiment --dry-run
uv run pytest -q
```

The dry run validates records, filters, prices, and orders, then prints lower and
upper request counts without reading credentials, sending requests, or writing
results. The default 10 rows, 2 repeats, and 2 filters make **100–140 calls**:
40 independent, 20 combined, and 20–40 for each sequential order.

`jev-compare` runs exactly the combined plan plus forward and reverse sequential
orders. Its default worst case is **50 requests**, enforced against the default
100-call budget before credentials are read or requests are made. Larger runs need
an explicit sufficient `--max-api-calls` (also accepted by `jev-experiment`, along
with `--task` and the pricing options below):

```bash
uv run jev-compare --dry-run
uv run jev-compare --input data/sample.jsonl --filters gratitude admiration \
  --max-records 100 --repeats 2 --max-api-calls 1000 \
  --output results/m2_100x2.jsonl
```

`jev-compare` writes overlay-style names (`name.jsonl`, `name.calls.jsonl`,
`name.runs.jsonl`, `name.summary.json`, `name.summary.csv`, `name.manifest.json`)
when given `--output`, or the directory layout below with `--output-dir`.

## Live experiment

Set `TYPESAFE_API_KEY` in the environment or the local gitignored `.env`.

```bash
uv run jev-experiment --max-records 10 --repeats 2 \
  --filters gratitude admiration --output-dir results/milestone2-pilot
```

Provide explicit fixed orders with repeated `--order` flags. Each order must be
a permutation of all selected filters, with no duplicate plans:

```bash
uv run jev-experiment --max-records 100 --repeats 2 \
  --filters gratitude admiration optimism \
  --order gratitude admiration optimism \
  --order optimism admiration gratitude \
  --output-dir results/milestone2-three-filters
```

Orders are chosen before execution; results are not reused from another plan.
The model defaults to the existing pinned `jev-1.14.0`; `--model` and `--threshold`
allow explicit overrides. Avoid threshold tuning on the reporting sample.

## Importing a verified SemBench-derived task

SemBench's [Movie scenario](https://github.com/SemBench/SemBench/blob/main/src/scenario/movie/README.md)
provides withheld `scoreSentiment` labels, but its positive-review/movie-ID query
has one semantic predicate and one structured condition. It does not supply two
independent semantic labels. A second semantic predicate requires validated
annotations; a deterministic movie-ID test must not be passed off as a semantic
filter. Adding annotations creates a derived task that must be disclosed separately
from an unchanged official SemBench query.

1. Obtain a text-task CSV/JSONL export locally, including withheld gold-label
   columns, and record the SemBench/data revision.
2. Verify labels for at least two semantic predicates per record. Annotate and
   validate any missing predicate separately, and describe the protocol in
   `label_source`.
3. Copy [the Movie task template](../../examples/sembench_movie.task.template.json)
   and fill in the revision and all second-predicate fields. The template
   deliberately contains nulls and cannot run as though annotations existed.

A task defines dataset `name`, `scenario`, `revision`, `source_url`, `id_column`,
`text_column`, and a `predicates` map. Each predicate needs `kind: "semantic"`,
`prompt`, `label_column`, `label_source`, and disjoint `positive_values` and
`negative_values` string lists (matching CSV cells or the string form of a JSONL
scalar). Use distinct annotation columns.

```bash
uv run jev-prepare-sembench --source /path/to/verified_reviews.csv \
  --task /path/to/verified_task.json --limit 100 --seed 42 \
  --output data/sembench_movie_verified.jsonl

uv run jev-compare --input data/sembench_movie_verified.jsonl \
  --task /path/to/verified_task.json --max-records 100 --repeats 2 \
  --max-api-calls 1000 --dry-run
```

Remove `--dry-run` and choose a new output path for live evaluation. The importer
samples deterministically, rejects missing/unmapped labels and duplicate IDs, and
writes the Milestone 1 JSONL schema plus a provenance manifest. Only the text column
goes to Jev; ground truth stays in `gold`, and conjunction gold is the logical AND.
Input/task checksums, provenance, prompts, annotation sources, label support, and
selected IDs are kept in manifests. An adjacent sample manifest is checked before
execution, including dry runs, and imported samples require the matching task
definition. The importer cannot certify annotation accuracy.

## Cost

Raw input/output token counts are always recorded when the SDK returns them. Choose
**one** estimator matching the actual endpoint/account contract; do not mix them.
Add `--pricing-source` to record the source/date or account contract. Rates must be
finite and nonnegative. No provider rate is hardcoded or inferred from a model alias.

- **Token pricing:** supply **both** `--input-usd-per-million` and
  `--output-usd-per-million` using verified account rates. If a token class is free,
  explicitly pass zero.
- **Request-credit pricing:** `--credits-per-question-per-block` and
  `--bytes-per-block`, with optional `--usd-per-credit`. The
  [System One API docs](https://docs.system-one.dev/en/docs/api) describe
  model-specific `pricing.creditsPerBlock` and `pricing.bytesPerBlock`. Estimated
  credits = question count × ceil(canonical logical request bytes / bytes per
  block) × credit rate. Bytes cover the UTF-8 model, text, and named
  questions/prompts, **not measured SDK wire bytes**; compression, serialization,
  endpoint changes, and account contracts may change actual charges. Request-based
  estimates still work when token usage is unavailable.

One combined call can evaluate two questions and cost more credits than a
short-circuited sequential plan, so call or token counts alone do not establish a
bill. Billing/account records are required for exact charges.

Token-estimated USD = (input tokens × input rate + output tokens × output rate) / 1e6.
Missing rates or missing token usage produce `null`, never an assumed zero.
These estimates exclude account credits, discounts, taxes, and other charges;
they are not an invoice. SDK retries are disabled so one invocation corresponds
to one attempted request; a failed/ambiguous network request may still incur
provider charges that cannot be recovered from a successful response's usage.

## Artifacts and interpretation

The output directory must be new. It contains:

- `calls.jsonl`: one row per attempted request (errors record their type without exception messages); successful requests, with plan/repeat/id, question
  names, probabilities, model, request id, latency, input/output tokens, and cost.
  Combined-request usage appears **once**, regardless of question count.
- `decisions.jsonl`: one row per input/plan/repeat, with final AND gold/prediction,
  evaluated probabilities, and explicit skipped filters. Skipped probabilities
  remain absent. A row that stops early is still evaluated for final-query quality.
- `runs.jsonl`: each plan/repeat's wall time, model calls, actual execution position,
  and stage input/pass counts. Sequential stage counts expose early-exit savings.
- `summary.json`: calls, semantic question count, runtime, summed API latency,
  token totals, and cost per plan; confusion matrix, precision/recall/F1,
  accuracy, selectivity, and positive support for **every repeat**; repeat decision
  consistency and agreement with fresh independent decisions.
- `summary.csv`: one row per plan with first-repeat quality and all-repeat execution measurements.
- `manifest.json`: completion status, input and selected-record checksums/IDs,
  model, prompts, threshold, plans, pricing assumptions, SDK/Python versions,
  timestamps, and output hashes for complete runs.

Quality is measured over all selected rows against the conjunction of the original
gold labels. Per-stage pass rates in sequential plans are conditional on reaching
that stage, so they are not full-dataset per-filter quality scores. Use Milestone
1 for those. Combined questions may change model outputs; agreement with the
independent plan is reported separately from agreement with gold annotations.

The frozen 100-row sample has no gold-positive anger examples. Conjunctions can
also have zero positives even when each filter has positives separately. Every
repeat reports gold-positive support and flags zero support; a zero positive-class
F1/recall in that case is uninformative. Accuracy alone can be misleading for rare
conjunctions. Use a larger, preselected labeled sample for meaningful quality
comparisons, and report its provenance without overwriting the existing snapshot.

Runtime includes local plan processing and serial requests; API latency measures
only `system_one` calls. Neither is a server-only inference measurement. Raw
request and decision rows are flushed incrementally. On failure, the manifest
is marked failed, partial observations are retained, and no completed summary is
published. Start a new output directory for a rerun; there is no automatic resume.

Existing outputs are never overwritten. No sampling, cost-based ordering optimizer,
or threshold tuning is introduced here.

Tests use fake clients and synthetic fixtures and cover combined request shape/usage,
fixed-order early exits, threshold equality, conjunction quality, order-dependent
call counts, repetitions, billing block boundaries, real SDK serialization over mock
HTTP, budget checks, benchmark import/provenance, missing labels/pricing/usage,
invalid inputs, CLI artifacts, and partial-run failure preservation. They make no
paid model calls.
