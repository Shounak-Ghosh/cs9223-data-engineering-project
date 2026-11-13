# Milestone 3: sampling and selectivity estimation

Run commands from the repository root with the existing uv project. Milestone 3
estimates one semantic filter's **selectivity** and **per-call cost** from a
small random sample, then measures how close those estimates are to a full-data
run. These estimates are the per-filter inputs for the 11/05 cost-based
optimizer, which will order filters without running each one over all records.

The filter is chosen with `--filter`, and nothing is hard-coded. The examples use
`acting_quality`. For the teammate's run, `positive` works identically.

| Predicate | Prompt | Gold positives (of 100) |
|---|---|---|
| `acting_quality` | "Does this movie review comment on the quality of the acting or performances?" | 21 |
| `positive` | "Does this movie review express a positive sentiment?" | 71 |

## Definitions

- **Selectivity (primary)** is Jev's **pass rate**: the fraction of records with
  `probability >= threshold` (0.5). It decides how many records reach the next
  filter, so it is the number the optimizer needs. **Gold selectivity** (label
  prevalence) is always reported beside it, and the two are never substituted
  for each other.
- **Full-data reference**: the filter on all 100 records, one question per call.
  Repeat 1 defines the reference, and repeat 2 is used only for repeat
  consistency.
- **Sample estimate** (`estimate.json`, from `sampling.estimate_filter`) is built
  from k records drawn with a seed and without replacement. It contains:
  - pass rate with a 95% Wilson interval;
  - mean input/output tokens, latency and cost per call;
  - extrapolated totals for N records (N × the per-call mean, where N is the
    full input size, not k);
  - precision, recall, F1 and accuracy against gold on the sample.
- **Undefined is `null`, never 0.** A small sample can contain no gold positives
  or no predicted passes. Precision, recall and F1 are then `null`. Missing
  token usage or missing pricing also gives a `null` cost.

## Setup and offline checks

```bash
uv sync --locked
uv run pytest -q
```

Dry runs validate the task, manifest checksum, gold labels and budget. They
print the planned call count and the sampled IDs **without** reading
credentials, importing the SDK, calling Jev or writing files.

## Run sequence

For live runs, put `TYPESAFE_API_KEY` in the gitignored `.env`. Pricing flags
are optional and work exactly as in Milestone 2 (see that README for the
current rates and source).

Commands actually used for the reported results (run on 2026-11-12; 421 calls
in total including one `jev-smoke` check). Run the same block with
`--filter positive` for the second predicate.

```bash
# A. Full-data reference: 100 records x 2 repeats = 200 calls
uv run jev-filter-profile --filter acting_quality --repeats 2 --max-api-calls 200 --dry-run
uv run jev-filter-profile --filter acting_quality --repeats 2 --max-api-calls 200 \
  --input-usd-per-million 0.042 --output-usd-per-million 0 \
  --pricing-source "https://docs.typesafe.ai/models; checked 2026-11-09; Jev 1.14"

# B. One live sample run: 10 calls
uv run jev-filter-profile --filter acting_quality --sample-size 10 --seed 42 \
  --input-usd-per-million 0.042 --output-usd-per-million 0 \
  --pricing-source "https://docs.typesafe.ai/models; checked 2026-11-09; Jev 1.14"

# C. Offline comparison: 0 calls
uv run jev-sample-analysis --full-run results/milestone3/acting_quality-full \
  --live-sample results/milestone3/acting_quality-k10-seed42

# D. Optional plot (matplotlib is not a project dependency)
uv run --with matplotlib python -m jev_milestone_3.plot results/milestone3/acting_quality-analysis

# E. Ordering agreement between the two filters (needs both full runs): 0 calls
uv run jev-ordering-analysis --full-runs results/milestone3/acting_quality-full \
  results/milestone3/positive-full
```

Default output directories are `results/milestone3/<filter>-full`,
`<filter>-k<K>-seed<S>`, `<filter>-analysis` and `<a>-vs-<b>-ordering`.

`--input` and `--task` default to `data/sembench_movie_verified.jsonl` and
`data/verified_task.json`. `--max-api-calls` defaults to 100, so the 200-call
full run has to raise it explicitly. Output directories must not exist yet, and
every file is created exclusively.

### `jev-filter-profile` outputs

| File | Contents |
|---|---|
| `calls.jsonl` | One row per call, streamed as it happens: `id`, `filter`, `repeat`, `gold`, `probability`, `predicted`, `threshold`, `model`, `request_id`, `latency_ms`, `input_tokens`, `output_tokens`, `estimated_cost_usd`. An error row holds only the exception class name. |
| `estimate.json` | `estimate_filter` on repeat 1, extrapolated to the full input size. The optimizer reads these keys, so keep them stable. |
| `manifest.json` | `running`, then `complete` or `failed`; filter, prompt, threshold, requested and resolved models, repeats, `sample_size` (or `"all"`), seed, selected IDs, input and task checksums, pricing and its source, `sdk_max_retries: 0`, SDK and Python versions, and output hashes. |

On failure, the rows written so far are kept and the manifest records
`failed`, the error class and the completed call count. No `estimate.json` is
written.

### `jev-sample-analysis` outputs

It refuses a full run whose status is not `complete`, a run that was made with
`--sample-size`, a live sample whose filter, prompt, threshold or input differ
from the full run, and any `calls.jsonl` that no longer matches its manifest
hash.

| File | Contents |
|---|---|
| `analysis.json` | The reference estimate, aggregates per k, repeat consistency and the live-sample comparison. |
| `analysis.csv` | One row per k, ready for plotting. |
| `manifest.json` | Hashes of the input runs, sizes, draws, seed, versions and output hashes. `api_calls: 0`. |

### `jev-ordering-analysis` outputs

Each draw samples the **same** k record IDs for both filters, which is what
the optimizer would do: run both filters on one sample. Draw `i` uses seed
`seed + i`.

| Column | Meaning |
|---|---|
| `selectivity_order_agreement` | The sampled pass rates rank the two filters the same way as the full data. A sampled tie counts as wrong. |
| `plan_agreement` | The order that minimizes cost(X) + sel(X) × cost(Y) on the sample is the full-data optimum. The cost is the per-call USD estimate, or input tokens or latency when no pricing is given. |
| `mean_regret`, `max_regret` | Full-data cost of the chosen order ÷ optimal full-data cost − 1. This is what a wrong choice actually costs. |

`ordering.json` also records the full-data reference: both pass rates, the
per-call costs, both order costs and the regret of the wrong order.
`ordering.csv` has one row per k, and `manifest.json` holds the input hashes.

## Why offline resampling is valid

The full run already has Jev's probability, tokens and latency for every
record. Drawing k of its repeat-1 rows is statistically the same as calling Jev
on those k records, except for run-to-run model noise, and repeat consistency
measures that noise. So 1000 draws for each of 6 sample sizes cost **zero**
extra calls. Draw `i` uses seed `seed + i`, which makes the analysis exactly
reproducible.

The live k = 10 run checks the whole path end to end. `analysis.json` splits its
error into two parts:

- `full_run_same_ids_estimate`: the same 10 IDs taken from the full run. It
  differs from the full reference only because of which records were drawn.
- `decision_agreement_with_full_run`: the live call vs. the full-run call on
  each record, which is pure model noise.

## How to read `analysis.json` / `analysis.csv` per k

| Column | Meaning |
|---|---|
| `mean_abs_error_selectivity`, `p90_…`, `max_…` | \|sample pass rate − full pass rate\| over the draws. The 90th percentile is nearest-rank. |
| `ci_coverage` | Fraction of draws whose 95% Wilson interval contains the full pass rate. It should be close to 0.95. With N = 100, sampling without replacement makes it run slightly above 0.95 for large k. |
| `mean_ci_width` | How uncertain one sample of size k honestly is. |
| `zero_pass_fraction` | Fraction of samples with no passes at all. Such a sample estimates selectivity 0 and would push a selective filter to the front for the wrong reason. |
| `undefined_precision_fraction`, `undefined_recall_fraction` | Fraction of samples where those metrics are `null`. |
| `mean_rel_error_input_tokens`, `…_latency`, `…_cost` | Relative error of the sampled per-call mean. Because totals are N × the mean, this is also the relative error of the predicted full-run total. It is `null` when the reference is 0 or unknown. |
| `mean_f1`, `std_f1`, `f1_defined_fraction` | Sample F1 over the draws where it is defined. |

## Results

Both predicates were profiled on 2026-11-12. Every model resolved to
`jev-1.14.0`, the threshold was 0.5 (not tuned) and SDK retries were off. All
runs completed with no errors. Sources:
`results/milestone3/{acting_quality,positive}-{full,k10-seed42,analysis}/` and
`results/milestone3/acting_quality-vs-positive-ordering/`.

### 1. Full-data reference (repeat 1, N = 100)

| | `acting_quality` | `positive` |
|---|---|---|
| **Jev pass rate** [95% Wilson CI] | **0.15** [0.093, 0.233] | **0.63** [0.532, 0.718] |
| Gold selectivity | 0.21 | 0.71 |
| TP / FP / FN / TN | 13 / 2 / 8 / 77 | 63 / 0 / 8 / 29 |
| Precision / recall / F1 / accuracy | 0.867 / 0.619 / 0.722 / 0.90 | 1.000 / 0.887 / 0.940 / 0.92 |
| Input / output tokens per call | 313.6 / 21 | 308.6 / 20 |
| Latency per call: mean (median, p95) | 83 ms (80, 118) | 81 ms (78, 122) |
| Estimated cost per 100 calls | $0.00132 | $0.00130 |
| Repeat consistency (identical decisions) | 99 / 100 | 99 / 100 |
| Max / mean probability range across repeats | 0.07 / 0.006 | 0.07 / 0.007 |
| Pass rate, repeat 1 → repeat 2 | 0.15 → 0.16 | 0.63 → 0.64 |

Findings:
- **Jev is conservative at 0.5.** Its pass rate is below the gold rate for
  both predicates (0.15 vs. 0.21, 0.63 vs. 0.71). Almost every error is a
  false negative: 2 false positives in total against 16 false negatives. This
  is exactly why the optimizer must use Jev's pass rate and not label
  prevalence. Planning with gold rates would overestimate how many records
  survive the first filter by about 40% for `acting_quality`.
- **Jev is very stable.** Across two independent repeats, one record per
  predicate flipped its decision, and the largest probability change was 0.07.
- Output tokens are the same on every call (21 and 20). Input tokens track
  review length. Output is billed at $0, so cost is proportional to input
  tokens.

### 2. Selectivity error and CI coverage vs. k (headline)

Results from 1000 offline draws per k (seed 0) out of the repeat-1 rows.

| k | `acting_quality` mean / p90 abs error | CI coverage | `positive` mean / p90 abs error | CI coverage |
|---|---|---|---|---|
| 5 | 0.132 / 0.250 | 0.968 | 0.168 / 0.370 | 0.950 |
| 10 | 0.091 / 0.150 | 0.951 | 0.121 / 0.230 | 0.960 |
| 15 | 0.073 / 0.150 | 0.951 | 0.096 / 0.170 | 0.975 |
| 20 | 0.058 / **0.100** | 0.988 | 0.080 / 0.170 | 0.962 |
| 30 | 0.045 / 0.083 | 0.987 | 0.061 / 0.130 | 0.984 |
| 50 | 0.029 / 0.050 | 0.997 | 0.040 / **0.070** | 0.998 |

Plots: `results/milestone3/acting_quality-analysis/analysis.png` and
`results/milestone3/positive-analysis/analysis.png`.

- The error shrinks roughly as 1/√k. For a given k, `positive` has the larger
  absolute error, because a pass rate near 0.6 has a larger binomial variance
  than one near 0.15.
- **The Wilson interval is honest at every k.** Coverage is ≥ 0.95 in every
  cell. It rises toward 1 for large k because sampling without replacement
  from N = 100 is less variable than the interval assumes, so the interval is
  conservative and never overconfident.

### 3. Zero-pass samples

| k | 5 | 10 | 15 | 20 | 30 | 50 |
|---|---|---|---|---|---|---|
| `acting_quality` | **44.4%** | **17.8%** | 8.1% | 2.6% | 0.4% | 0% |
| `positive` | 0.6% | 0% | 0% | 0% | 0% | 0% |

For a selective filter, small samples often see **no passes at all**. That
gives an estimated selectivity of 0 and leaves precision undefined, which is
why those metrics are `null` rather than 0. A rough guide is 0.85^k (about
44% at k = 5, 20% at k = 10, 4% at k = 20), and sampling without replacement
makes the real rate slightly lower. A zero-pass sample does *not* mean the
filter rejects everything. Its Wilson interval still reaches 0.43 at k = 5 and
0.28 at k = 10, so the uncertainty is visible to the optimizer.

Quality metrics need much larger samples than selectivity does. At k = 10,
sample F1 for `acting_quality` is defined in only 80% of draws, and its value
is 0.76 ± 0.24.

### 4. Token and latency prediction

Mean relative error of the sampled per-call mean. Because the predicted full
total is N × that mean, this is also the relative error of the predicted
100-record total.

| k | 5 | 10 | 20 | 50 |
|---|---|---|---|---|
| Input tokens and cost (both filters) | 1.6% | 1.1% | 0.7–0.8% | 0.4% |
| Latency, `acting_quality` / `positive` | 7.5% / 8.7% | 5.2% / 5.8% | 3.6% / 4.0% | 1.8% / 2.1% |

Tokens and cost are almost free to estimate: 5 records already give 1.6%.
**Latency is the one weak spot, and the cause is run-to-run drift, not
sampling.** The live k = 10 runs below predicted total latency 7% and 13% too
high, even though resampling within a run suggests about 5%. On the same 10
records, the live run averaged 89 ms vs. 84 ms in the full run for
`acting_quality`, and 91 ms vs. 69 ms for `positive`. Latency should be
treated as accurate to about ±15% across runs. Token counts reproduce exactly.

### 5. Live k = 10 sample vs. full reference

| | `acting_quality` | `positive` |
|---|---|---|
| Live pass rate [95% CI] | 0.10 (1/10) [0.018, 0.404] | 0.70 (7/10) [0.397, 0.892] |
| Full reference | 0.15 | 0.63 |
| Abs error / CI contains reference | 0.05 / yes | 0.07 / yes |
| Same 10 IDs taken from the full run | 0.10 | 0.70 |
| Live vs. full-run decisions on those 10 IDs | 10/10 agree (max probability difference 0.01) | 10/10 agree (max probability difference 0.01) |
| Predicted total input tokens (live vs. actual) | 30,640 vs. 31,356 (−2.3%) | 30,140 vs. 30,856 (−2.3%) |
| Predicted total cost (live vs. actual) | $0.00129 vs. $0.00132 (−2.3%) | $0.00127 vs. $0.00130 (−2.3%) |
| Predicted total latency (live vs. actual) | 8.93 s vs. 8.32 s (+7.2%) | 9.07 s vs. 8.05 s (+12.6%) |

For both predicates the live sample matches the same records in the full run
exactly. **All of the selectivity error is sampling error**, meaning which 10
records were drawn, and **none is model noise**. Both errors are smaller than
the typical k = 10 error in the table in section 2 (0.091 and 0.121), and both
intervals contain the reference. This confirms that the offline resampling in
section 2 stands in faithfully for live sampling.

### 6. Recommended minimum k for the optimizer: **k = 20**

The rule was fixed before the runs: take the smallest k with p90 abs error ≤
0.10, zero-pass fraction ≤ 0.05 and CI coverage ≥ 0.93.

- `acting_quality` meets the rule at **k = 20**: p90 error 0.10, 2.6%
  zero-pass samples, coverage 0.988. At k = 15 the p90 error is 0.15 and 8.1%
  of samples are zero-pass.
- `positive` reaches p90 ≤ 0.10 only at k = 50, because a pass rate near 0.6
  is inherently noisier. However, the optimizer's actual decision does not
  need that precision. **Ordering agreement**
  (`acting_quality-vs-positive-ordering/`, same k records for both filters,
  1000 draws):

| k | 5 | 10 | 15 | 20 | 30 | 50 |
|---|---|---|---|---|---|---|
| Correct order (`acting_quality` → `positive`) | 92.1% | 99.0% | 99.9% | **100%** | 100% | 100% |
| Mean regret (extra cost vs. optimum) | 3.2% | 0.41% | 0.04% | **0%** | 0% | 0% |

  The full-data optimum is `acting_quality` first, at 1.51e-5 USD per record
  vs. 2.13e-5 USD per record the other way. The wrong order costs **40.6%
  more**. At k = 20, no draw out of 1000 picked it.

**Recommendation: k = 20 per filter**, sampling the same records for both
filters.
- It is the smallest k at which the selective filter's estimate is reliable,
  with p90 error ≤ 0.10 and under 3% zero-pass risk.
- It ordered these two filters correctly in every draw.
- It predicts token and cost totals to within about 1%.

k = 10 is a defensible cheaper choice for ordering alone (99% correct, 0.4%
expected regret), but it has an 18% chance that the selective filter shows no
passes.

Caveats for the optimizer:
- These two filters are easy to order, because their pass rates of 0.15 and
  0.63 are far apart. Filters with similar selectivities need a larger k, or
  they can be treated as interchangeable, since the regret is small exactly
  when the pass rates are close.
- On this 100-record set, k = 20 per filter is 20% of a full pass, so the
  sample is only cheap relative to N when N is large. Answers from the sample
  can be reused, because sampled records already have both decisions.
- The quality metrics (F1 and so on) are **not** reliable at k = 20 (see
  section 3). The optimizer should take selectivity and cost from samples, not
  accuracy.

### 7. Annotation caveat (Fleiss κ)

The `acting_quality` gold labels are **consensus labels**. The κ = 1.0 recorded
in `data/verified_task.json` was measured *after* consensus: all three
annotators revised their sheets within a minute of each other in commits
`3866ebc` (A, 15 labels), `1441269` (B, 8 labels) and `5474014` (C, 13 labels).

Agreement on the **original, pre-consensus** sheets (`b1a943c` A, `3958bb1` B,
`d223b65` C) was:

| | Pre-consensus | Post-consensus |
|---|---|---|
| Fleiss κ (A, B, C) | **0.454** | 1.000 |
| Cohen κ A–B / A–C / B–C | 0.367 / 0.247 / 0.738 | 1.000 / 1.000 / 1.000 |
| Unanimous items | 71 / 100 | 100 / 100 |
| Positives (A / B / C) | 10 / 25 / 34 | 21 / 21 / 21 |

Report the pre-consensus κ as inter-annotator agreement (moderate), and describe
the gold labels as adjudicated consensus labels. The 0.454 κ is also a reminder
that "comments on the acting" is a fuzzy predicate. Some of Jev's disagreement
with gold may be legitimate ambiguity rather than model error.

## Code map

| Module | Role |
|---|---|
| `sampling.py` | Pure functions: `draw_sample`, `wilson_interval`, `safe_binary_metrics`, `estimate_filter`, `compare` |
| `profile.py` | `run_single_filter`: the live runner, with one Noul question per call and no reuse between repeats |
| `resample.py` | `resampling_analysis`, `repeat_consistency`, `ordering_agreement`: offline, with no SDK |
| `cli.py` | `jev-filter-profile`, `jev-sample-analysis` and `jev-ordering-analysis` |
| `plot.py` | Optional PNG from `analysis.csv` |

Milestone 2's `run_experiment` deliberately rejects single-filter plans, so
Milestone 3 uses its own small runner. The Milestone 1 and 2 code and their
validation are unchanged.
