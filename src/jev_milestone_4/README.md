# Milestone 4: sample-based cost optimizer

Implements the proposal's 11/05 milestone: "Develop a simple cost model that uses
the sampled estimates to determine filter ordering and whether conditions should
be evaluated together or separately."

Run commands from the repository root with the existing uv project. Milestones 1
to 3 remain available and unchanged. `jev-optimize` reuses the Milestone 2
execution engine (`run_experiment`) and the Milestone 3 sampler (`draw_sample`);
it adds the cost model, the plan choice, and a predicted-vs-actual comparison. It
does **not** yet re-plan during execution (adaptive optimization is a stretch goal
in the proposal).

## 0. Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/) and network access to the TypeSafe API for live runs
- `TYPESAFE_API_KEY` in the gitignored `.env` or exported in your shell (never commit it)
- The frozen SemBench Movie task from Milestone 2: `data/sembench_movie_verified.jsonl` and `data/verified_task.json`

## 1. Setup and offline checks

```bash
uv sync --locked
uv run pytest -q
uv run jev-optimize --input data/sembench_movie_verified.jsonl \
  --task data/verified_task.json --dry-run
```

The dry run validates the data, task, filters, sample size, objective, and prices,
then prints the call bounds. It does not read credentials, import the SDK, submit
Jev requests, or write files. For the SemBench task with the defaults it prints:

```json
{
  "records": 100,
  "sample_rows": 20,
  "execution_rows": 80,
  "sample_calls": 60,
  "execution_max_calls": 160,
  "worst_case_calls": 220
}
```

`tests/test_optimizer.py` uses a fake client and makes **no API calls**. It covers
selective-first and cheaper-first ordering, choosing the combined plan, the
quality guard rejecting it, a correlated three-filter case where multiplying
selectivities picks the wrong order, every objective, predicted vs actual and
regret, the dry-run credential guard, budget and argument rejection, and all three
CLI modes end to end.

## 2. How the optimizer works

One `jev-optimize` run has four phases.

| Phase | What happens | Calls (n filters, k sample rows, m remaining rows) |
| --- | --- | --- |
| 1. Sample | Draw `k` rows with `draw_sample(records, k, seed)`; run the **independent** plan (one question per call) and the **combined** plan (all questions in one call) on them | `k × (n + 1)` |
| 2. Estimate | Per filter: Jev pass rate and mean cost per call. Combined: mean cost per row and agreement with independent decisions | 0 |
| 3. Choose | Score every fixed sequential order and the combined plan; pick the cheapest plan that passes the quality guard | 0 |
| 4. Execute | Run the chosen plan on the `m` non-sample rows (or every candidate with `--evaluate-all-plans`) and compare predicted with measured cost | at most `m × n` (chosen) or `m × (n! × n + 1)` (all plans) |

Sample rows are never re-executed. In the end-to-end result they are answered by
the independent sample run, so the optimizer still produces an answer for every
row, and its total cost includes the sampling overhead.

## 3. Cost model

For a sequential order `f1, f2, ..., fn`, expected cost per row is

```text
cost(f1) + P(pass f1) × cost(f2) + P(pass f1 and f2) × cost(f3) + ...
```

- `cost(f)` is the mean measured cost of a single-question call for `f` on the sample.
- `P(...)` is the fraction of **sample rows** that pass every earlier filter
  (`reach_fractions` in `plan.json`). It is estimated jointly from the sample,
  not by multiplying single selectivities, so correlated filters are handled.
- `plan.json` also stores `independence_assumption_cost_per_row` for each order,
  so the report can show where the independence assumption would mislead.

The combined plan's expected cost per row is its measured mean cost per call.
Jev shares the record's state across questions in one request, so a combined
call usually costs less than the sum of separate calls.

`--objective` sets the cost unit:

| Objective | Cost of one call | Notes |
| --- | --- | --- |
| `input_tokens` (default) | SDK input tokens | Proportional to USD for Jev 1.14, whose output tokens are free |
| `usd` | Token-priced USD | Requires both `--input-usd-per-million` and `--output-usd-per-million` |
| `latency` | API latency in seconds | Serial wall-clock per SDK call, not server-only inference time |
| `calls` | 1 per request | Ignores request size |

Ties break deterministically: lower cost, then sequential before combined, then
plan name. At most 6 filters are supported, because orders are enumerated
exhaustively (6! = 720).

Prices follow Milestone 2: checked on **2026-11-09**, `jev-1.14.0` costs USD 0.042
per million input tokens and output is free. Recheck before a new run; see the
[Milestone 2 pricing section](../jev_milestone_2/README.md#model-and-pricing-typesafe).

## 4. Quality guard

Asking several questions in one request can change Jev's answers (Milestone 2
records these disagreements). Cost alone would usually pick the combined plan, so
the combined plan is only **allowed** when its final AND decision matches the
independent decision on at least `--agreement-threshold` of the sample rows
(default 0.95). When it fails, `plan.json` keeps its predicted cost but marks it
`"allowed": false` with the measured agreement in `reason`.

`--agreement-threshold 0` disables the guard. Choose the threshold before looking
at the reporting results; do not tune it on the same rows used to report quality.

## 5. Run sequence

Outputs must go to a new directory; the command never overwrites or resumes.
`--max-api-calls` (default 1000) rejects a run whose worst case exceeds the budget
before any request is sent.

```bash
# A. Validate the exact settings: 0 calls
uv run jev-optimize --input data/sembench_movie_verified.jsonl \
  --task data/verified_task.json --k 20 --seed 42 \
  --input-usd-per-million 0.042 --output-usd-per-million 0 \
  --pricing-source "https://docs.typesafe.ai/models; checked 2026-11-09; Jev 1.14" \
  --output-dir results/milestone4/sembench-k20-seed42-all \
  --evaluate-all-plans --dry-run

# B. Optimize, then run every candidate on the 80 remaining rows: at most 460 calls
uv run jev-optimize --input data/sembench_movie_verified.jsonl \
  --task data/verified_task.json --k 20 --seed 42 \
  --input-usd-per-million 0.042 --output-usd-per-million 0 \
  --pricing-source "https://docs.typesafe.ai/models; checked 2026-11-09; Jev 1.14" \
  --output-dir results/milestone4/sembench-k20-seed42-all \
  --evaluate-all-plans

# C. Optional: optimizer as it would be used, chosen plan only: at most 220 calls
uv run jev-optimize --input data/sembench_movie_verified.jsonl \
  --task data/verified_task.json --k 20 --seed 42 \
  --input-usd-per-million 0.042 --output-usd-per-million 0 \
  --pricing-source "https://docs.typesafe.ai/models; checked 2026-11-09; Jev 1.14" \
  --output-dir results/milestone4/sembench-k20-seed42

# D. Optional: plan only, no execution: 60 calls
uv run jev-optimize --input data/sembench_movie_verified.jsonl \
  --task data/verified_task.json --k 20 --seed 42 --plan-only \
  --output-dir results/milestone4/sembench-k20-seed42-plan
```

Run B is the one to commit and report: because every candidate runs on the same
rows, it shows whether the chosen plan was actually the cheapest. Other useful
flags: `--filters` (subset of task predicates), `--max-records`, `--objective`,
`--threshold` (default 0.5, not calibrated), and `--model` (default `jev-1.14.0`).
Milestone 5 adds `--sample-mode cheap --guard-rows g`, which takes pass rates from
the combined calls and asks one-question calls only on `g` sample rows; see the
[Milestone 5 README](../jev_milestone_5/README.md#3-refinement-cheap-sampling-mode).

## 6. Outputs

| Artifact | Contents |
| --- | --- |
| `manifest.json` | Status (`running`, `complete`, `failed`), mode, settings, sample IDs, input/task checksums, prices and source, call bounds, resolved models, SDK and Python versions, timestamps |
| `sample/calls.jsonl`, `sample/decisions.jsonl`, `sample/runs.jsonl` | Milestone 2 records for the independent and combined runs on the sample |
| `plan.json` | `estimates` (per-filter pass rate and cost, combined cost and agreement, sample decision matrix) and `choice` (every candidate's predicted cost, guard result, chosen plan) |
| `execution/calls.jsonl`, `execution/decisions.jsonl`, `execution/runs.jsonl` | Milestone 2 records for the executed plan(s) on the remaining rows |
| `summary.json` | Execution summary, predicted vs actual, sampling overhead, end-to-end optimizer cost and quality |

Field meanings are the same as in the
[Milestone 2 artifacts table](../jev_milestone_2/README.md#artifacts-and-interpretation).
Raw records are flushed as they arrive; an interrupted run keeps partial files and
a `failed` manifest.

## 7. How to read `plan.json` and `summary.json`

- `estimates.per_filter.<name>.selectivity`: Jev pass rate on the sample (not gold prevalence).
- `estimates.combined.agreement_with_independent`: fraction of sample rows where the combined and independent AND decisions match; this is what the quality guard checks.
- `choice.candidates[]`: one entry per plan, sorted with allowed plans first and then by predicted cost. Each has `predicted_cost_per_row`, `reach_fractions`, `independence_assumption_cost_per_row`, `allowed`, and `reason`.
- `summary.predicted_vs_actual.plans.<plan>`: predicted and measured cost per row and `relative_error = (predicted - actual) / actual`.
- `summary.predicted_vs_actual.chosen_was_cheapest` and `regret`: only defined with `--evaluate-all-plans`. `regret = actual(chosen) / actual(cheapest) - 1`; 0 means the optimizer picked the best plan.
- `summary.sampling_overhead`: calls, tokens, USD, and latency spent on the sample.
- `summary.optimizer_end_to_end`: calls, tokens, USD, latency, and quality for all rows (sample rows plus the chosen plan's rows), including the sampling overhead.
- `summary.execution`: the Milestone 2 summary of the executed plans, including per-plan precision/recall/F1 and stage pass counts.

**Interpretation:**

- Compare the optimizer's end-to-end cost against each baseline's measured cost
  per row × 100 rows, not against the 80 execution rows alone.
- Small samples can miss rare filter combinations, so later reach fractions can be 0.
  Milestone 3 found k = 20 gives the full-data filter order in 100% of resamples
  for this task; the joint estimate is noisier than single selectivities.
- `acting_quality` has Jev recall 0.62 (Milestone 3), so a cheaper plan does not
  hide low quality of the filter itself; report F1 next to cost.
- Repeated identical requests are not guaranteed to return identical answers, so
  predicted and actual costs for sequential plans can differ even with no bias.

## 8. What counts as milestone completed?

- [ ] All teammates can run `uv sync --locked` and `uv run pytest -q`.
- [x] `jev-optimize --dry-run` prints call bounds for the SemBench task without credentials.
- [x] A live `--evaluate-all-plans` run on the SemBench task completes with `"status": "complete"`.
- [x] `plan.json` shows each candidate's predicted cost, the guard decision, and the chosen plan.
- [x] `summary.json` shows predicted vs actual cost per plan and the regret of the chosen plan.
- [x] End-to-end optimizer cost (with sampling overhead) and F1 are compared with the always-combined and fixed-order baselines.
- [ ] Settings, sample IDs, prices and source, model and SDK versions are recorded; results are committed under `results/milestone4/`; `.env` remains uncommitted.

## 9. Results

Run B: `results/milestone4/sembench-k20-seed42-all` (SemBench Movie, 100 rows,
`k = 20`, seed 42, objective `input_tokens`, `jev-1.14.0`, threshold 0.5, guard
0.95). Costs are SDK input tokens per row; execution covers the 80 non-sample rows.

### 1. Sample estimates (k = 20)

| Filter | Jev passes | Pass rate | Mean input tokens / call |
| --- | --- | --- | --- |
| `acting_quality` | 2 / 20 | 0.10 | 308.4 |
| `positive` | 10 / 20 | 0.50 | 303.4 |
| Combined call (both questions) | n/a | n/a | 324.4 |

The combined call costs only 16 to 21 tokens more than one single-question call,
because Jev shares the review text across questions in one request. Combined and
independent decisions agreed on **20 / 20** sample rows (1.000), so the quality
guard allowed the combined plan.

### 2. Predicted vs actual cost per plan (80 execution rows)

| Plan | Allowed | Predicted cost / row | Actual cost / row | Relative error | Calls | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| `combined:acting_quality,positive` (chosen) | yes | 324.4 | 330.9 | -1.9% | 80 | 0.737 |
| `sequential:acting_quality,positive` | yes | 338.7 | 365.5 | -7.3% | 93 | 0.737 |
| `sequential:positive,acting_quality` | yes | 457.6 | 521.7 | -12.3% | 134 | 0.737 |

- **Chosen plan:** `combined:acting_quality,positive`, the lowest predicted cost.
  It was also the cheapest measured plan, so **regret = 0**.
- **Ranking:** the predicted ranking of all three plans matched the measured ranking.
- **Ordering:** among sequential plans, the more selective `acting_quality` first
  was 30% cheaper than `positive` first (365.5 vs 521.7 tokens per row).
- **Estimation error:** every plan was under-predicted, and more so the more it
  depends on pass rates. The sample under-estimated both pass rates compared with
  the execution rows: `acting_quality` 0.10 vs 13 / 80 = 0.16, and `positive`
  0.50 vs 54 / 80 = 0.68. The combined plan does not depend on pass rates, so its
  error (-1.9%) only reflects slightly longer reviews in the execution rows.
- **Joint vs independence estimate:** with two filters the first stage always sees
  every row, so both estimates are identical here (`reach_fractions` 1.0, 0.1 and
  1.0, 0.5). They only differ with three or more filters.

### 3. Quality

All three plans made the **same final decision on every execution row** (0
prediction disagreements against combined): precision 1.00, recall 0.58, F1 0.737
(7 TP, 0 FP, 5 FN, 68 TN; 12 gold positives). Choosing a cheaper plan cost no
quality on this task. The low recall comes from the `acting_quality` filter itself
(Milestone 3 measured recall 0.62), not from the plan.

Optimizer output over all 100 rows (sample rows answered by the independent sample
run): precision 1.00, recall 0.57, F1 0.727 (8 TP, 0 FP, 6 FN, 86 TN).

### 4. End-to-end cost including sampling (100 rows)

| Approach | Input tokens | Calls | Relative to optimizer |
| --- | --- | --- | --- |
| Optimizer (60 sampling calls + chosen plan on 80 rows) | 45,192 | 140 | baseline |
| Always combined, all 100 rows (actual/row × 100) | 33,085 | 100 | 27% fewer tokens |
| Best fixed order, `acting_quality` first (actual/row × 100) | 36,553 | ~116 | 19% fewer tokens |
| Worst fixed order, `positive` first (actual/row × 100) | 52,168 | ~168 | 15% more tokens |

The 60 sampling calls (18,724 tokens, 41% of the optimizer's total) dominate at
100 rows. Sampling costs 12,107 tokens more than running the chosen plan on those
20 rows. Holding the measured per-row costs fixed, the optimizer beats the worst
fixed order after about 63 rows and the best fixed order after about 350 rows; it
never beats always-combined when combined is the plan it picks. Total measured
cost of the optimizer run was USD 0.0019 at USD 0.042 per million input tokens.

### 5. Takeaways

- The cost model chose correctly (regret 0) and ranked all plans correctly from 20 rows.
- For this task, a combined request is the cheapest plan with no quality loss.
- Pass-rate estimates from 20 rows were biased low here, so sequential costs were
  under-predicted by 7 to 12%. Larger `k` or several seeds would show how much of
  this is sampling noise (Milestone 3 reported the variance across seeds).
- The sampling overhead only pays off on larger inputs. The 11/12 and 11/19
  experiments should report break-even sizes and repeat the run over several seeds.

## 10. Code map

| File | Role |
| --- | --- |
| `src/jev_milestone_4/optimizer.py` | Pure cost model, no I/O: sample estimates, joint reach fractions, plan scoring, quality guard, predicted vs actual |
| `src/jev_milestone_4/cli.py` | `jev-optimize`: validation, call bounds, sample, choose, execute, write artifacts |
| `tests/test_optimizer.py` | Offline tests with a fake client |

## Next milestone

11/12 and 11/19: run the optimizer across more queries and selectivity patterns
(several seeds and `k` values, GoEmotions filter pairs with different
selectivities, and a three-filter query) and compare against the always-combined
and fixed-order baselines. If time permits, add adaptive re-planning: update the
plan when the observed pass rates during execution drift from the sample estimates.

### References

- TypeSafe models and pricing: https://docs.typesafe.ai/models
- TypeSafe Python SDK: https://docs.typesafe.ai/sdk/python
- TypeSafe Noul primitive: https://docs.typesafe.ai/primitives/noul
- SemBench Movie scenario: https://github.com/SemBench/SemBench/blob/main/src/scenario/movie/README.md
