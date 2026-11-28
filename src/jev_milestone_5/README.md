# Milestone 5: initial end-to-end experiments and cost-model refinement

Implements the proposal's 11/12 milestone: "Compare optimized execution against
always-combined and fixed/sequential baselines on an initial set of queries.
Identify implementation problems and refine the cost model if needed."

Run commands from the repository root with the existing uv project. This milestone
adds three things:

1. **A replay benchmark** (`jev-collect`, `jev-benchmark`). Every plan in this
   project is built from two request types: a one-question call and a combined
   call. Once every row has both recorded, any plan, and the optimizer with any
   seed, sample size or sampling mode, can be replayed offline with **no new API
   calls**. That makes it affordable to compare the optimizer against every
   baseline over 20 seeds instead of one run.
2. **A cost-model refinement**: `jev-optimize --sample-mode cheap`, which cuts the
   sampling overhead that Milestone 4 identified as its main weakness.
3. **Results** for the SemBench Movie query, computed from calls already committed
   by Milestones 2 and 4, and for four GoEmotions queries (two and three filters)
   recorded with `jev-collect`.

## 0. Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/); network access to the TypeSafe API only for `jev-collect` and live `jev-optimize` runs
- `TYPESAFE_API_KEY` in the gitignored `.env` for live runs (never commit it)
- Committed inputs: `data/sembench_movie_verified.jsonl`, `data/sample.jsonl`, `results/milestone2-sembench/`, `results/milestone4/`

## 1. Setup and offline checks

```bash
uv sync --locked
uv run pytest -q
uv run jev-collect --input data/sample.jsonl \
  --query gratitude,admiration --query admiration,optimism \
  --query gratitude,optimism --query gratitude,admiration,optimism --dry-run
uv run jev-optimize --input data/sembench_movie_verified.jsonl \
  --task data/verified_task.json --sample-mode cheap --dry-run
```

Dry runs validate inputs and print call bounds without reading credentials,
sending requests, or writing files. The `jev-collect` dry run prints
`"total_calls": 700` (300 one-question calls and 400 combined calls). The cheap
`jev-optimize` dry run prints:

```json
{
  "records": 100,
  "sample_rows": 20,
  "execution_rows": 80,
  "sample_calls": 30,
  "execution_max_calls": 160,
  "worst_case_calls": 220,
  "guard_rows": 5,
  "sample_fill_max_calls": 30
}
```

 `tests/test_milestone5.py` uses a fake client
and makes **no API calls**. Besides unit tests, it checks that the replay agrees
with live execution: every replayed plan and every replayed optimizer run (both
sampling modes, combined-wins and sequential-wins cases) produces the same chosen
plan, call count, input tokens, and F1 as the live code path.

## 2. Why the replay is trustworthy

| Check | Result |
| --- | --- |
| Repeated requests (same row, same questions) across the Milestone 2 and 4 runs | **447 / 447** identical input tokens; **446 / 447** identical decisions |
| Replayed baselines vs Shounak's measured Milestone 2 totals | Identical: 32,956 (combined), 36,362 and 50,889 tokens (fixed orders) |
| Replayed optimizer (full mode, k = 20, seed 42) vs the live Milestone 4 run | Identical: combined chosen, 140 calls, 45,192 tokens, F1 0.727, from either data source |

Input tokens depend only on the request, so replayed **costs are exact**. Decisions
can vary slightly between repeated requests (1 in 447 here); the replay uses the
first recorded answer. Latency is replayed from the recording and is not
re-measured.

## 3. Refinement: cheap sampling mode

In Milestone 4 (now `--sample-mode full`, still the default), each sample row gets
every one-question call **and** a combined call: `k × (n + 1)` calls. At 100 rows
that was 41% of the optimizer's tokens, and the combined answers were thrown away.

`--sample-mode cheap` uses the fact that a combined call already returns a
decision for every filter:

| Step | Full mode (Milestone 4) | Cheap mode (Milestone 5) |
| --- | --- | --- |
| Pass rates per filter | One-question calls on all `k` rows | Combined calls on all `k` rows |
| One-question cost per filter | All `k` rows | `g` guard rows (`--guard-rows`, default 5) |
| Combined cost per row | All `k` rows | All `k` rows |
| Quality guard (combined vs independent agreement) | All `k` rows | `g` guard rows |
| Sampling calls | `k × (n + 1)` | `k + g × n` |
| Answers for sample rows | Independent sample run | Combined sample calls if combined wins; otherwise guard rows from their one-question calls and the other `k - g` rows by the chosen plan (`sample_fill/`) |

The cost model, candidate plans, tie-breaking, and quality guard are unchanged
(`build_estimates` in `jev_milestone_4/optimizer.py` now feeds both modes and the
replay). With `g = k` cheap mode is identical to full mode, which the benchmark
confirms. The trade-off: the guard sees `g` rows instead of `k`, so with `g = 5` it
cannot detect a combined-vs-independent disagreement rate below about 1 in 5.

## 4. Run sequence

Outputs must go to new directories; nothing is overwritten.

### A. SemBench Movie replay: 0 API calls (already committed)

```bash
uv run jev-benchmark --input data/sembench_movie_verified.jsonl \
  --calls results/milestone2-sembench/calls.jsonl \
          results/milestone4/sembench-k20-seed42-all/sample/calls.jsonl \
          results/milestone4/sembench-k20-seed42-all/execution/calls.jsonl \
  --query acting_quality,positive --label sembench-movie --guard-rows 3 5 10 \
  --output-dir results/milestone5/sembench-replay
```

Shounak's Milestone 2 run covers both filters on every row (each sequential order
asks its first filter on all 100 rows) and the combined call on every row, so no
collection run is needed. The Milestone 4 calls are loaded second; they only feed
the repeat check.

### B. Live check of cheap mode on SemBench: at most 220 calls

```bash
uv run jev-optimize --input data/sembench_movie_verified.jsonl \
  --task data/verified_task.json --k 20 --seed 42 --sample-mode cheap --guard-rows 5 \
  --input-usd-per-million 0.042 --output-usd-per-million 0 \
  --pricing-source "https://docs.typesafe.ai/models; checked 2026-11-09; Jev 1.14" \
  --output-dir results/milestone5/sembench-live-cheap-k20-g5-seed42
```

The replay predicts: combined chosen, **110 calls, 36,035 input tokens, F1 0.727**
(vs 140 calls and 45,192 tokens for the same seed in full mode).

### C. Collect GoEmotions calls: 700 calls

```bash
uv run jev-collect --input data/sample.jsonl \
  --query gratitude,admiration --query admiration,optimism \
  --query gratitude,optimism --query gratitude,admiration,optimism \
  --input-usd-per-million 0.042 --output-usd-per-million 0 \
  --pricing-source "https://docs.typesafe.ai/models; checked 2026-11-09; Jev 1.14" \
  --output-dir results/milestone5/goemotions-collect
```

This records 3 one-question calls per row (the union of filters) and one combined
call per row per query: 300 + 400 calls. `anger` is excluded because the frozen
sample has no positive anger labels.

### D. GoEmotions replay: 0 API calls

```bash
uv run jev-benchmark --input data/sample.jsonl \
  --calls results/milestone5/goemotions-collect/*/calls.jsonl \
  --query gratitude,admiration --query admiration,optimism \
  --query gratitude,optimism --query gratitude,admiration,optimism \
  --label goemotions --guard-rows 3 5 10 \
  --output-dir results/milestone5/goemotions-replay
```

`jev-benchmark` defaults: 20 seeds (0 to 19), `k` in {10, 20}, both modes,
objective `input_tokens`, agreement threshold 0.95, projections to 100, 1,000 and
10,000 rows. It fails with a list of gaps if the recorded calls do not cover every
row for a query.

## 5. Outputs

| `jev-collect` artifact | Contents |
| --- | --- |
| `manifest.json` | Status, queries, prompts, input checksums, prices, model and SDK versions |
| `independent/{calls,decisions,runs}.jsonl` | One-question calls for every filter on every row |
| `combined-<query>/{calls,decisions,runs}.jsonl` | One combined call per row for that query |

| `jev-benchmark` artifact | Contents |
| --- | --- |
| `report.md` | Ready-to-paste tables (also printed) |
| `baselines.csv` | Every plan over all rows: calls, tokens, latency, USD, precision, recall, F1 |
| `optimizer_runs.csv` | One row per query, mode, guard size, `k`, and seed: chosen plan, sampling/fill/execution/total cost, predicted vs actual cost per row, F1 |
| `optimizer_summary.csv` | Per query, mode, guard size, and `k`: share of seeds that picked the oracle plan, plan regret, cost ratios vs always-combined, always-separate, best and worst fixed order and oracle, prediction error, guard rejections, F1 |
| `projection.csv` | Cost ratios at larger input sizes and break-even row counts |
| `summary.json` | Everything above, nested |
| `manifest.json` | Inputs and checksums, settings, repeat check, `api_calls_made: 0` |

### How to read the results

- **Always-combined**: every filter in one Jev request per row (`combined:...`).
- **Always-separate**: every filter as its own request on every row, no early exit
  (`independent:...`). These are the two baselines the project guidelines name.
- **Fixed order**: filters as separate requests, one after another, skipping later
  filters once a row fails (`sequential:...`). Best and worst are the cheapest and
  most expensive orders in hindsight.
- **Oracle plan**: the candidate plan with the lowest measured cost on all rows,
  known only in hindsight.
- **Picked oracle plan**: share of seeds whose chosen plan was the oracle plan.
- **Plan regret**: extra cost of the chosen plan over the oracle plan on all rows.
- **Cost ratios** (`vs ...`): the optimizer's total cost, **including sampling**,
  divided by running one fixed strategy on every row. Negative means cheaper.
- **Prediction error**: `(predicted - actual) / actual` cost per row of the chosen
  plan on the execution rows.
- **Break-even rows**: input size above which the optimizer is cheaper than that
  strategy, assuming new rows cost what recorded rows cost on average.

**Interpretation:**

- Replayed costs are exact; replayed decisions use the first recorded answer
  (1 in 447 repeated requests disagreed).
- Projections reuse the recorded per-row costs and do not model drift between the
  sample and the rest of a larger table.
- Every result is for objective `input_tokens`, which is proportional to USD for
  `jev-1.14.0` (output tokens are free).

## 6. Results: SemBench Movie, `acting_quality AND positive`

100 rows, 14 gold positives, objective `input_tokens`, 20 seeds. Source:
`results/milestone5/sembench-replay/`.

### Baselines over all 100 rows

| Plan | Input tokens | Calls | F1 |
| --- | --- | --- | --- |
| `combined:acting_quality,positive` (always-combined, oracle) | 32,956 | 100 | 0.727 |
| `sequential:acting_quality,positive` (best order) | 36,362 | 116 | 0.727 |
| `sequential:positive,acting_quality` (worst order) | 50,889 | 164 | 0.727 |
| `independent:acting_quality,positive` (always-separate) | 62,212 | 200 | 0.727 |

### Optimizer at 100 rows, mean over 20 seeds

| Mode | k | g | Picked oracle | Sampling tokens | Total tokens | vs always-combined | vs always-separate | vs best order | vs worst order | Prediction error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full (M4) | 10 | | 90% | 9,546 | 39,554 | +20.0% | -36.4% | +8.8% | -22.3% | -1.2% |
| full (M4) | 20 | | 95% | 19,070 | 45,594 | +38.3% | -26.7% | +25.4% | -10.4% | -0.3% |
| cheap | 10 | 5 | 90% | 6,435 | 36,597 | +11.0% | -41.2% | +0.6% | -28.1% | -1.1% |
| **cheap** | **20** | **5** | **95%** | **9,728** | **36,501** | **+10.8%** | **-41.3%** | **+0.4%** | **-28.3%** | **-0.3%** |
| cheap | 20 | 3 | 100% | 8,481 | 34,833 | +5.7% | -44.0% | -4.2% | -31.6% | +0.2% |
| cheap | 20 | 10 | 95% | 12,837 | 39,516 | +19.9% | -36.5% | +8.7% | -22.3% | -0.3% |

Every setting kept F1 at 0.727 and matched the independent answers on every row.
The combined plan passed the quality guard in every seed (100% agreement).

### Projected to larger inputs (same per-row costs)

| Mode | k | g | vs always-combined @1,000 | @10,000 | vs best order @1,000 | @10,000 | Break-even vs best order | vs worst order | vs always-separate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full (M4) | 20 | | +4.3% | +0.9% | -5.5% | -8.6% | 385 rows | 70 rows | 43 rows |
| **cheap** | **20** | **5** | **+1.5%** | **+0.6%** | **-8.0%** | **-8.8%** | **104 rows** | **19 rows** | **12 rows** |
| cheap | 20 | 3 | +0.6% | +0.1% | -8.8% | -9.3% | 55 rows | 11 rows | 6 rows |

### Findings

- **Cheap sampling halves the overhead.** At k = 20 it cuts sampling from 19,070 to
  9,728 tokens. The optimizer's gap to always-combined at 100 rows drops from
  +38% to +11%; it now matches the best fixed order at 100 rows and beats it from
  about 104 rows. For the same seed as Milestone 4 (42), the total drops from
  45,192 to 36,035 tokens (-20%).
- **Against always-separate, the optimizer saves 41%** at 100 rows (cheap,
  k = 20), including sampling, and breaks even after about 12 rows. Even the
  worst fixed order saves 18% over always-separate, because skipping the second
  question for rows that already failed is free.
- **The plan choice is reliable.** 95% to 100% of seeds picked the oracle plan at
  k = 20. When a seed missed, it picked the best fixed order (regret +10.3%),
  never the worst one (+54%).
- **Milestone 4's under-prediction was sampling noise, not bias.** Across 20 seeds
  the mean prediction error is -0.3% at k = 20 (range -12% to +2%); seed 42 sat
  near the bad end.
- **Smaller guards trade safety for cost.** `g = 3` was cheapest and picked the
  oracle every time here, but a 3-row guard is weak evidence that combining does
  not change answers. `g = 5` is the default and the recommended setting.
- **Joint vs independence estimates** coincide for two filters (the first stage
  always sees every row).
- **The live run matches the replay.** Run B (cheap, k = 20, g = 5, seed 42) chose
  the combined plan with 110 calls, 36,035 input tokens and F1 0.727, exactly as the
  replay predicted. Source: `results/milestone5/sembench-live-cheap-k20-g5-seed42/`.

### About "the optimizer costs more than always-combined"

On this query, always-combined **is** the oracle, so no strategy that has to
discover the best plan can beat it. The optimizer's extra cost is the price of
finding out, and it is a fixed cost: with cheap sampling it is +11% at 100 rows,
+1.5% at 1,000 and +0.6% at 10,000. What always-combined cannot do is notice when
it is the wrong choice:

- when one filter is selective enough that a sequential order is cheaper (with a
  combined call costing about 1.06 one-question calls here, that happens when the
  first filter passes fewer than about 6% of rows), and
- when asking several questions in one request changes the answers, which only
  the quality guard checks.

The GoEmotions queries (section 7) test the first case directly, and show it is
rare with Jev's pricing: on four queries with filters positive on only 4% to 9% of
rows, a sequential order won once, by 0.9%.

## 7. Results: GoEmotions, four queries

100 rows from the frozen GoEmotions sample, 20 seeds, objective `input_tokens`.
Source: `results/milestone5/goemotions-replay/` (calls recorded in
`results/milestone5/goemotions-collect/`, 700 calls).

### Baselines over all 100 rows (input tokens)

| Query | Always-combined | Best order | Worst order | Always-separate | Oracle plan |
| --- | --- | --- | --- | --- | --- |
| gratitude AND admiration | **30,697** | 31,630 | 36,164 | 58,194 | combined |
| admiration AND optimism | **30,997** | 32,196 | 36,236 | 58,494 | combined |
| gratitude AND optimism | **30,797** | 31,639 | 32,176 | 58,294 | combined |
| gratitude AND admiration AND optimism | 32,497 | **32,221** | 37,661 | 87,491 | `sequential:gratitude,optimism,admiration` |

A combined call costs about 1.06 one-question calls for two filters and 1.11 for
three. Even with filters this rare, the best order was only 3% to 4% more expensive
than combining for two filters, and 0.9% cheaper for three.

### Optimizer at 100 rows, mean over 20 seeds

| Query | Setting | Picked oracle | vs always-combined | vs always-separate | vs best order | vs worst order | Prediction error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gratitude AND admiration | cheap, k20, g5 | 45% | +19.3% | -37.0% | +15.8% | +1.3% | -3.3% |
| gratitude AND admiration | cheap, k10, g3 | 75% | +8.2% | -42.9% | +5.1% | -8.1% | -2.5% |
| admiration AND optimism | cheap, k20, g5 | 60% | +17.0% | -38.0% | +12.6% | +0.1% | -2.6% |
| admiration AND optimism | cheap, k10, g3 | 70% | +9.0% | -42.3% | +4.9% | -6.8% | -3.0% |
| gratitude AND optimism | cheap, k20, g5 | 35% | +21.2% | -36.0% | +18.0% | +16.0% | -4.3% |
| gratitude AND optimism | cheap, k10, g3 | 45% | +11.5% | -41.1% | +8.5% | +6.7% | -4.6% |
| three filters | cheap, k20, g5 | 10% | +27.7% | -52.6% | +28.8% | +10.2% | -7.4% |
| three filters | cheap, k10, g3 | 5% | +14.8% | -57.3% | +15.8% | -0.9% | -7.3% |
| three filters | full, k20 (M4) | 10% | +55.5% | -42.2% | +56.8% | +34.2% | -7.3% |

Projected to 10,000 rows, the optimizer (cheap, k20, g5) is within +0.8% to +2.2%
of always-combined and about 46% (two filters) to 63% (three filters) cheaper than
always-separate.

### Findings

- **Always-combined was the cheapest strategy or within 0.9% of it on all five
  queries** (SemBench and GoEmotions). Jev charges little extra for more questions
  in one request, so a sequential order only pays off when the first filter rejects
  almost every row; even then the saving is around 1%.
- **The optimizer always beats always-separate**: with cheap sampling, by 33% to
  57% at 100 rows including sampling. It beats the worst fixed order by 22% to 32%
  on SemBench, but not reliably on GoEmotions at 100 rows, where even the worst
  order is only 2% to 17% more expensive than the best.
- **It does not beat always-combined.** With plans this close, sampling costs more
  than the best plan could save: +8% to +28% at 100 rows (cheap, g ≤ 5) and +1% to
  +2% at 10,000.
- **Picking the oracle is hard when plans are nearly tied.** The optimizer picked
  the oracle in 30% to 75% of seeds on two-filter queries and 0% to 10% on the
  three-filter query, where seven plans sit within a few percent. Its mistakes were
  cheap because the alternatives were close.
- **Cost predictions are biased low when plans are close** (-2.0% to -7.8%,
  largest with seven candidates). This is the optimizer's curse: the plan whose
  sampled cost happened to be lowest is the one chosen, so its estimate is low on
  average. On SemBench, where combining wins clearly, the mean error was -0.3%.
- **Smaller samples are better here.** k = 10 with g = 3 beat k = 20 on every
  GoEmotions query, because a larger sample does not fix a near-tie but costs more.
- **Quality.** The queries have 0 or 1 gold positive rows, so F1 is not meaningful.
  On `gratitude AND optimism`, the combined plan produced one more false positive
  than the sequential orders (F1 0.500 vs 0.667), a reminder that combining can
  change answers even when the 5-row guard does not catch it.

### Answer to the guideline questions so far

- **Which filter should run first?** The rarer one, as the cost model predicts; the
  best order beat the worst by 2% to 17% on GoEmotions and 40% on SemBench.
- **When should filters run together?** With Jev, almost always: on five queries
  combining was the cheapest or within 0.9%. A simple rule (combine unless a filter
  is rejected by nearly every sampled row, and check answers on a few rows) captures
  nearly all of the savings with far less sampling than the full optimizer.

## 8. What counts as milestone completed?

- [x] Optimized execution compared against always-combined, always-separate, and every fixed order, including sampling cost, on the SemBench query (20 seeds, 2 sample sizes, 2 modes).
- [x] Implementation problem identified (sampling overhead, discarded combined answers) and the cost model refined (`--sample-mode cheap`).
- [x] Replay validated against measured runs (exact token totals, exact Milestone 4 seed-42 result).
- [x] Live cheap-mode check (run B) matches the replay prediction (110 calls, 36,035 tokens, F1 0.727).
- [x] GoEmotions queries collected and benchmarked (runs C and D); section 7 filled in.
- [ ] All teammates can run `uv sync --locked` and `uv run pytest -q`; results committed under `results/milestone5/`; `.env` uncommitted.

## 9. Code map

| File | Role |
| --- | --- |
| `src/jev_milestone_4/optimizer.py` | `build_estimates` (shared), `estimate_from_sample` (full), `estimate_from_cheap_sample` (cheap) |
| `src/jev_milestone_4/cli.py` | `jev-optimize --sample-mode {full,cheap} --guard-rows g` |
| `src/jev_milestone_5/collect.py` | `jev-collect`: record one-question and combined calls once per row |
| `src/jev_milestone_5/replay.py` | Call store, plan replay, optimizer replay, aggregation, projections |
| `src/jev_milestone_5/benchmark.py` | `jev-benchmark`: CSV, JSON and Markdown reports |
| `tests/test_milestone5.py` | Offline tests, including replay-equals-live checks |

## Next milestone

11/19 and 11/26 (expanded evaluation and failure cases), based on the findings
above:

- **Tie tolerance**: prefer combining unless a sequential plan is predicted to be
  cheaper by a margin (for example 5%), to avoid paying for near-ties and reduce the
  optimizer's curse.
- **Stop sampling early** once the decision is clear, instead of a fixed `k`.
- **Adaptive re-planning** during execution, which the replay benchmark can test
  without new API calls.
- A latency objective, and a larger input to measure the projections directly.

### References

- Project 8 guidelines, Direction 2 (Combining and Ordering Semantic Operators); related work cited there: Abacus [3] and LOTUS [1]
- TypeSafe models and pricing: https://docs.typesafe.ai/models
- TypeSafe Python SDK: https://docs.typesafe.ai/sdk/python
- TypeSafe Noul primitive: https://docs.typesafe.ai/primitives/noul
- SemBench Movie scenario: https://github.com/SemBench/SemBench/blob/main/src/scenario/movie/README.md
- GoEmotions test data: https://huggingface.co/datasets/SetFit/go_emotions/tree/main