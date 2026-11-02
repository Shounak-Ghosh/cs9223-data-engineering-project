# Jev semantic operators — Milestone 1

An intentionally small, **single-filter only** baseline for the DE project proposal's 10/15 milestone: set up Jev access, load a labeled text dataset, and evaluate individual semantic filters reproducibly. It does **not** yet implement combined questions, sequential query execution, or an optimizer.

## 0. Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/) and network access to PyPI, TypeSafe API, and Hugging Face
- `TYPESAFE_API_KEY` (Should not be shared or committed)
- Python 3.11 (uv installs/manages this when needed)

## 1. Install and verify API connectivity

From this directory:

```bash
uv run jev-smoke
```

Alternatively, skip `.env` and use `export TYPESAFE_API_KEY='YOUR_KEY'` in your terminal. `.env` is gitignored. The smoke test sends **one** API request and prints the resolved model, P(yes), and token usage.

## 2. Load a labeled dataset (fixed sample)

```bash
uv run jev-prepare --limit 100 --seed 42
```

This downloads **`SetFit/go_emotions` / `test.jsonl`** through the Hugging Face Hub, resolves the exact Git commit, and writes:

- `data/sample.jsonl`: 100 reproducibly sampled records, each with `id`, `source_row`, `text`, and ground truth for four emotion filters.
- `data/sample.manifest.json`: upstream repo/commit SHA, source SHA256, sample SHA256, sampling seed and parameters.

Prints how many positive ground-truth labels occur per filter. The dataset is multi-label; a record can pass more than one condition. Small samples may contain no positives for rare labels; enlarge the sample before evaluating quality.

For teammates reproducing a sample from its manifest, run:

```bash
uv run jev-prepare --limit 100 --seed 42 --revision REPLACE_WITH_COMMIT_SHA
```

where `COMMIT_SHA` refers to the `dataset_revision` field in `data/sample.manifest.json`. We commit `data/sample.jsonl` and its manifest so later experiments use the exact same frozen records. Dataset annotations are subjective, and their coverage is not interchangeable with all SemBench queries. Dataset: https://huggingface.co/datasets/SetFit/go_emotions

## 3. Verify independent semantic filters

Start small; this makes 10 × 2 × 2 = **40 requests** (each request has ONE Jev Noul question):

```bash
uv run jev-evaluate --max-records 10 --filters gratitude admiration --repeats 2 --output results/pilot.jsonl
```

Run larger only after the pilot behaves as expected:

```bash
uv run jev-evaluate --max-records 100 --filters gratitude admiration anger optimism --repeats 2 --output results/eval100.jsonl
```

Outputs are:

- `results/pilot.jsonl` — one record per `(text, filter, repeat)`, including ground truth, Jev yes-probability, threshold decision, model ID, response request ID, API token usage, and wall-clock latency.
- `results/pilot.manifest.json` — SDK version, Python version, prompt definitions, model, sample checksum and settings.
- `results/pilot.summary.json` — per-filter confusion matrix, precision, recall, F1, gold/predicted selectivity, API calls, input tokens, first-pass quality, and classification repeat agreement.

The default model is **`jev-1.14.0`** (version-pinned rather than the moving `jev-latest` alias). If your account doesn't support that version, run `uv run jev-evaluate --model <supported-version>` after checking available models. Repeated identical API inputs do not *guarantee* identical responses; the measured repeat agreement is an empirical check.

**Interpretation:**

- `gold_selectivity`: label prevalence in the selected records (a property of the dataset).
- `predicted_selectivity`: fraction passing Jev's threshold (a property of the filter/model).
- `precision`, `recall`, `f1`: agreement with the annotation, on first repetition only.
- `repeat_consistency`: fraction of records where all repeated *binary decisions* agree (for a given filter).
- `max_repeat_probability_range`: largest within-record variation in the returned probability for a filter.
- `api_calls` and `total_api_latency_seconds`: actual counts and summed wall-clock API elapsed time, not total parallel or batched wall time.

The fixed 0.5 threshold is a starting assumption, **not calibrated**; don't tune it on the same held-out test sample used for reporting performance.

## 4. Run offline unit tests

```bash
uv run pytest -q
```

Tests cover reproducible sampling, classification metrics, repeat agreement, and enforcing one filter per request using a fake client. Tests make **no Jev API calls**. The live smoke and evaluation commands do require your key and internet.

## 5. What counts as milestone completed?

- [ ] All teammates can run `uv sync` and `uv run pytest -q`.
- [ ] `uv run jev-smoke` succeeds and prints a model version / probability.
- [ ] `uv run jev-prepare --limit 100 --seed 42` creates a JSONL dataset and hashed manifest.
- [ ] Two filters run over at least 10 records, twice each, with saved results and summary.
- [ ] Raw probabilities and binary pass/fail results are retained, including API call counts and token usage.
- [ ] Repeated-call agreement and ground-truth precision/recall/F1 are documented.
- [ ] Data snapshot, prompts, model version, SDK version, and run settings are recorded; `.env` remains uncommitted.

## Next milestone

Keep `jev-evaluate` as the **independent execution** reference and add (a) a combined-question implementation that sends multiple `Noul` questions in a single request for one record and (b) a sequential implementation that skips later filters for failed rows. Preserve the one-question reference for quality checks because batched questions may change model outputs.

### References

- TypeSafe official SDK: https://docs.typesafe.ai/sdk/python
- TypeSafe models: https://docs.typesafe.ai/models
- TypeSafe Noul primitive: https://docs.typesafe.ai/primitives/noul
- SemBench source: https://github.com/SemBench/SemBench
- GoEmotions test data: https://huggingface.co/datasets/SetFit/go_emotions/tree/main
- uv projects: https://docs.astral.sh/uv/guides/projects/
