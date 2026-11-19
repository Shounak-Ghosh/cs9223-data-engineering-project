# cs9223-data-engineering-project
CS9223 Data Engineering Project with Prof. Juliana Freire

# Setup Instructions
Install `uv` package manager if not already done so:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Set up environment by doing the following: 
```bash
uv sync
# Commit the generated uv.lock file to pin dependency versions for teammates.
cp .env.example .env
# Edit .env and replace the placeholder with your key.
```

# Milestone 2

Compare independent, combined, and fixed-order sequential semantic AND queries:

```bash
uv sync --locked
uv run jev-experiment --dry-run
uv run pytest -q
```

See [Milestone 2 instructions](src/jev_milestone_2/README.md) for live runs,
explicit filter orders, cost rates, artifacts, and quality interpretation.

# Milestone 3

Estimate each filter's selectivity, tokens, and latency from a small sample and
compare the estimates with full-data measurements:

```bash
uv sync --locked
uv run pytest -q
uv run jev-filter-profile --help
```

See [Milestone 3 instructions](src/jev_milestone_3/README.md) for the run
sequence, outputs, results, and the recommended sample size (k = 20).

# Milestone 4

Sample-based cost optimizer: estimate filter costs and selectivities from a seeded
sample, choose the cheapest filter order or the combined plan (with a quality
guard), execute it, and compare predicted with actual cost:

```bash
uv sync --locked
uv run pytest -q
uv run jev-optimize --input data/sembench_movie_verified.jsonl \
  --task data/verified_task.json --dry-run
```

See [Milestone 4 instructions](src/jev_milestone_4/README.md) for the cost model,
quality guard, live runs, outputs, and interpretation.
