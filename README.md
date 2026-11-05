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

The [Milestone 2 dataset and comparison guide](MILESTONE2.md) describes
`uv run jev-compare --dry-run`, SemBench task imports, verified labels,
request-credit pricing, and CSV exports. GoEmotions remains the default
development sample; SemBench-derived tasks are imported separately.
