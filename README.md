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
