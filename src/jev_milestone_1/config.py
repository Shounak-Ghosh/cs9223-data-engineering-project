"""Fixed filter definitions: use names and prompts unchanged across repetitions."""

MODEL = "jev-1.14.0"  # Versioned model, not the shifting jev-latest alias.
DATASET_REPO = "SetFit/go_emotions"
DATASET_FILE = "test.jsonl"
THRESHOLD = 0.5

FILTER_PROMPTS = {
    "gratitude": "Does this text express gratitude or thanks?",
    "admiration": "Does this text express admiration for someone or something?",
    "anger": "Does this text express anger?",
    "optimism": "Does this text express optimism or hope about the future?",
}
