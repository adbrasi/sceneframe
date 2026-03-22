import json
import os
from pathlib import Path

# Load .env file if present (no extra dependency needed)
_env_file = Path.cwd() / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

DEFAULT_MODEL = "google/gemini-3.1-flash-lite-preview"
DEFAULT_SEGMENT_WORKERS = 1
DEFAULT_CLASSIFY_WORKERS = 50
DEFAULT_MIN_DURATION = 2.0
DEFAULT_MAX_DURATION = 60.0
DEFAULT_RESEGMENT_THRESHOLD = 8.0
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def get_api_key(api_key: str | None = None) -> str:
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError(
            "API key required. Set OPENROUTER_API_KEY env var or use --api-key flag."
        )
    return key


def load_categories(path: Path | None = None) -> list[dict]:
    """Load categories from JSON file. Returns list of {name, description} dicts."""
    categories_path = path or Path.cwd() / "categories.json"
    if not categories_path.exists():
        raise FileNotFoundError(
            f"categories.json not found at {categories_path}. "
            "Use --categories-file to specify its location."
        )
    with open(categories_path) as f:
        data = json.load(f)

    cats = data["categories"]
    # Support both old format (list of strings) and new format (list of dicts)
    if cats and isinstance(cats[0], str):
        return [{"name": c, "description": ""} for c in cats]
    return cats


def load_system_prompt(path: Path | None = None) -> str:
    """Load system prompt from file. Falls back to system_prompt.txt in cwd."""
    prompt_path = path or Path.cwd() / "system_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"system_prompt.txt not found at {prompt_path}. "
            "Use --system-prompt to specify its location."
        )
    return prompt_path.read_text(encoding="utf-8").strip()
