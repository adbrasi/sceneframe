import asyncio
import base64
import json
import logging
import random
import re
from pathlib import Path

import aiohttp
from tqdm import tqdm

from .config import OPENROUTER_API_URL

logger = logging.getLogger(__name__)

_MIME_MAP = {
    ".mp4": "video/mp4",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
}

_MAX_RETRIES = 5
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_FATAL_STATUSES = {401, 402, 403}


def _build_user_prompt(categories: list[dict]) -> str:
    cat_lines = "\n".join(
        f"- {c['name']}: {c['description']}" for c in categories
    )
    cat_names = "|".join(c["name"] for c in categories)
    return f"""Watch this anime/series video clip carefully and do TWO things:

1. **DESCRIBE** the scene following the system prompt instructions — a detailed, cinematic, present-tense paragraph describing everything that happens.

2. **CLASSIFY** the scene into one of these categories:
{cat_lines}

Respond ONLY with valid JSON, no markdown, no extra text:
{{
    "category": "one of: {cat_names}",
    "confidence": 0.0-1.0,
    "description": "your detailed cinematic description of the scene here",
    "characters_count": number of visible characters
}}"""


def _save_txt_response(clip_path: Path, raw_response: str) -> None:
    """Save raw LLM response to a .txt file with the same name as the clip."""
    txt_path = clip_path.with_suffix(".txt")
    txt_path.write_text(raw_response, encoding="utf-8")


async def classify_clip_async(
    session: aiohttp.ClientSession,
    clip_path: Path,
    api_key: str,
    model: str,
    categories: list[dict],
    system_prompt: str,
    valid_names: set[str],
    user_prompt: str,
) -> dict | None:
    """Classify a single clip using OpenRouter API (async)."""
    file_size = clip_path.stat().st_size
    if file_size > 20 * 1024 * 1024:
        logger.warning(
            "Clip too large (%d MB), skipping: %s",
            file_size // (1024 * 1024), clip_path.name,
        )
        return None

    def _read_and_encode():
        data = clip_path.read_bytes()
        mime = _MIME_MAP.get(clip_path.suffix.lower(), "video/mp4")
        encoded = base64.b64encode(data).decode("utf-8")
        del data
        return mime, encoded

    mime_type, base64_video = await asyncio.to_thread(_read_and_encode)
    data_url = f"data:{mime_type};base64,{base64_video}"
    del base64_video

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "video_url", "video_url": {"url": data_url}},
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }

    for attempt in range(_MAX_RETRIES):
        try:
            async with session.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=180),
            ) as response:
                # Fatal errors: don't retry (auth, billing, forbidden)
                if response.status in _FATAL_STATUSES:
                    body = await response.text()
                    logger.error(
                        "Fatal HTTP %d for %s: %s", response.status, clip_path.name, body[:200]
                    )
                    return {"_fatal": True, "_status": response.status}

                if response.status in _RETRY_STATUSES:
                    if attempt < _MAX_RETRIES - 1:
                        # Full jitter to avoid thundering herd with 50 workers
                        max_wait = 2 ** (attempt + 1)
                        wait = random.uniform(0, max_wait)
                        logger.warning(
                            "HTTP %d on %s, retry %d/%d in %.1fs...",
                            response.status, clip_path.name, attempt + 1, _MAX_RETRIES, wait,
                        )
                        await asyncio.sleep(wait)
                        continue
                    else:
                        logger.error(
                            "HTTP %d on final attempt for %s", response.status, clip_path.name
                        )
                        return None

                response.raise_for_status()
                data = await response.json()

            raw_content = data["choices"][0]["message"]["content"]

            # Save raw response to .txt file
            _save_txt_response(clip_path, raw_content)

            content = raw_content.strip()
            content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
            content = content.strip()

            result = json.loads(content)

            if result.get("category") not in valid_names:
                result["category"] = "other"

            result["raw_response"] = raw_content
            return result

        except asyncio.TimeoutError:
            logger.warning(
                "Timeout for %s (attempt %d/%d)",
                clip_path.name, attempt + 1, _MAX_RETRIES,
            )
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
        except aiohttp.ClientError as e:
            logger.warning(
                "Request failed for %s (attempt %d/%d): %s",
                clip_path.name, attempt + 1, _MAX_RETRIES, e,
            )
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(
                "Failed to parse response for %s (attempt %d/%d): %s",
                clip_path.name, attempt + 1, _MAX_RETRIES, e,
            )
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(1)

    logger.error("All %d retries exhausted for %s", _MAX_RETRIES, clip_path.name)
    return None


async def _classify_batch(
    clips: list[Path],
    api_key: str,
    model: str,
    categories: list[dict],
    system_prompt: str,
    workers: int,
    progress_callback=None,
) -> list[tuple[Path, dict | None]]:
    """Classify a batch of clips concurrently using a shared session and semaphore."""
    semaphore = asyncio.Semaphore(workers)
    valid_names = {c["name"] for c in categories}
    user_prompt = _build_user_prompt(categories)
    fatal_error = asyncio.Event()

    connector = aiohttp.TCPConnector(limit=workers, limit_per_host=workers)
    async with aiohttp.ClientSession(connector=connector) as session:

        async def _limited_classify(clip: Path) -> tuple[Path, dict | None]:
            if fatal_error.is_set():
                return clip, None
            async with semaphore:
                result = await classify_clip_async(
                    session, clip, api_key, model, categories, system_prompt,
                    valid_names, user_prompt,
                )
                # Detect fatal errors (auth/billing) and abort remaining
                if isinstance(result, dict) and result.get("_fatal"):
                    fatal_error.set()
                    return clip, None
                return clip, result

        tasks = [_limited_classify(clip) for clip in clips]

        results = []
        with tqdm(total=len(tasks), desc="Classifying") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                pbar.update(1)
                results.append(result)
                if progress_callback:
                    progress_callback(result)

        if fatal_error.is_set():
            logger.error("Fatal API error detected — aborting classification")

    return results


def _atomic_write_jsonl(path: Path, entries: list[dict]) -> None:
    """Write JSONL atomically via tmp file + rename to prevent corruption on crash."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for entry in entries:
            clean = {k: v for k, v in entry.items() if k != "raw_response"}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")
    tmp.replace(path)


def classify_clips(
    clips_dir: Path,
    api_key: str,
    model: str,
    categories: list[dict],
    system_prompt: str,
    workers: int,
    existing_metadata: list[dict] | None = None,
) -> list[dict]:
    """Classify all clips in a directory. Returns list of classification results.

    Saves intermediate progress atomically so work is not lost on crash.
    """
    video_extensions = {".mp4", ".mkv", ".avi", ".webm", ".mov"}
    clips_dir = Path(clips_dir).resolve()

    clips = sorted(
        [f for f in clips_dir.iterdir() if f.suffix.lower() in video_extensions]
    )

    if not clips:
        logger.error("No video clips found in %s", clips_dir)
        return []

    classified_files = set()
    if existing_metadata:
        classified_files = {
            m.get("file") for m in existing_metadata if "category" in m and "file" in m
        }

    clips_to_process = [c for c in clips if c.name not in classified_files]

    if classified_files:
        logger.info(
            "Resuming: %d already classified, %d remaining",
            len(classified_files), len(clips_to_process),
        )

    if not clips_to_process:
        logger.info("All clips already classified")
        return existing_metadata if existing_metadata is not None else []

    logger.info("Classifying %d clips with %d workers", len(clips_to_process), workers)
    results = list(existing_metadata) if existing_metadata else []
    results_index = {m.get("file"): m for m in results if m.get("file")}

    # Progress file for crash recovery
    progress_path = clips_dir / "segments_metadata.jsonl"
    classified_count = 0

    def _save_progress(item: tuple[Path, dict | None]):
        nonlocal classified_count
        clip, classification = item
        if classification:
            if clip.name in results_index:
                results_index[clip.name].update(classification)
            else:
                entry = {"file": clip.name}
                entry.update(classification)
                results.append(entry)
                results_index[clip.name] = entry
            classified_count += 1

            # Save progress every 10 clips (atomic write)
            if classified_count % 10 == 0:
                _atomic_write_jsonl(progress_path, results)

    batch_results = asyncio.run(
        _classify_batch(
            clips_to_process, api_key, model, categories, system_prompt, workers,
            progress_callback=_save_progress,
        )
    )

    # Process any results not handled by callback
    for clip, classification in batch_results:
        if classification and clip.name not in results_index:
            entry = {"file": clip.name}
            entry.update(classification)
            results.append(entry)
            results_index[clip.name] = entry

    # Final save (atomic)
    _atomic_write_jsonl(progress_path, results)
    logger.info("Classification complete: %d/%d successful", classified_count, len(clips_to_process))

    return results
