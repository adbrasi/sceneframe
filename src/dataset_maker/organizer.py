import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def organize_dataset(
    clips_dir: Path,
    output_dir: Path,
    metadata: list[dict],
    *,
    move: bool = False,
) -> Path:
    """Organize clips into category folders and write metadata.jsonl.

    Also transfers .txt files (raw LLM responses) alongside video files.
    When move=True, files are moved instead of copied (saves disk space in pipelines).
    """
    clips_dir = Path(clips_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"

    transfer = shutil.move if move else shutil.copy2
    organized_entries = []

    for entry in metadata:
        category = entry.get("category", "other")
        filename = entry.get("file")
        if not filename:
            continue

        source_file = clips_dir / filename
        if not source_file.exists():
            logger.warning("Clip not found: %s", source_file)
            continue

        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        dest_file = category_dir / filename
        if dest_file.exists():
            source_stem = Path(entry.get("source", "unknown")).stem
            dest_file = category_dir / f"{source_stem}_{filename}"

        transfer(str(source_file), dest_file)

        # Transfer .txt file (raw LLM response) if it exists
        txt_source = source_file.with_suffix(".txt")
        if txt_source.exists():
            txt_dest = dest_file.with_suffix(".txt")
            transfer(str(txt_source), txt_dest)

        # Write clean entry without raw_response (already in .txt)
        clean_entry = {k: v for k, v in entry.items() if k != "raw_response"}
        organized_entries.append(clean_entry)

    # Write metadata only for successfully organized clips
    with open(metadata_path, "w") as f:
        for entry in organized_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(
        "Organized %d clips into %s, metadata at %s",
        len(organized_entries),
        output_dir,
        metadata_path,
    )
    return metadata_path


def load_metadata(path: Path) -> list[dict]:
    """Load metadata from JSONL file."""
    path = Path(path).resolve()
    if not path.exists():
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def save_metadata(path: Path, metadata: list[dict]) -> None:
    """Save metadata to JSONL file."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
