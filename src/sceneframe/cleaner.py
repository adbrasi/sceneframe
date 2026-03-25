"""Image pair cleaning: solid color removal, deduplication, NSFW filtering."""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Pattern for pair files: NNNNNN_A.jpg, NNNNNN_B.jpg, NNNNNN_C.jpg
_PAIR_RE = re.compile(r"^(\d+)_([ABC])\.jpg$", re.IGNORECASE)

# Popcount lookup table for fast hamming distance computation
_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def scan_pairs(directory: Path) -> dict[str, dict[str, Path]]:
    """Scan directory for image pairs. Returns {label: {"A": path, "B": path, ...}}."""
    pairs: dict[str, dict[str, Path]] = {}
    for f in sorted(directory.iterdir()):
        if not f.is_file():
            continue
        m = _PAIR_RE.match(f.name)
        if m:
            label, suffix = m.groups()
            pairs.setdefault(label, {})[suffix.upper()] = f
    return pairs


# ---------------------------------------------------------------------------
# Solid color detection
# ---------------------------------------------------------------------------

def is_solid_color(image: np.ndarray, std_threshold: float = 12.0) -> bool:
    """Check if an image is mostly a single solid color.

    Uses standard deviation across channels on a downscaled version.
    A truly solid image has std ~0; we allow up to std_threshold for noise.
    """
    if image is None:
        return True
    small = cv2.resize(image, (64, 64))
    for c in range(small.shape[2] if small.ndim == 3 else 1):
        channel = small[:, :, c] if small.ndim == 3 else small
        if channel.std() >= std_threshold:
            return False
    return True


def find_solid_color_labels(
    directory: Path,
    std_threshold: float = 12.0,
    workers: int = 8,
) -> set[str]:
    """Find pair labels that contain a solid-color image."""
    pairs = scan_pairs(directory)
    labels_to_remove: set[str] = set()

    tasks: list[tuple[str, Path]] = []
    for label, files in pairs.items():
        for path in files.values():
            tasks.append((label, path))

    def _check(task: tuple[str, Path]) -> str | None:
        label, path = task
        img = cv2.imread(str(path))
        return label if is_solid_color(img, std_threshold) else None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in tqdm(pool.map(_check, tasks), total=len(tasks), desc="Solid colors"):
            if result is not None:
                labels_to_remove.add(result)

    return labels_to_remove


# ---------------------------------------------------------------------------
# Perceptual duplicate detection (dhash)
# ---------------------------------------------------------------------------

def _compute_dhash_bytes(image: np.ndarray, hash_size: int = 16) -> bytes:
    """Compute difference hash as raw bytes. Returns hash_size*hash_size/8 bytes."""
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
    diff = gray[:, 1:] > gray[:, :-1]
    return np.packbits(diff.flatten()).tobytes()


def find_duplicate_labels(
    directory: Path,
    hash_size: int = 16,
    threshold: int = 12,
    workers: int = 8,
) -> set[str]:
    """Find duplicate pairs by comparing _A images using dhash.

    Parameters
    ----------
    hash_size : int
        Hash grid size. 16 → 256-bit hash (32 bytes). Higher = more precise.
    threshold : int
        Maximum hamming distance to consider a duplicate.
        With hash_size=16 (256 bits), threshold=12 ≈ 95% similarity.
        Common values: 8 (97%), 12 (95%), 16 (94%).
    """
    pairs = scan_pairs(directory)

    labels: list[str] = []
    paths: list[Path] = []
    for label, files in sorted(pairs.items()):
        if "A" in files:
            labels.append(label)
            paths.append(files["A"])

    if len(paths) < 2:
        return set()

    # Compute hashes in parallel
    def _hash(path: Path) -> bytes:
        img = cv2.imread(str(path))
        if img is None:
            return b"\x00" * (hash_size * hash_size // 8)
        return _compute_dhash_bytes(img, hash_size)

    raw_hashes: list[bytes] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        raw_hashes = list(tqdm(pool.map(_hash, paths), total=len(paths), desc="Hashing"))

    # Build numpy array for vectorized comparison
    hash_len = len(raw_hashes[0])
    hash_array = np.frombuffer(b"".join(raw_hashes), dtype=np.uint8).reshape(len(raw_hashes), hash_len)

    # Greedy duplicate detection: for each image, mark all near-duplicates
    n = len(hash_array)
    to_remove_idx: set[int] = set()

    for i in tqdm(range(n), desc="Deduplicating"):
        if i in to_remove_idx:
            continue
        # Vectorized hamming distance: XOR + popcount
        xor = hash_array[i] ^ hash_array[i + 1:]  # (remaining, hash_len)
        distances = _POPCOUNT_TABLE[xor].sum(axis=1)  # (remaining,)
        dups = np.where(distances <= threshold)[0] + i + 1
        to_remove_idx.update(dups.tolist())

    return {labels[i] for i in to_remove_idx}


# ---------------------------------------------------------------------------
# NSFW filter (optional, requires torch + transformers)
# ---------------------------------------------------------------------------

def find_nsfw_labels(
    directory: Path,
    keep_nsfw: bool = True,
    batch_size: int = 32,
    device: str | None = None,
    confidence: float = 0.5,
) -> set[str]:
    """Filter images by NSFW content using Falconsai/nsfw_image_detection.

    Parameters
    ----------
    keep_nsfw : bool
        If True (default), keep NSFW images and remove SFW ones (reverse filter).
    confidence : float
        Minimum confidence to classify as NSFW.
    """
    try:
        import torch  # noqa: F401
        from PIL import Image
        from transformers import pipeline
    except ImportError:
        raise RuntimeError(
            "NSFW filter requires torch and transformers. "
            "Install with: pip install torch transformers Pillow"
        )

    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = scan_pairs(directory)
    labels: list[str] = []
    paths: list[Path] = []
    for label, files in sorted(pairs.items()):
        if "A" in files:
            labels.append(label)
            paths.append(files["A"])

    if not paths:
        return set()

    classifier = pipeline(
        "image-classification",
        model="Falconsai/nsfw_image_detection",
        device=device,
        batch_size=batch_size,
    )

    to_remove: set[str] = set()

    for i in tqdm(range(0, len(paths), batch_size), desc="NSFW filter"):
        batch_paths = paths[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]

        results = classifier(images)

        for label, result in zip(batch_labels, results):
            top = max(result, key=lambda x: x["score"])
            is_nsfw = top["label"] == "nsfw" and top["score"] >= confidence

            if keep_nsfw and not is_nsfw:
                to_remove.add(label)
            elif not keep_nsfw and is_nsfw:
                to_remove.add(label)

    return to_remove


# ---------------------------------------------------------------------------
# Orphan cleanup
# ---------------------------------------------------------------------------

def find_orphan_labels(directory: Path) -> set[str]:
    """Find pair labels where A or B is missing (incomplete pairs)."""
    pairs = scan_pairs(directory)
    return {label for label, files in pairs.items() if "A" not in files or "B" not in files}


# ---------------------------------------------------------------------------
# Main cleaning pipeline
# ---------------------------------------------------------------------------

def clean_directory(
    directory: Path,
    remove_solid: bool = True,
    remove_dups: bool = True,
    nsfw: bool = False,
    keep_nsfw: bool = True,
    nsfw_confidence: float = 0.5,
    nsfw_batch_size: int = 32,
    nsfw_device: str | None = None,
    hash_threshold: int = 12,
    solid_threshold: float = 12.0,
    workers: int = 8,
    dry_run: bool = False,
) -> dict[str, int]:
    """Run the full cleaning pipeline on a directory of image pairs.

    Steps (in order):
    1. Remove solid-color pairs
    2. Remove duplicate pairs (dhash)
    3. NSFW filter (optional)
    4. Remove orphaned pairs (missing A or B)

    Returns dict with counts for each step.
    """
    stats = {
        "solid_removed": 0,
        "duplicates_removed": 0,
        "nsfw_removed": 0,
        "orphans_removed": 0,
        "total_removed": 0,
        "remaining": 0,
    }

    all_to_remove: set[str] = set()

    # Step 1: solid colors
    if remove_solid:
        solid = find_solid_color_labels(directory, solid_threshold, workers)
        stats["solid_removed"] = len(solid)
        all_to_remove.update(solid)
        logger.info("Solid color pairs to remove: %d", len(solid))

    # Step 2: duplicates
    if remove_dups:
        dups = find_duplicate_labels(directory, threshold=hash_threshold, workers=workers)
        new_dups = dups - all_to_remove
        stats["duplicates_removed"] = len(new_dups)
        all_to_remove.update(dups)
        logger.info("Duplicate pairs to remove: %d", len(new_dups))

    # Step 3: NSFW filter
    if nsfw:
        nsfw_labels = find_nsfw_labels(
            directory, keep_nsfw, nsfw_batch_size, nsfw_device, nsfw_confidence
        )
        new_nsfw = nsfw_labels - all_to_remove
        stats["nsfw_removed"] = len(new_nsfw)
        all_to_remove.update(nsfw_labels)
        logger.info("NSFW filtered pairs to remove: %d", len(new_nsfw))

    # Delete marked files
    if not dry_run and all_to_remove:
        pairs = scan_pairs(directory)
        for label in all_to_remove:
            if label in pairs:
                for path in pairs[label].values():
                    path.unlink(missing_ok=True)

    # Step 4: orphan cleanup
    orphans = find_orphan_labels(directory)
    if not dry_run and orphans:
        pairs = scan_pairs(directory)
        for label in orphans:
            if label in pairs:
                for path in pairs[label].values():
                    path.unlink(missing_ok=True)
    stats["orphans_removed"] = len(orphans)

    stats["total_removed"] = (
        stats["solid_removed"]
        + stats["duplicates_removed"]
        + stats["nsfw_removed"]
        + stats["orphans_removed"]
    )

    # Count remaining
    remaining_pairs = scan_pairs(directory)
    stats["remaining"] = len(remaining_pairs)

    return stats
