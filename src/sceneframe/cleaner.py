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
# Duplicate detection (cosine similarity on downscaled pixels)
# ---------------------------------------------------------------------------

_FEATURE_SIZE = 64  # Downscale images to 64x64 grayscale for comparison


def _compute_feature_vector(image: np.ndarray) -> np.ndarray:
    """Downscale image to a grayscale feature vector for similarity comparison."""
    resized = cv2.resize(image, (_FEATURE_SIZE, _FEATURE_SIZE))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
    vec = gray.flatten().astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def find_duplicate_labels(
    directory: Path,
    similarity: float = 0.96,
    workers: int = 8,
    chunk_size: int = 1000,
) -> set[str]:
    """Find duplicate pairs by comparing _A images using cosine similarity.

    Downscales each image to 64x64 grayscale, normalizes, and computes
    cosine similarity via matrix multiplication in chunks. This catches
    near-identical frames (eye blinks, slight movements, compression diffs).

    Parameters
    ----------
    similarity : float
        Minimum cosine similarity to consider a duplicate (0-1).
        0.96 = very similar (eye blink, slight movement).
        0.98 = nearly identical (compression artifacts only).
        0.93 = moderately similar (small pose changes).
    chunk_size : int
        Number of images per chunk for similarity computation.
        Controls memory usage: chunk_size * N * 4 bytes per chunk.
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

    # Compute feature vectors in parallel
    def _featurize(path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            return np.zeros(_FEATURE_SIZE * _FEATURE_SIZE, dtype=np.float32)
        return _compute_feature_vector(img)

    vectors: list[np.ndarray] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        vectors = list(tqdm(pool.map(_featurize, paths), total=len(paths), desc="Featurizing"))

    # Stack into matrix: (N, feature_dim)
    features = np.vstack(vectors)  # (N, 4096) float32
    n = len(features)

    # Greedy dedup via chunked cosine similarity (matrix multiplication)
    to_remove_idx: set[int] = set()

    for start in tqdm(range(0, n, chunk_size), desc="Deduplicating"):
        end = min(start + chunk_size, n)
        chunk = features[start:end]  # (chunk, dim)

        # Compare chunk against all images after it
        # Only check forward to avoid double-counting
        remaining = features[start + 1:]  # (N - start - 1, dim)
        if remaining.shape[0] == 0:
            break

        # Cosine similarity via dot product (vectors are pre-normalized)
        sim_matrix = chunk @ remaining.T  # (chunk, N - start - 1)

        for local_i in range(end - start):
            global_i = start + local_i
            if global_i in to_remove_idx:
                continue

            # Similarities for this image against all after it
            # Offset: local_i maps to comparisons starting at (global_i + 1)
            # In sim_matrix row local_i, column 0 = global_i+1, column 1 = global_i+2, etc.
            # But we computed against features[start+1:], so column offset depends on local_i
            col_offset = local_i  # first local_i columns are within-chunk (before global_i+1 relative to start+1)
            # Actually: remaining starts at start+1. Image global_i compares against start+1..N-1.
            # Column j in sim_matrix[local_i] corresponds to global index (start + 1 + j).
            # We only want columns where global index > global_i, i.e. j > global_i - start - 1 = local_i - 1
            # So we want columns from local_i onward.
            row_sims = sim_matrix[local_i, local_i:]  # similarities against images after global_i
            dup_cols = np.where(row_sims >= similarity)[0]
            # Map back to global indices: global_i + 1 + dup_col
            for dc in dup_cols:
                dup_global = global_i + 1 + dc
                if dup_global not in to_remove_idx:
                    to_remove_idx.add(dup_global)

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

    Checks BOTH _A and _B images for each pair. A pair is considered NSFW
    if ANY of its images is classified as NSFW.

    Parameters
    ----------
    keep_nsfw : bool
        If True (default), keep NSFW pairs and remove SFW ones (reverse filter).
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

    # Collect all images (A and B) for classification
    all_labels: list[str] = []
    all_paths: list[Path] = []
    for label, files in sorted(pairs.items()):
        for suffix in ("A", "B"):
            if suffix in files:
                all_labels.append(label)
                all_paths.append(files[suffix])

    if not all_paths:
        return set()

    classifier = pipeline(
        "image-classification",
        model="Falconsai/nsfw_image_detection",
        device=device,
        batch_size=batch_size,
    )

    # Track which labels have at least one NSFW image
    nsfw_labels: set[str] = set()

    for i in tqdm(range(0, len(all_paths), batch_size), desc="NSFW filter"):
        batch_paths = all_paths[i : i + batch_size]
        batch_labels = all_labels[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]

        results = classifier(images)

        for label, result in zip(batch_labels, results):
            top = max(result, key=lambda x: x["score"])
            if top["label"] == "nsfw" and top["score"] >= confidence:
                nsfw_labels.add(label)

    # Decide which pairs to remove based on mode
    all_pair_labels = set(pairs.keys())
    if keep_nsfw:
        # Keep NSFW, remove SFW (pairs with NO nsfw image)
        return all_pair_labels - nsfw_labels
    else:
        # Remove NSFW
        return nsfw_labels


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
    similarity: float = 0.96,
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
        dups = find_duplicate_labels(directory, similarity=similarity, workers=workers)
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
