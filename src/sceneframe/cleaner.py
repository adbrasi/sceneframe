"""Image pair cleaning: solid color, blur, duplicates, character, and NSFW filtering."""

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Pattern for pair files: NNNNNN_A.jpg, NNNNNN_B.jpg, NNNNNN_C.jpg
_PAIR_RE = re.compile(r"^(\d+)_([ABC])\.jpg$", re.IGNORECASE)


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
# Blur detection (Laplacian variance on _A images)
# ---------------------------------------------------------------------------

JPEG_QUALITY = 95


def is_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
    """Check if an image is blurry using Laplacian variance.

    A sharp image has high variance (many diverse edges), a blurry one
    has low variance (everything smoothed out).
    """
    if image is None:
        return True
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < threshold


def find_blur_labels(
    directory: Path,
    threshold: float = 100.0,
    workers: int = 8,
) -> set[str]:
    """Find pair labels where _A image is blurry (motion blur, out of focus)."""
    pairs = scan_pairs(directory)
    tasks: list[tuple[str, Path]] = []
    for label, files in pairs.items():
        if "A" in files:
            tasks.append((label, files["A"]))

    if not tasks:
        return set()

    def _check(task: tuple[str, Path]) -> str | None:
        label, path = task
        img = cv2.imread(str(path))
        return label if is_blurry(img, threshold) else None

    labels_to_remove: set[str] = set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in tqdm(pool.map(_check, tasks), total=len(tasks), desc="Blur check"):
            if result is not None:
                labels_to_remove.add(result)

    return labels_to_remove


def retry_blur_pairs(
    directory: Path,
    labels_to_retry: set[str],
    threshold: float = 100.0,
    max_retries: int = 3,
    frame_advance: int = 12,
) -> set[str]:
    """Try alternative frames for pairs where _A is blurry.

    Advances frame_advance frames forward per attempt (up to max_retries).
    Overwrites _A directly if a non-blurry replacement is found.

    Returns the set of labels that still need to be removed.
    """
    metadata = _load_metadata(directory)
    if not metadata:
        logger.warning("No pairs_metadata.jsonl found — cannot retry blurry pairs")
        return labels_to_retry

    retryable = {l for l in labels_to_retry if l in metadata}
    no_metadata = labels_to_retry - retryable

    if no_metadata:
        logger.info("No metadata for %d blurry pairs — will be removed without retry", len(no_metadata))
    if not retryable:
        return labels_to_retry

    logger.info("Retrying blur for %d pairs with alternative frames...", len(retryable))

    # Group by video to open each one only once
    by_video: dict[str, list[str]] = {}
    for label in retryable:
        video = metadata[label]["video"]
        by_video.setdefault(video, []).append(label)

    saved = 0
    for video_path_str, video_labels in by_video.items():
        video_path = Path(video_path_str)
        if not video_path.exists():
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        try:
            for label in video_labels:
                meta = metadata[label]
                frame_info = meta["frame_a"]
                idx = frame_info["index"]
                scene_start = frame_info["scene_start"]
                scene_end = frame_info["scene_end"]

                for attempt in range(1, max_retries + 1):
                    alt_idx = idx + (frame_advance * attempt)
                    if alt_idx >= scene_end:
                        break

                    cap.set(cv2.CAP_PROP_POS_FRAMES, alt_idx)
                    ret, alt_frame = cap.read()
                    if not ret:
                        continue

                    if not is_blurry(alt_frame, threshold):
                        final_path = directory / f"{label}_A.jpg"
                        cv2.imwrite(
                            str(final_path), alt_frame,
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                        )
                        retryable.discard(label)
                        saved += 1
                        # Delete stale control images
                        for stale in ("_C.jpg", "_image_base.jpg"):
                            (directory / f"{label}{stale}").unlink(missing_ok=True)
                        break
        finally:
            cap.release()

    logger.info("Blur retry: %d saved, %d still removed", saved, len(retryable))
    return retryable | no_metadata


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
    similarity: float = 0.93,
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
# NSFW retry: try alternative frames before giving up
# ---------------------------------------------------------------------------

def _load_metadata(directory: Path) -> dict[str, dict]:
    """Load pairs_metadata.jsonl into a label-keyed dict."""
    import json

    meta_path = directory / "pairs_metadata.jsonl"
    if not meta_path.exists():
        return {}

    metadata = {}
    for line_num, line in enumerate(meta_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skipping truncated JSON at line %d in %s", line_num, meta_path)
            continue
        metadata[entry["label"]] = entry
    return metadata


def retry_nsfw_pairs(
    directory: Path,
    labels_to_retry: set[str],
    keep_nsfw: bool = True,
    batch_size: int = 32,
    device: str | None = None,
    confidence: float = 0.5,
) -> set[str]:
    """Try alternative frames for pairs that failed NSFW filter.

    For each failed pair, classifies A and B individually to determine which
    frame(s) triggered the filter, then reads the metadata to find the source
    video, extracts a frame ~1s forward/backward within scene bounds for the
    offending frame(s), re-classifies the pair, and replaces the image only
    if it now passes.

    Replacement frames are written to temporary paths (*_retry.jpg) and only
    promoted to final paths if the pair passes re-classification.

    Returns the set of labels that STILL need to be removed (retry failed).
    """
    try:
        import torch  # noqa: F401
        from PIL import Image
        from transformers import pipeline
    except ImportError:
        raise RuntimeError(
            "NSFW retry requires torch and transformers. "
            "Install with: pip install torch transformers Pillow"
        )

    # Clean up stale _retry.jpg temp files from previous interrupted runs
    for stale in directory.glob("*_retry.jpg"):
        stale.unlink(missing_ok=True)

    metadata = _load_metadata(directory)
    if not metadata:
        logger.warning("No pairs_metadata.jsonl found — cannot retry NSFW pairs")
        return labels_to_retry

    # Filter to labels that have metadata
    retryable = {l for l in labels_to_retry if l in metadata}
    no_metadata = labels_to_retry - retryable

    if no_metadata:
        logger.info("No metadata for %d pairs — will be removed without retry", len(no_metadata))
    if not retryable:
        return labels_to_retry

    logger.info("Retrying NSFW for %d pairs with alternative frames...", len(retryable))

    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    classifier = pipeline(
        "image-classification",
        model="Falconsai/nsfw_image_detection",
        device=device,
        batch_size=batch_size,
    )

    # Step 1: Batch-classify A and B to find which frames triggered NSFW
    step1_labels: list[str] = []
    step1_suffixes: list[str] = []
    step1_paths: list[Path] = []
    for label in sorted(retryable):
        for suffix in ("A", "B"):
            path = directory / f"{label}_{suffix}.jpg"
            if path.exists():
                step1_labels.append(label)
                step1_suffixes.append(suffix)
                step1_paths.append(path)

    flagged_suffixes: dict[str, set[str]] = {}
    for i in tqdm(range(0, len(step1_paths), batch_size), desc="NSFW retry: identifying"):
        batch_paths = step1_paths[i : i + batch_size]
        batch_labels = step1_labels[i : i + batch_size]
        batch_suffixes = step1_suffixes[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]

        results = classifier(images)
        for lbl, suffix, result in zip(batch_labels, batch_suffixes, results):
            top = max(result, key=lambda x: x["score"])
            is_nsfw = top["label"] == "nsfw" and top["score"] >= confidence
            should_flag = (not is_nsfw) if keep_nsfw else is_nsfw
            if should_flag:
                flagged_suffixes.setdefault(lbl, set()).add(suffix)

    # Labels with no flagged suffixes already pass — no retry needed
    already_pass = retryable - set(flagged_suffixes.keys())

    # Group by video to open each one only once
    by_video: dict[str, list[str]] = {}
    for label in flagged_suffixes:
        video = metadata[label]["video"]
        by_video.setdefault(video, []).append(label)

    # Step 2: Extract alternative frames only for flagged suffixes, write to temp paths
    # Track which (label, suffix) pairs got a temp replacement
    replaced: dict[str, set[str]] = {}  # label -> set of suffixes replaced

    for video_path_str, video_labels in by_video.items():
        video_path = Path(video_path_str)
        if not video_path.exists():
            logger.warning("Video not found for retry: %s", video_path)
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("Cannot open video for retry: %s", video_path)
            continue

        try:
            for label in video_labels:
                meta = metadata[label]
                fps = meta["fps"]
                offset = max(1, round(fps))

                for suffix, frame_key in [("A", "frame_a"), ("B", "frame_b")]:
                    if suffix not in flagged_suffixes[label]:
                        continue  # This frame was not flagged, skip

                    frame_info = meta[frame_key]
                    idx = frame_info["index"]
                    scene_start = frame_info["scene_start"]
                    scene_end = frame_info["scene_end"]

                    # Try forward, then backward
                    for direction in (offset, -offset):
                        alt_idx = idx + direction
                        if alt_idx < scene_start or alt_idx >= scene_end:
                            continue

                        cap.set(cv2.CAP_PROP_POS_FRAMES, alt_idx)
                        ret, alt_frame = cap.read()
                        if not ret:
                            continue

                        # Save to temporary path (not overwriting original)
                        temp_path = directory / f"{label}_{suffix}_retry.jpg"
                        cv2.imwrite(
                            str(temp_path), alt_frame,
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                        )
                        replaced.setdefault(label, set()).add(suffix)
                        break  # got a replacement, stop trying directions
        finally:
            cap.release()

    replaced_labels = set(replaced.keys())
    if not replaced_labels:
        logger.info("No alternative frames found — all retry pairs will be removed")
        return labels_to_retry

    # Step 3: Re-classify pairs using temp files where available, originals otherwise
    logger.info("Re-classifying %d pairs with replacement frames...", len(replaced_labels))

    all_labels: list[str] = []
    all_paths: list[Path] = []
    for label in sorted(replaced_labels):
        for suffix in ("A", "B"):
            # Use temp file if this suffix was replaced, otherwise use original
            if label in replaced and suffix in replaced[label]:
                path = directory / f"{label}_{suffix}_retry.jpg"
            else:
                path = directory / f"{label}_{suffix}.jpg"
            if path.exists():
                all_labels.append(label)
                all_paths.append(path)

    nsfw_labels: set[str] = set()
    for i in range(0, len(all_paths), batch_size):
        batch_paths = all_paths[i : i + batch_size]
        batch_labels = all_labels[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]

        results = classifier(images)
        for lbl, result in zip(batch_labels, results):
            top = max(result, key=lambda x: x["score"])
            if top["label"] == "nsfw" and top["score"] >= confidence:
                nsfw_labels.add(lbl)

    # Determine which replaced pairs now pass
    if keep_nsfw:
        # Want NSFW: pairs that are now NSFW pass
        now_pass = nsfw_labels & replaced_labels
    else:
        # Want SFW: pairs that are now SFW pass
        now_pass = replaced_labels - nsfw_labels

    # Step 4: For passing pairs, promote temp files to final; delete stale control images
    for label in now_pass:
        if label not in replaced:
            continue
        for suffix in replaced[label]:
            temp_path = directory / f"{label}_{suffix}_retry.jpg"
            final_path = directory / f"{label}_{suffix}.jpg"
            if temp_path.exists():
                temp_path.rename(final_path)
        # Delete stale _C and _image_base since they correspond to old frames
        for stale_suffix in ("_C.jpg", "_image_base.jpg"):
            stale_path = directory / f"{label}{stale_suffix}"
            stale_path.unlink(missing_ok=True)

    # Step 5: For pairs that still fail, delete temp files (originals are untouched)
    now_pass = now_pass | already_pass
    still_fail = retryable - now_pass
    for label in still_fail:
        if label not in replaced:
            continue
        for suffix in replaced[label]:
            temp_path = directory / f"{label}_{suffix}_retry.jpg"
            temp_path.unlink(missing_ok=True)

    logger.info("NSFW retry: %d saved, %d still removed", len(now_pass), len(still_fail))

    return still_fail | no_metadata


# ---------------------------------------------------------------------------
# Character detection (YOLO: person + anime face)
# ---------------------------------------------------------------------------

YOLO_COCO_MODEL = "yolov8n.pt"
YOLO_ANIME_MODEL = "https://huggingface.co/Anzhc/Anzhcs_YOLOs/resolve/main/Anzhc%20Face%20seg%20640%20v4%20y11n.pt"

_yolo_models: tuple | None = None
_yolo_load_lock = threading.Lock()
_yolo_inference_lock = threading.Lock()


def _get_yolo_models(device: str | None = None, anime_model_id: str | None = None):
    """Load YOLO models (singleton, thread-safe). Returns (coco_model, anime_model)."""
    global _yolo_models

    if _yolo_models is not None:
        return _yolo_models

    with _yolo_load_lock:
        if _yolo_models is not None:
            return _yolo_models

        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError(
                "Character detection requires ultralytics. "
                "Install with: pip install ultralytics"
            )

        anime_id = anime_model_id or YOLO_ANIME_MODEL

        logger.info("Loading YOLO COCO model: %s", YOLO_COCO_MODEL)
        coco_model = YOLO(YOLO_COCO_MODEL)
        if device:
            coco_model.to(device)

        logger.info("Loading YOLO anime face model: %s", anime_id)
        anime_model = YOLO(anime_id)
        if device:
            anime_model.to(device)

        _yolo_models = (coco_model, anime_model)
        return _yolo_models


def find_no_character_labels(
    directory: Path,
    labels_to_check: set[str],
    confidence: float = 0.5,
    batch_size: int = 32,
    device: str | None = None,
    anime_model_id: str | None = None,
) -> set[str]:
    """Find labels where _A has no detectable character/person.

    Uses two YOLO models simultaneously:
    1. Standard COCO (detects "person", class 0)
    2. Anime face model (detects anime/cartoon faces)

    If EITHER model detects a character, the image passes.
    Returns labels to REMOVE (no character found in _A).
    """
    if not labels_to_check:
        return set()

    coco_model, anime_model = _get_yolo_models(device, anime_model_id)

    # Collect _A paths for labels to check
    all_labels: list[str] = []
    all_paths: list[str] = []
    for label in sorted(labels_to_check):
        path = directory / f"{label}_A.jpg"
        if path.exists():
            all_labels.append(label)
            all_paths.append(str(path))

    if not all_paths:
        return set()

    logger.info("Character detection on %d _A images...", len(all_paths))

    # Track which labels have a character detected
    has_character: set[str] = set()

    for i in tqdm(range(0, len(all_paths), batch_size), desc="Character detection"):
        batch_paths = all_paths[i : i + batch_size]
        batch_labels = all_labels[i : i + batch_size]

        with _yolo_inference_lock:
            # COCO model: check for "person" (class 0)
            coco_results = coco_model.predict(
                batch_paths, conf=confidence, verbose=False, classes=[0],
            )
            # Anime face model: check for any detection
            anime_results = anime_model.predict(
                batch_paths, conf=confidence, verbose=False,
            )

        for label, coco_r, anime_r in zip(batch_labels, coco_results, anime_results):
            coco_has = len(coco_r.boxes) > 0 if coco_r.boxes is not None else False
            anime_has = len(anime_r.boxes) > 0 if anime_r.boxes is not None else False
            if coco_has or anime_has:
                has_character.add(label)

    no_character = labels_to_check - has_character
    logger.info("Character detection: %d have characters, %d do not",
                len(has_character), len(no_character))
    return no_character


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
    blur: bool = False,
    blur_threshold: float = 100.0,
    blur_max_retries: int = 3,
    character: bool = False,
    character_percentage: float = 100.0,
    character_confidence: float = 0.5,
    character_batch_size: int = 32,
    character_device: str | None = None,
    character_anime_model: str | None = None,
    character_seed: int | None = None,
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
    2. Remove blurry _A frames (with retry)
    3. Remove duplicate pairs (cosine similarity)
    4. Remove pairs without characters in _A (YOLO)
    5. NSFW filter (optional)
    6. Remove orphaned pairs (missing A or B)

    Returns dict with counts for each step.
    """
    stats = {
        "solid_removed": 0,
        "blur_removed": 0,
        "duplicates_removed": 0,
        "character_removed": 0,
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

    # Step 2: blur on _A (with retry)
    if blur:
        blur_labels = find_blur_labels(directory, blur_threshold, workers)
        new_blur = blur_labels - all_to_remove

        if new_blur:
            still_blurry = retry_blur_pairs(
                directory, new_blur,
                threshold=blur_threshold,
                max_retries=blur_max_retries,
            )
        else:
            still_blurry = set()

        stats["blur_removed"] = len(still_blurry)
        all_to_remove.update(still_blurry)
        logger.info("Blurry pairs to remove: %d (after retry)", len(still_blurry))

    # Step 3: duplicates
    if remove_dups:
        dups = find_duplicate_labels(directory, similarity=similarity, workers=workers)
        new_dups = dups - all_to_remove
        stats["duplicates_removed"] = len(new_dups)
        all_to_remove.update(dups)
        logger.info("Duplicate pairs to remove: %d", len(new_dups))

    # Step 4: character detection on _A (YOLO)
    if character:
        import random

        pairs = scan_pairs(directory)
        candidates = {l for l, f in pairs.items() if "A" in f} - all_to_remove

        if character_seed is not None:
            random.seed(character_seed)

        if character_percentage < 100.0:
            sample_size = max(1, int(len(candidates) * character_percentage / 100.0))
            labels_to_check = set(random.sample(sorted(candidates), sample_size))
        else:
            labels_to_check = candidates

        no_char = find_no_character_labels(
            directory, labels_to_check,
            confidence=character_confidence,
            batch_size=character_batch_size,
            device=character_device,
            anime_model_id=character_anime_model,
        )
        stats["character_removed"] = len(no_char)
        all_to_remove.update(no_char)
        logger.info("No-character pairs to remove: %d", len(no_char))

    # Step 5: NSFW filter (with retry)
    if nsfw:
        nsfw_labels = find_nsfw_labels(
            directory, keep_nsfw, nsfw_batch_size, nsfw_device, nsfw_confidence
        )
        new_nsfw = nsfw_labels - all_to_remove

        if new_nsfw:
            still_remove = retry_nsfw_pairs(
                directory, new_nsfw,
                keep_nsfw=keep_nsfw,
                batch_size=nsfw_batch_size,
                device=nsfw_device,
                confidence=nsfw_confidence,
            )
        else:
            still_remove = set()

        stats["nsfw_removed"] = len(still_remove)
        all_to_remove.update(still_remove)
        logger.info("NSFW filtered pairs to remove: %d (after retry)", len(still_remove))

    # Delete marked files
    if not dry_run and all_to_remove:
        pairs = scan_pairs(directory)
        for label in all_to_remove:
            if label in pairs:
                for path in pairs[label].values():
                    path.unlink(missing_ok=True)

    # Step 6: orphan cleanup
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
        + stats["blur_removed"]
        + stats["duplicates_removed"]
        + stats["character_removed"]
        + stats["nsfw_removed"]
        + stats["orphans_removed"]
    )

    # Count remaining
    remaining_pairs = scan_pairs(directory)
    stats["remaining"] = len(remaining_pairs)

    return stats
