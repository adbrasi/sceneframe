"""Smart filter: NSFW-first, YOLO-fallback cleaning pipeline.

Each image (_A and _B) must individually pass at least one check:
- NSFW detected, OR
- YOLO detects a person/character

If EITHER image in a pair fails ALL checks (even after retries) → pair is DELETED.

Pipeline per image:
1. NSFW classification → pass if NSFW
2. NSFW retry (advance frames) → pass if new frame is NSFW
3. YOLO (person + anime face) → pass if character detected
4. YOLO retry (advance frames) → pass if character detected
5. Still nothing → image fails → pair deleted

This replaces --nsfw and --character when --smart-filter is active.
"""

import gc
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
from tqdm import tqdm

logger = logging.getLogger(__name__)

JPEG_QUALITY = 95


@dataclass
class SmartFilterResult:
    nsfw_approved: int = 0
    nsfw_retry_approved: int = 0
    yolo_approved: int = 0
    yolo_retry_approved: int = 0
    deleted: int = 0

    @property
    def total_deleted(self) -> int:
        return self.deleted


def _cleanup_stale_temps(directory: Path):
    for stale in directory.glob("*_retry.jpg"):
        stale.unlink(missing_ok=True)


def _load_metadata(directory: Path) -> dict[str, dict]:
    import json
    meta_path = directory / "pairs_metadata.jsonl"
    if not meta_path.exists():
        return {}
    metadata = {}
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        metadata[entry["label"]] = entry
    return metadata


# ---------------------------------------------------------------------------
# NSFW helpers (per-image tracking)
# ---------------------------------------------------------------------------

def _nsfw_classify_images(
    paths: list[Path],
    classifier,
    batch_size: int,
    confidence: float,
) -> set[int]:
    """Classify images. Returns set of indices that are NSFW."""
    from PIL import Image

    nsfw_indices: set[int] = set()

    def _load_batch(batch_paths):
        return [Image.open(p).convert("RGB") for p in batch_paths]

    prefetch_pool = ThreadPoolExecutor(max_workers=8)
    batches = list(range(0, len(paths), batch_size))
    next_future = None

    for bi, i in enumerate(tqdm(batches, desc="Smart filter: NSFW")):
        batch_paths = paths[i : i + batch_size]

        if next_future is not None:
            images = next_future.result()
        else:
            images = _load_batch(batch_paths)

        if bi + 1 < len(batches):
            next_i = batches[bi + 1]
            next_future = prefetch_pool.submit(_load_batch, paths[next_i : next_i + batch_size])
        else:
            next_future = None

        results = classifier(images)
        for j, result in enumerate(results):
            top = max(result, key=lambda x: x["score"])
            if top["label"] == "nsfw" and top["score"] >= confidence:
                nsfw_indices.add(i + j)

    prefetch_pool.shutdown(wait=False)
    return nsfw_indices


# ---------------------------------------------------------------------------
# Frame retry
# ---------------------------------------------------------------------------

def _extract_retry_frames_for_images(
    directory: Path,
    image_keys: list[tuple[str, str]],  # [(label, suffix), ...]
    metadata: dict[str, dict],
    max_retries: int,
) -> dict[tuple[str, str], Path]:
    """Extract alternative frames for specific (label, suffix) pairs.

    Returns {(label, suffix): temp_path} for successfully extracted frames.
    """
    suffix_to_key = {"A": "frame_a", "B": "frame_b"}

    # Group by video
    by_video: dict[str, list[tuple[str, str]]] = {}
    for label, suffix in image_keys:
        if label in metadata:
            by_video.setdefault(metadata[label]["video"], []).append((label, suffix))

    replaced: dict[tuple[str, str], Path] = {}

    for video_path_str, items in by_video.items():
        video_path = Path(video_path_str)
        if not video_path.exists():
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        try:
            for label, suffix in items:
                meta = metadata[label]
                fps = meta["fps"]
                offset_base = max(1, round(fps))
                frame_key = suffix_to_key.get(suffix)
                if not frame_key or frame_key not in meta:
                    continue

                frame_info = meta[frame_key]
                idx = frame_info["index"]
                scene_start = frame_info["scene_start"]
                scene_end = frame_info["scene_end"]

                for attempt in range(1, max_retries + 1):
                    offset = offset_base * attempt
                    found = False

                    for direction in (offset, -offset):
                        alt_idx = idx + direction
                        if alt_idx < scene_start or alt_idx >= scene_end:
                            continue

                        cap.set(cv2.CAP_PROP_POS_FRAMES, alt_idx)
                        ret, frame = cap.read()
                        if not ret:
                            continue

                        temp_path = directory / f"{label}_{suffix}_retry.jpg"
                        cv2.imwrite(
                            str(temp_path), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                        )
                        replaced[(label, suffix)] = temp_path
                        found = True
                        break

                    if found:
                        break
        finally:
            cap.release()

    return replaced


def _promote_image_temps(
    directory: Path,
    replaced: dict[tuple[str, str], Path],
    passing_keys: set[tuple[str, str]],
):
    """Promote temp files for passing images, delete temps for failing ones."""
    promoted_labels: set[str] = set()
    for key, temp_path in replaced.items():
        label, suffix = key
        if key in passing_keys:
            final_path = directory / f"{label}_{suffix}.jpg"
            if temp_path.exists():
                temp_path.rename(final_path)
                promoted_labels.add(label)
        else:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    # Delete stale control images for promoted labels
    for label in promoted_labels:
        for stale in ("_C.jpg", "_image_base.jpg"):
            (directory / f"{label}{stale}").unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# YOLO helpers (per-image tracking)
# ---------------------------------------------------------------------------

def _yolo_classify_images(
    paths: list[str],
    coco_model,
    anime_model,
    batch_size: int,
    confidence: float,
) -> set[int]:
    """Run YOLO on images. Returns set of indices with character detected."""
    has_character: set[int] = set()

    for i in tqdm(range(0, len(paths), batch_size), desc="Smart filter: YOLO"):
        batch_paths = paths[i : i + batch_size]

        coco_results = coco_model.predict(batch_paths, conf=confidence, verbose=False, classes=[0])
        anime_results = anime_model.predict(batch_paths, conf=confidence, verbose=False)

        for j, (coco_r, anime_r) in enumerate(zip(coco_results, anime_results)):
            coco_has = len(coco_r.boxes) > 0 if coco_r.boxes is not None else False
            anime_has = len(anime_r.boxes) > 0 if anime_r.boxes is not None else False
            if coco_has or anime_has:
                has_character.add(i + j)

    return has_character


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def smart_filter_directory(
    directory: Path,
    nsfw_batch_size: int = 64,
    nsfw_device: str | None = None,
    nsfw_confidence: float = 0.5,
    nsfw_max_retries: int = 3,
    yolo_batch_size: int = 32,
    yolo_device: str | None = None,
    yolo_confidence: float = 0.5,
    yolo_max_retries: int = 3,
    yolo_anime_model: str | None = None,
    labels_to_skip: set[str] | None = None,
    dry_run: bool = False,
) -> SmartFilterResult:
    """Run the smart filter. Each image must pass NSFW or YOLO individually."""
    from .cleaner import scan_pairs

    _cleanup_stale_temps(directory)

    result = SmartFilterResult()
    pairs = scan_pairs(directory)
    skip = labels_to_skip or set()
    all_labels = {l for l in pairs if "A" in pairs[l] and "B" in pairs[l]} - skip

    if not all_labels:
        return result

    metadata = _load_metadata(directory)

    # Build image list: each image tracked individually
    # image_keys[i] = (label, suffix), image_paths[i] = Path
    image_keys: list[tuple[str, str]] = []
    image_paths: list[Path] = []
    for label in sorted(all_labels):
        for suffix in ("A", "B"):
            if suffix in pairs[label]:
                image_keys.append((label, suffix))
                image_paths.append(pairs[label][suffix])

    # Track which images have passed (by index)
    passed: set[int] = set()

    # -----------------------------------------------------------------------
    # Phase 1: NSFW batch
    # -----------------------------------------------------------------------
    try:
        import torch
        from transformers import pipeline as hf_pipeline
    except ImportError:
        raise RuntimeError(
            "Smart filter requires torch and transformers. "
            "Install with: pip install torch transformers Pillow"
        )

    if nsfw_device is None:
        nsfw_device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Smart filter phase 1: NSFW scan on %d images (%d pairs)",
                len(image_paths), len(all_labels))

    classifier = hf_pipeline(
        "image-classification",
        model="Falconsai/nsfw_image_detection",
        device=nsfw_device,
        batch_size=nsfw_batch_size,
    )

    nsfw_passed = _nsfw_classify_images(image_paths, classifier, nsfw_batch_size, nsfw_confidence)
    passed.update(nsfw_passed)
    result.nsfw_approved = len(nsfw_passed)
    logger.info("NSFW pass 1: %d/%d images passed", len(nsfw_passed), len(image_paths))

    # -----------------------------------------------------------------------
    # Phase 2: NSFW retry for images that failed
    # -----------------------------------------------------------------------
    failed_indices = set(range(len(image_keys))) - passed
    failed_image_keys = [(image_keys[i], i) for i in failed_indices]

    if failed_image_keys and not dry_run and metadata:
        # Only retry images with metadata
        retryable = [(key, idx) for key, idx in failed_image_keys if key[0] in metadata]

        if retryable:
            logger.info("Smart filter phase 2: NSFW retry for %d images", len(retryable))
            keys_to_retry = [key for key, _ in retryable]
            idx_map = {key: idx for key, idx in retryable}

            replaced = _extract_retry_frames_for_images(
                directory, keys_to_retry, metadata, nsfw_max_retries,
            )

            if replaced:
                # Re-classify the temp files
                retry_paths = [replaced[key] for key in replaced]
                retry_keys_ordered = list(replaced.keys())

                from PIL import Image
                retry_nsfw = set()
                for i in range(0, len(retry_paths), nsfw_batch_size):
                    bp = retry_paths[i : i + nsfw_batch_size]
                    images = [Image.open(p).convert("RGB") for p in bp]
                    results = classifier(images)
                    for j, res in enumerate(results):
                        top = max(res, key=lambda x: x["score"])
                        if top["label"] == "nsfw" and top["score"] >= nsfw_confidence:
                            retry_nsfw.add(i + j)

                passing_keys: set[tuple[str, str]] = set()
                for ri in retry_nsfw:
                    key = retry_keys_ordered[ri]
                    passing_keys.add(key)
                    original_idx = idx_map[key]
                    passed.add(original_idx)

                _promote_image_temps(directory, replaced, passing_keys)
                result.nsfw_retry_approved = len(passing_keys)
                logger.info("NSFW retry: %d images saved", len(passing_keys))

    # Unload NSFW model
    del classifier
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    # -----------------------------------------------------------------------
    # Phase 3: YOLO for still-failed images
    # -----------------------------------------------------------------------
    still_failed_indices = set(range(len(image_keys))) - passed

    if still_failed_indices:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError(
                "Smart filter requires ultralytics for YOLO. "
                "Install with: pip install ultralytics"
            )

        from .cleaner import YOLO_COCO_MODEL, YOLO_ANIME_MODEL

        logger.info("Smart filter phase 3: YOLO scan on %d images", len(still_failed_indices))

        anime_id = yolo_anime_model or YOLO_ANIME_MODEL
        coco_model = YOLO(YOLO_COCO_MODEL)
        anime_model = YOLO(anime_id)
        if yolo_device:
            coco_model.to(yolo_device)
            anime_model.to(yolo_device)

        # Build paths for failed images only
        yolo_indices = sorted(still_failed_indices)
        yolo_paths = []
        for idx in yolo_indices:
            label, suffix = image_keys[idx]
            # Use current file (may have been updated by NSFW retry)
            path = directory / f"{label}_{suffix}.jpg"
            yolo_paths.append(str(path))

        yolo_passed_local = _yolo_classify_images(
            yolo_paths, coco_model, anime_model, yolo_batch_size, yolo_confidence,
        )
        # Map local indices back to global
        for local_idx in yolo_passed_local:
            passed.add(yolo_indices[local_idx])

        result.yolo_approved = len(yolo_passed_local)
        logger.info("YOLO pass 1: %d/%d images passed", len(yolo_passed_local), len(yolo_indices))

        # -------------------------------------------------------------------
        # Phase 4: YOLO retry
        # -------------------------------------------------------------------
        still_failed_indices = set(range(len(image_keys))) - passed
        yolo_failed_keys = [(image_keys[i], i) for i in still_failed_indices]

        if yolo_failed_keys and not dry_run and metadata:
            retryable_yolo = [(key, idx) for key, idx in yolo_failed_keys if key[0] in metadata]

            if retryable_yolo:
                logger.info("Smart filter phase 4: YOLO retry for %d images", len(retryable_yolo))
                keys_to_retry = [key for key, _ in retryable_yolo]
                idx_map_yolo = {key: idx for key, idx in retryable_yolo}

                replaced_yolo = _extract_retry_frames_for_images(
                    directory, keys_to_retry, metadata, yolo_max_retries,
                )

                if replaced_yolo:
                    retry_paths_y = [str(replaced_yolo[key]) for key in replaced_yolo]
                    retry_keys_y = list(replaced_yolo.keys())

                    yolo_retry_passed = _yolo_classify_images(
                        retry_paths_y, coco_model, anime_model, yolo_batch_size, yolo_confidence,
                    )

                    passing_keys_y: set[tuple[str, str]] = set()
                    for ri in yolo_retry_passed:
                        key = retry_keys_y[ri]
                        passing_keys_y.add(key)
                        passed.add(idx_map_yolo[key])

                    _promote_image_temps(directory, replaced_yolo, passing_keys_y)
                    result.yolo_retry_approved = len(passing_keys_y)
                    logger.info("YOLO retry: %d images saved", len(passing_keys_y))

        del coco_model, anime_model
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Phase 5: Determine pairs to delete
    # -----------------------------------------------------------------------
    # A pair is deleted if ANY of its images failed all checks
    failed_images = set(range(len(image_keys))) - passed
    labels_with_failed_image: set[str] = set()
    for idx in failed_images:
        label, suffix = image_keys[idx]
        labels_with_failed_image.add(label)

    to_delete = labels_with_failed_image
    result.deleted = len(to_delete)

    if not dry_run and to_delete:
        pairs = scan_pairs(directory)
        for label in to_delete:
            if label in pairs:
                for path in pairs[label].values():
                    path.unlink(missing_ok=True)
            for stale in ("_C.jpg", "_image_base.jpg"):
                (directory / f"{label}{stale}").unlink(missing_ok=True)

    total_images = len(image_keys)
    passed_count = len(passed)
    logger.info(
        "Smart filter done: %d/%d images passed, %d pairs deleted",
        passed_count, total_images, result.deleted,
    )

    return result
