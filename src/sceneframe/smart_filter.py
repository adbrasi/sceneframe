"""Smart filter: NSFW-first, YOLO-fallback cleaning pipeline.

Checks BOTH _A and _B images. A pair is KEPT if:
1. ANY image is classified as NSFW (pass 1 or retry), OR
2. ANY YOLO model detects a character in ANY image (pass 1 or retry)

Otherwise the pair is DELETED.

This replaces --nsfw and --character when --smart-filter is active.
"""

import gc
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class SmartFilterResult:
    nsfw_approved: int = 0
    nsfw_retry_approved: int = 0
    yolo_approved: int = 0
    yolo_retry_approved: int = 0
    deleted: int = 0
    no_metadata_deleted: int = 0

    @property
    def total_deleted(self) -> int:
        return self.deleted + self.no_metadata_deleted


def _cleanup_stale_temps(directory: Path):
    """Remove leftover _retry.jpg files from interrupted runs."""
    for stale in directory.glob("*_retry.jpg"):
        stale.unlink(missing_ok=True)


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
            continue
        metadata[entry["label"]] = entry
    return metadata


# ---------------------------------------------------------------------------
# NSFW classification helpers
# ---------------------------------------------------------------------------

def _nsfw_classify_batch(
    all_paths: list[Path],
    all_labels: list[str],
    classifier,
    batch_size: int,
) -> dict[str, bool]:
    """Classify paths as NSFW. Returns {label: True if ANY image is NSFW}."""
    from PIL import Image

    nsfw_labels: set[str] = set()

    def _load_batch(paths):
        return [Image.open(p).convert("RGB") for p in paths]

    prefetch_pool = ThreadPoolExecutor(max_workers=8)
    batches = list(range(0, len(all_paths), batch_size))
    next_future = None

    for bi, i in enumerate(tqdm(batches, desc="Smart filter: NSFW")):
        batch_paths = all_paths[i : i + batch_size]
        batch_labels = all_labels[i : i + batch_size]

        if next_future is not None:
            images = next_future.result()
        else:
            images = _load_batch(batch_paths)

        if bi + 1 < len(batches):
            next_i = batches[bi + 1]
            next_future = prefetch_pool.submit(_load_batch, all_paths[next_i : next_i + batch_size])
        else:
            next_future = None

        results = classifier(images)
        for label, result in zip(batch_labels, results):
            top = max(result, key=lambda x: x["score"])
            if top["label"] == "nsfw" and top["score"] >= 0.5:
                nsfw_labels.add(label)

    prefetch_pool.shutdown(wait=False)
    return nsfw_labels


# ---------------------------------------------------------------------------
# Frame retry helpers
# ---------------------------------------------------------------------------

JPEG_QUALITY = 95


def _extract_retry_frames(
    directory: Path,
    labels: set[str],
    metadata: dict[str, dict],
    max_retries: int,
    suffixes: list[str] | None = None,
) -> dict[str, dict[str, Path]]:
    """Extract alternative frames for given labels, grouped by video.

    Returns {label: {"A": temp_path, "B": temp_path}} for successfully extracted frames.
    Only extracts for suffixes specified (default: both A and B).
    """
    if suffixes is None:
        suffixes = ["A", "B"]

    suffix_to_key = {"A": "frame_a", "B": "frame_b"}

    by_video: dict[str, list[str]] = {}
    for label in labels:
        if label in metadata:
            by_video.setdefault(metadata[label]["video"], []).append(label)

    replaced: dict[str, dict[str, Path]] = {}

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
                fps = meta["fps"]
                offset_base = max(1, round(fps))

                for suffix in suffixes:
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
                            replaced.setdefault(label, {})[suffix] = temp_path
                            found = True
                            break

                        if found:
                            break
        finally:
            cap.release()

    return replaced


def _promote_temps(directory: Path, replaced: dict[str, dict[str, Path]], passing_labels: set[str]):
    """Promote temp files for passing labels, delete temps for failing ones."""
    for label, suffix_map in replaced.items():
        if label in passing_labels:
            for suffix, temp_path in suffix_map.items():
                final_path = directory / f"{label}_{suffix}.jpg"
                if temp_path.exists():
                    temp_path.rename(final_path)
            # Delete stale control images
            for stale in ("_C.jpg", "_image_base.jpg"):
                (directory / f"{label}{stale}").unlink(missing_ok=True)
        else:
            for suffix, temp_path in suffix_map.items():
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# YOLO classification helpers
# ---------------------------------------------------------------------------

def _yolo_classify_labels(
    directory: Path,
    labels: set[str],
    coco_model,
    anime_model,
    batch_size: int,
    confidence: float,
) -> set[str]:
    """Run YOLO on both _A and _B for given labels. Returns labels with character detected."""
    all_labels: list[str] = []
    all_paths: list[str] = []

    for label in sorted(labels):
        for suffix in ("A", "B"):
            path = directory / f"{label}_{suffix}.jpg"
            if path.exists():
                all_labels.append(label)
                all_paths.append(str(path))

    if not all_paths:
        return set()

    has_character: set[str] = set()

    for i in tqdm(range(0, len(all_paths), batch_size), desc="Smart filter: YOLO"):
        batch_paths = all_paths[i : i + batch_size]
        batch_labels = all_labels[i : i + batch_size]

        coco_results = coco_model.predict(batch_paths, conf=confidence, verbose=False, classes=[0])
        anime_results = anime_model.predict(batch_paths, conf=confidence, verbose=False)

        for label, coco_r, anime_r in zip(batch_labels, coco_results, anime_results):
            coco_has = len(coco_r.boxes) > 0 if coco_r.boxes is not None else False
            anime_has = len(anime_r.boxes) > 0 if anime_r.boxes is not None else False
            if coco_has or anime_has:
                has_character.add(label)

    return has_character


def _yolo_classify_temps(
    directory: Path,
    replaced: dict[str, dict[str, Path]],
    labels: set[str],
    coco_model,
    anime_model,
    batch_size: int,
    confidence: float,
) -> set[str]:
    """Run YOLO on temp/original files for retry labels. Returns labels with character detected."""
    all_labels: list[str] = []
    all_paths: list[str] = []

    for label in sorted(labels):
        for suffix in ("A", "B"):
            if label in replaced and suffix in replaced[label]:
                path = replaced[label][suffix]
            else:
                path = directory / f"{label}_{suffix}.jpg"
            if path.exists():
                all_labels.append(label)
                all_paths.append(str(path))

    if not all_paths:
        return set()

    has_character: set[str] = set()

    for i in range(0, len(all_paths), batch_size):
        batch_paths = all_paths[i : i + batch_size]
        batch_labels = all_labels[i : i + batch_size]

        coco_results = coco_model.predict(batch_paths, conf=confidence, verbose=False, classes=[0])
        anime_results = anime_model.predict(batch_paths, conf=confidence, verbose=False)

        for label, coco_r, anime_r in zip(batch_labels, coco_results, anime_results):
            coco_has = len(coco_r.boxes) > 0 if coco_r.boxes is not None else False
            anime_has = len(anime_r.boxes) > 0 if anime_r.boxes is not None else False
            if coco_has or anime_has:
                has_character.add(label)

    return has_character


# ---------------------------------------------------------------------------
# Main smart filter pipeline
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
    """Run the smart NSFW→YOLO filter pipeline.

    Pipeline:
    1. NSFW batch scan (both A and B) → approved pairs kept
    2. NSFW retry for failures → more pairs saved
    3. YOLO scan on remaining failures (both A and B, both models) → more kept
    4. YOLO retry → last chance
    5. Delete remaining failures
    """
    from .cleaner import scan_pairs

    _cleanup_stale_temps(directory)

    result = SmartFilterResult()
    pairs = scan_pairs(directory)
    skip = labels_to_skip or set()
    all_labels = {l for l in pairs if "A" in pairs[l] and "B" in pairs[l]} - skip

    if not all_labels:
        return result

    metadata = _load_metadata(directory)

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

    logger.info("Smart filter phase 1: NSFW scan on %d pairs", len(all_labels))

    classifier = hf_pipeline(
        "image-classification",
        model="Falconsai/nsfw_image_detection",
        device=nsfw_device,
        batch_size=nsfw_batch_size,
    )

    # Collect all A+B paths
    nsfw_scan_labels: list[str] = []
    nsfw_scan_paths: list[Path] = []
    for label in sorted(all_labels):
        for suffix in ("A", "B"):
            if suffix in pairs[label]:
                nsfw_scan_labels.append(label)
                nsfw_scan_paths.append(pairs[label][suffix])

    nsfw_approved = _nsfw_classify_batch(
        nsfw_scan_paths, nsfw_scan_labels, classifier, nsfw_batch_size,
    )
    result.nsfw_approved = len(nsfw_approved)
    nsfw_failed = all_labels - nsfw_approved
    logger.info("NSFW pass 1: %d approved, %d failed", result.nsfw_approved, len(nsfw_failed))

    # -----------------------------------------------------------------------
    # Phase 2: NSFW retry
    # -----------------------------------------------------------------------
    if nsfw_failed and not dry_run:
        retryable = {l for l in nsfw_failed if l in metadata}
        no_meta = nsfw_failed - retryable

        if retryable:
            logger.info("Smart filter phase 2: NSFW retry for %d pairs", len(retryable))

            # Identify which individual images are NOT nsfw (need replacement)
            from PIL import Image

            flagged: dict[str, set[str]] = {}
            step1_labels, step1_suffixes, step1_paths = [], [], []
            for label in sorted(retryable):
                for suffix in ("A", "B"):
                    path = directory / f"{label}_{suffix}.jpg"
                    if path.exists():
                        step1_labels.append(label)
                        step1_suffixes.append(suffix)
                        step1_paths.append(path)

            for i in range(0, len(step1_paths), nsfw_batch_size):
                bp = step1_paths[i : i + nsfw_batch_size]
                bl = step1_labels[i : i + nsfw_batch_size]
                bs = step1_suffixes[i : i + nsfw_batch_size]
                images = [Image.open(p).convert("RGB") for p in bp]
                results = classifier(images)
                for lbl, suf, res in zip(bl, bs, results):
                    top = max(res, key=lambda x: x["score"])
                    is_nsfw = top["label"] == "nsfw" and top["score"] >= nsfw_confidence
                    if not is_nsfw:
                        flagged.setdefault(lbl, set()).add(suf)

            # Extract retry frames only for flagged suffixes
            suffixes_per_label = {l: list(s) for l, s in flagged.items()}
            replaced = _extract_retry_frames(
                directory, set(flagged.keys()), metadata, nsfw_max_retries,
            )

            if replaced:
                # Re-classify replaced pairs
                re_labels, re_paths = [], []
                for label in sorted(replaced.keys()):
                    for suffix in ("A", "B"):
                        if label in replaced and suffix in replaced[label]:
                            path = replaced[label][suffix]
                        else:
                            path = directory / f"{label}_{suffix}.jpg"
                        if path.exists():
                            re_labels.append(label)
                            re_paths.append(path)

                retry_nsfw = _nsfw_classify_batch(
                    re_paths, re_labels, classifier, nsfw_batch_size,
                )
                _promote_temps(directory, replaced, retry_nsfw)
                result.nsfw_retry_approved = len(retry_nsfw)
                nsfw_failed = nsfw_failed - retry_nsfw
                logger.info("NSFW retry: %d saved", result.nsfw_retry_approved)

        nsfw_failed = nsfw_failed  # includes no_meta labels

    # Unload NSFW model to free VRAM for YOLO
    del classifier
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    # -----------------------------------------------------------------------
    # Phase 3: YOLO batch (only on NSFW failures)
    # -----------------------------------------------------------------------
    if nsfw_failed:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError(
                "Smart filter requires ultralytics for YOLO. "
                "Install with: pip install ultralytics"
            )

        from .cleaner import YOLO_COCO_MODEL, YOLO_ANIME_MODEL

        logger.info("Smart filter phase 3: YOLO scan on %d pairs", len(nsfw_failed))

        anime_id = yolo_anime_model or YOLO_ANIME_MODEL
        coco_model = YOLO(YOLO_COCO_MODEL)
        anime_model = YOLO(anime_id)
        if yolo_device:
            coco_model.to(yolo_device)
            anime_model.to(yolo_device)

        yolo_approved = _yolo_classify_labels(
            directory, nsfw_failed, coco_model, anime_model,
            yolo_batch_size, yolo_confidence,
        )
        result.yolo_approved = len(yolo_approved)
        yolo_failed = nsfw_failed - yolo_approved
        logger.info("YOLO pass 1: %d approved, %d failed", result.yolo_approved, len(yolo_failed))

        # -------------------------------------------------------------------
        # Phase 4: YOLO retry
        # -------------------------------------------------------------------
        if yolo_failed and not dry_run:
            retryable_yolo = {l for l in yolo_failed if l in metadata}
            if retryable_yolo:
                logger.info("Smart filter phase 4: YOLO retry for %d pairs", len(retryable_yolo))
                replaced_yolo = _extract_retry_frames(
                    directory, retryable_yolo, metadata, yolo_max_retries,
                )
                if replaced_yolo:
                    retry_yolo_approved = _yolo_classify_temps(
                        directory, replaced_yolo, set(replaced_yolo.keys()),
                        coco_model, anime_model, yolo_batch_size, yolo_confidence,
                    )
                    _promote_temps(directory, replaced_yolo, retry_yolo_approved)
                    result.yolo_retry_approved = len(retry_yolo_approved)
                    yolo_failed = yolo_failed - retry_yolo_approved
                    logger.info("YOLO retry: %d saved", result.yolo_retry_approved)

        # Clean up YOLO models
        del coco_model, anime_model
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    else:
        yolo_failed = set()

    # -----------------------------------------------------------------------
    # Phase 5: Delete remaining failures
    # -----------------------------------------------------------------------
    to_delete = yolo_failed
    result.deleted = len(to_delete)

    if not dry_run and to_delete:
        pairs = scan_pairs(directory)
        for label in to_delete:
            if label in pairs:
                for path in pairs[label].values():
                    path.unlink(missing_ok=True)
            # Also delete control images
            for stale in ("_C.jpg", "_image_base.jpg"):
                (directory / f"{label}{stale}").unlink(missing_ok=True)

    total_kept = result.nsfw_approved + result.nsfw_retry_approved + result.yolo_approved + result.yolo_retry_approved
    logger.info(
        "Smart filter done: %d kept (%d nsfw + %d nsfw-retry + %d yolo + %d yolo-retry), %d deleted",
        total_kept, result.nsfw_approved, result.nsfw_retry_approved,
        result.yolo_approved, result.yolo_retry_approved, result.deleted,
    )

    return result
