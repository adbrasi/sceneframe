import logging
from pathlib import Path

import cv2
import numpy as np

from .detector import SceneBoundary

logger = logging.getLogger(__name__)

FRAME_OFFSET = 3
JPEG_QUALITY = 95


def extract_frame(video_path: Path, frame_index: int) -> np.ndarray | None:
    """Extract a single frame from a video by index. Returns BGR numpy array or None."""
    video_path = Path(video_path)
    if not video_path.exists():
        return None

    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            return None
        return frame
    finally:
        cap.release()


def _save_frame(frame: np.ndarray, path: Path) -> bool:
    """Save a frame as JPEG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def _safe_frame_indices(
    scene: SceneBoundary, offset: int = FRAME_OFFSET
) -> tuple[int, int] | None:
    """Get safe start/end frame indices with offset applied.

    Returns None if the scene is too short for the offset.
    """
    total_frames = scene.end_frame - scene.start_frame
    # Need at least offset*2 + 1 frames so start and end don't overlap
    if total_frames <= offset * 2 + 1:
        return None
    # end_frame is exclusive (first frame of next scene), so last frame is end_frame - 1
    return scene.start_frame + offset, scene.end_frame - 1 - offset


def extract_intra_scene_pairs(
    video_path: Path,
    scenes: list[SceneBoundary],
    output_dir: Path,
) -> int:
    """Mode 1: Extract first and last frame of each scene as pairs.

    Saves as 0001_A.jpg / 0001_B.jpg, etc.
    Returns the number of pairs saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return 0

    pair_count = 0

    try:
        for scene in scenes:
            indices = _safe_frame_indices(scene)
            if indices is None:
                continue

            start_idx, end_idx = indices

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
            ret_a, frame_a = cap.read()
            if not ret_a:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, end_idx)
            ret_b, frame_b = cap.read()
            if not ret_b:
                continue

            pair_count += 1
            label = f"{pair_count:04d}"
            _save_frame(frame_a, output_dir / f"{label}_A.jpg")
            _save_frame(frame_b, output_dir / f"{label}_B.jpg")
    finally:
        cap.release()

    return pair_count


def extract_inter_scene_pairs_sequential(
    video_path: Path,
    scenes: list[SceneBoundary],
    output_dir: Path,
) -> int:
    """Mode 2a: Pair first frames of consecutive scenes (no overlap).

    Pairs: (scene1, scene2), (scene3, scene4), ...
    Odd last scene is discarded.
    Returns the number of pairs saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return 0

    pair_count = 0

    try:
        for i in range(0, len(scenes) - 1, 2):
            scene_a = scenes[i]
            scene_b = scenes[i + 1]

            idx_a = scene_a.start_frame + FRAME_OFFSET
            idx_b = scene_b.start_frame + FRAME_OFFSET

            # Guard against scenes too short for the offset
            if idx_a >= scene_a.end_frame or idx_b >= scene_b.end_frame:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx_a)
            ret_a, frame_a = cap.read()
            if not ret_a:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx_b)
            ret_b, frame_b = cap.read()
            if not ret_b:
                continue

            pair_count += 1
            label = f"{pair_count:04d}"
            _save_frame(frame_a, output_dir / f"{label}_A.jpg")
            _save_frame(frame_b, output_dir / f"{label}_B.jpg")
    finally:
        cap.release()

    return pair_count


def extract_inter_scene_pairs_sliding(
    video_path: Path,
    scenes: list[SceneBoundary],
    output_dir: Path,
) -> int:
    """Mode 2b: Pair first frames of consecutive scenes (sliding window).

    Pairs: (scene1, scene2), (scene2, scene3), (scene3, scene4), ...
    Returns the number of pairs saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(scenes) < 2:
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return 0

    pair_count = 0

    try:
        for i in range(len(scenes) - 1):
            scene_a = scenes[i]
            scene_b = scenes[i + 1]

            idx_a = scene_a.start_frame + FRAME_OFFSET
            idx_b = scene_b.start_frame + FRAME_OFFSET

            # Guard against scenes too short for the offset
            if idx_a >= scene_a.end_frame or idx_b >= scene_b.end_frame:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx_a)
            ret_a, frame_a = cap.read()
            if not ret_a:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx_b)
            ret_b, frame_b = cap.read()
            if not ret_b:
                continue

            pair_count += 1
            label = f"{pair_count:04d}"
            _save_frame(frame_a, output_dir / f"{label}_A.jpg")
            _save_frame(frame_b, output_dir / f"{label}_B.jpg")
    finally:
        cap.release()

    return pair_count
