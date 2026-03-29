import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ENGINES = ("pyscenedetect", "transnetv2")

# Singleton TransNetV2 model — loaded once, reused across all videos
_transnet_model = None
_transnet_lock = threading.Lock()
_inference_lock = threading.Lock()


@dataclass
class SceneBoundary:
    start_frame: int
    end_frame: int
    fps: float

    @property
    def duration_seconds(self) -> float:
        return (self.end_frame - self.start_frame) / self.fps


# ---------------------------------------------------------------------------
# PySceneDetect engine (CPU)
# ---------------------------------------------------------------------------

def _detect_pyscenedetect(video_path: Path, show_progress: bool) -> list[SceneBoundary]:
    """Detect scenes using PySceneDetect ContentDetector (CPU)."""
    from scenedetect import detect, ContentDetector, open_video as sv_open_video

    scene_list = detect(str(video_path), ContentDetector(), show_progress=show_progress)

    video = sv_open_video(str(video_path))
    fps = video.frame_rate
    total_frames = video.duration.frame_num
    del video

    if not scene_list:
        if total_frames > 0:
            return [SceneBoundary(start_frame=0, end_frame=total_frames, fps=fps)]
        return []

    boundaries = []
    for start_tc, end_tc in scene_list:
        boundaries.append(
            SceneBoundary(
                start_frame=start_tc.frame_num,
                end_frame=end_tc.frame_num,
                fps=fps,
            )
        )
    return boundaries


# ---------------------------------------------------------------------------
# TransNetV2 engine (GPU)
# ---------------------------------------------------------------------------

def _get_transnet_model():
    """Get or create the singleton TransNetV2 model (thread-safe)."""
    global _transnet_model

    if _transnet_model is not None:
        return _transnet_model

    with _transnet_lock:
        if _transnet_model is not None:
            return _transnet_model

        from transnetv2_pytorch import TransNetV2
        _transnet_model = TransNetV2(device="auto")
        device = getattr(_transnet_model, "device", "unknown")
        logger.info("TransNetV2 loaded on device: %s", device)
        return _transnet_model


def _decode_frames_opencv(video_path: Path) -> tuple[np.ndarray, float, int]:
    """Decode all frames from video using OpenCV, resized to 48x27 RGB.

    Returns (frames_array, fps, total_frames).
    frames_array shape: [N, 27, 48, 3] uint8 RGB.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small = cv2.resize(rgb, (48, 27))
            frames.append(small)

        if not frames:
            return np.empty((0, 27, 48, 3), dtype=np.uint8), fps, 0

        return np.array(frames, dtype=np.uint8), fps, len(frames)
    finally:
        cap.release()


def _detect_transnetv2(video_path: Path) -> list[SceneBoundary]:
    """Detect scenes using TransNetV2 on GPU with OpenCV decoding."""
    import torch

    model = _get_transnet_model()

    # Decode on CPU (thread-safe, each thread has its own VideoCapture)
    frames, fps, total_frames = _decode_frames_opencv(video_path)
    if total_frames == 0 or fps <= 0:
        return []

    # GPU inference with lock (CUDA not thread-safe for concurrent writes)
    frames_tensor = torch.from_numpy(frames)
    with _inference_lock:
        frames_tensor = frames_tensor.to(model.device)
        with torch.no_grad():
            single_pred, _ = model.predict_frames(frames_tensor)

    scenes_array = model.predictions_to_scenes(single_pred.cpu().numpy())

    if len(scenes_array) == 0:
        return [SceneBoundary(start_frame=0, end_frame=total_frames, fps=fps)]

    boundaries = []
    for start, end in scenes_array:
        boundaries.append(SceneBoundary(
            start_frame=int(start),
            end_frame=int(end),
            fps=fps,
        ))
    return boundaries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_scenes(
    video_path: Path,
    engine: str = "pyscenedetect",
    show_progress: bool = True,
) -> list[SceneBoundary]:
    """Detect scene boundaries in a video.

    Parameters
    ----------
    engine : str
        "pyscenedetect" (CPU, default) or "transnetv2" (GPU).

    Returns a list of SceneBoundary. Returns empty list on failure.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        logger.warning("Video not found: %s", video_path)
        return []

    try:
        if engine == "transnetv2":
            return _detect_transnetv2(video_path)
        else:
            return _detect_pyscenedetect(video_path, show_progress)
    except Exception as e:
        logger.error("[%s] Scene detection failed for %s: %s", engine, video_path.name, e)
        return []


def re_detect_long_scenes(
    video_path: Path,
    scenes: list[SceneBoundary],
    max_seconds: float = 20.0,
) -> list[SceneBoundary]:
    """Re-detect scenes longer than max_seconds with a more sensitive detector.

    Uses AdaptiveDetector with a lower threshold on the long segments.
    Short scenes pass through unchanged.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return scenes

    result = []

    for scene in scenes:
        if scene.duration_seconds <= max_seconds:
            result.append(scene)
            continue

        try:
            from scenedetect import SceneManager, open_video, AdaptiveDetector

            video = open_video(str(video_path))
            try:
                video.seek(scene.start_frame)

                scene_manager = SceneManager()
                scene_manager.add_detector(
                    AdaptiveDetector(adaptive_threshold=1.5)
                )

                scene_manager.detect_scenes(video, end_time=scene.end_frame)
                sub_scenes = scene_manager.get_scene_list()
            finally:
                del video

            if len(sub_scenes) <= 1:
                result.append(scene)
                continue

            fps = scene.fps
            for start_tc, end_tc in sub_scenes:
                result.append(
                    SceneBoundary(
                        start_frame=start_tc.frame_num,
                        end_frame=end_tc.frame_num,
                        fps=fps,
                    )
                )

            logger.info(
                "Re-segmented long scene (%ds) into %d sub-scenes",
                int(scene.duration_seconds),
                len(sub_scenes),
            )

        except Exception as e:
            logger.warning(
                "Re-detection failed for scene at frame %d: %s",
                scene.start_frame, e,
            )
            result.append(scene)

    return result
