import logging
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ENGINES = ("pyscenedetect", "transnetv2")

# Default workers per engine (tuned for EPYC 48-96 core + RTX 5090/PRO 6000)
DEFAULT_WORKERS = {
    "pyscenedetect": None,  # cpu_count - 2 (set at runtime)
    "transnetv2": 8,        # 6-8 optimal: ffmpeg decode is I/O bound, GPU inference is fast
}

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


def _get_video_meta(video_path: Path) -> tuple[float, int]:
    """Get FPS and total frame count from video metadata (no frame decoding)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0, 0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return fps, total
    finally:
        cap.release()


def _decode_frames_ffmpeg(video_path: Path, w: int = 48, h: int = 27) -> np.ndarray | None:
    """Decode all frames via ffmpeg subprocess pipe, resized to w×h RGB.

    This is the same approach used internally by TransNetV2's predict_video().
    FFmpeg decode runs natively in C — orders of magnitude faster than
    Python frame-by-frame OpenCV. Each thread spawns its own ffmpeg process
    so this is fully thread-safe.

    Returns ndarray [N, h, w, 3] uint8 RGB, or None on failure.
    """
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"scale={w}:{h}",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        "-loglevel", "error",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=300)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("ffmpeg decode failed for %s: %s", video_path.name, e)
        return None

    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace").strip()
        logger.warning("ffmpeg error for %s: %s", video_path.name, stderr[:200])
        return None

    raw = proc.stdout
    if len(raw) == 0:
        return None

    frame_bytes = h * w * 3
    n_frames = len(raw) // frame_bytes
    if n_frames == 0:
        return None

    return np.frombuffer(raw, dtype=np.uint8).reshape(n_frames, h, w, 3)


def _detect_transnetv2(video_path: Path) -> list[SceneBoundary]:
    """Detect scenes using TransNetV2 on GPU with ffmpeg pipe decoding.

    Pipeline:
    1. ffmpeg subprocess decodes + scales to 48×27 RGB (thread-safe, parallel)
    2. predict_frames() runs GPU inference (serialized via lock)
    3. predictions_to_scenes() converts to scene boundaries (CPU, no lock)

    TransNetV2 already detects dissolves and gradual transitions, so
    re_detect_long_scenes is NOT needed when using this engine.
    """
    model = _get_transnet_model()

    # Get FPS from video metadata (no frame decoding)
    fps, total_frames = _get_video_meta(video_path)
    if fps <= 0 or total_frames <= 0:
        return []

    # Decode frames via ffmpeg subprocess (thread-safe, runs in parallel)
    frames = _decode_frames_ffmpeg(video_path)
    if frames is None:
        return []

    # GPU inference with lock (CUDA not thread-safe for concurrent kernels)
    with _inference_lock:
        single_pred, _ = model.predict_frames(frames)

    # predictions_to_scenes expects numpy; predict_frames may return tensor
    if hasattr(single_pred, "cpu"):
        single_pred = single_pred.cpu().numpy()

    scenes_array = model.predictions_to_scenes(single_pred)

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

    NOTE: This is only useful for PySceneDetect. TransNetV2 already detects
    dissolves and gradual transitions, making this step redundant.
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
