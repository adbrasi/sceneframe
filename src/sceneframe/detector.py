import logging
from dataclasses import dataclass
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


@dataclass
class SceneBoundary:
    start_frame: int
    end_frame: int
    fps: float

    @property
    def duration_seconds(self) -> float:
        return (self.end_frame - self.start_frame) / self.fps


def _get_video_info(video_path: Path) -> tuple[float, int]:
    """Get fps and total frame count from a video."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return fps, total
    finally:
        cap.release()


def _detect_scenes_transnet(video_path: Path) -> list[SceneBoundary]:
    """Detect scenes using TransNetV2 on GPU. Fast (~500+ fps on RTX 5090)."""
    from transnetv2_pytorch import TransNetV2

    fps, total_frames = _get_video_info(video_path)
    if total_frames <= 0 or fps <= 0:
        return []

    model = TransNetV2(device="auto")
    scenes = model.detect_scenes(str(video_path))

    if not scenes:
        return [SceneBoundary(start_frame=0, end_frame=total_frames, fps=fps)]

    boundaries = []
    for scene in scenes:
        boundaries.append(SceneBoundary(
            start_frame=scene["start_frame"],
            end_frame=scene["end_frame"],
            fps=fps,
        ))
    return boundaries


def _detect_scenes_pyscenedetect(video_path: Path, show_progress: bool) -> list[SceneBoundary]:
    """Detect scenes using PySceneDetect ContentDetector (CPU fallback)."""
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


def detect_scenes(video_path: Path, show_progress: bool = True) -> list[SceneBoundary]:
    """Detect scene boundaries in a video.

    Uses TransNetV2 (GPU) if available, falls back to PySceneDetect (CPU).
    Returns a list of SceneBoundary. Returns empty list on failure.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        logger.warning("Video not found: %s", video_path)
        return []

    try:
        return _detect_scenes_transnet(video_path)
    except ImportError:
        logger.debug("TransNetV2 not available, using PySceneDetect (CPU)")
    except Exception as e:
        logger.warning("TransNetV2 failed for %s: %s — falling back to PySceneDetect", video_path.name, e)

    try:
        return _detect_scenes_pyscenedetect(video_path, show_progress)
    except Exception as e:
        logger.error("Scene detection failed for %s: %s", video_path, e)
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
