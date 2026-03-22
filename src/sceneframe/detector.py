import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SceneBoundary:
    start_frame: int
    end_frame: int
    fps: float

    @property
    def duration_seconds(self) -> float:
        return (self.end_frame - self.start_frame) / self.fps


def detect_scenes(video_path: Path, show_progress: bool = True) -> list[SceneBoundary]:
    """Detect scene boundaries in a video using PySceneDetect ContentDetector.

    Returns a list of SceneBoundary. Returns empty list on failure.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        logger.warning("Video not found: %s", video_path)
        return []

    try:
        from scenedetect import detect, ContentDetector

        scene_list = detect(str(video_path), ContentDetector(), show_progress=show_progress)

        if not scene_list:
            return []

        fps = scene_list[0][0].framerate
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
