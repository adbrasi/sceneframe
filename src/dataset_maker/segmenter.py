import json
import logging
import subprocess
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class SceneBoundary:
    start_time: float
    end_time: float
    duration: float


_transnet_model = None


def _get_transnet_model():
    """Cache TransNetV2 model per-process (avoids reloading TF graph per episode)."""
    global _transnet_model
    if _transnet_model is None:
        from transnetv2 import TransNetV2
        _transnet_model = TransNetV2()
    return _transnet_model


def _reset_transnet_model():
    """Reset cached model after errors (e.g., GPU OOM) so next call re-initializes."""
    global _transnet_model
    _transnet_model = None


def detect_with_transnetv2(video_path: Path, fps: float = 24.0) -> list[float]:
    """Detect shot boundaries using TransNetV2. Returns list of boundary timestamps."""
    try:
        model = _get_transnet_model()
        video_frames, single_frame_predictions, all_frame_predictions = (
            model.predict_video(str(video_path))
        )
        scenes = model.predictions_to_scenes(single_frame_predictions)

        # Free large numpy arrays immediately (~336 MB for 30-min episode)
        del video_frames, all_frame_predictions

        boundaries = []
        for start, end in scenes:
            boundaries.append(start / fps)
            boundaries.append(end / fps)

        return sorted(set(boundaries))
    except Exception as e:
        logger.warning("TransNetV2 failed for %s: %s", video_path, e)
        _reset_transnet_model()
        return []


def detect_with_pyscenedetect(
    video_path: Path, threshold: float | None = None
) -> list[float]:
    """Detect scene boundaries using PySceneDetect (fades/dissolves)."""
    try:
        from scenedetect import detect, AdaptiveDetector

        detector = AdaptiveDetector(adaptive_threshold=threshold) if threshold else AdaptiveDetector()
        scene_list = detect(str(video_path), detector, show_progress=True)
        boundaries = []
        for scene in scene_list:
            boundaries.append(scene[0].get_seconds())
            boundaries.append(scene[1].get_seconds())

        return sorted(set(boundaries))
    except Exception as e:
        logger.warning("PySceneDetect failed for %s: %s", video_path, e)
        return []


def merge_boundaries(
    transnet_boundaries: list[float],
    pyscene_boundaries: list[float],
    tolerance: float = 0.5,
) -> list[float]:
    """Merge boundary lists, deduplicating within tolerance seconds."""
    all_boundaries = sorted(set(transnet_boundaries + pyscene_boundaries))
    if not all_boundaries:
        return []

    merged = [all_boundaries[0]]
    for b in all_boundaries[1:]:
        if b - merged[-1] > tolerance:
            merged.append(b)

    return merged


def boundaries_to_scenes(
    boundaries: list[float],
    video_duration: float,
    min_duration: float,
    max_duration: float,
) -> list[SceneBoundary]:
    """Convert boundary timestamps to scene intervals, filtering by duration."""
    if not boundaries:
        return [
            SceneBoundary(
                start_time=0, end_time=video_duration, duration=video_duration
            )
        ]

    scenes = []
    points = [0.0] + boundaries + [video_duration]

    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        duration = end - start
        if min_duration <= duration <= max_duration:
            scenes.append(
                SceneBoundary(start_time=start, end_time=end, duration=duration)
            )

    return scenes


def _make_clip_name() -> str:
    """Generate a random unique clip filename."""
    return f"{uuid.uuid4().hex[:12]}.mp4"


def split_video(
    video_path: Path, scenes: list[SceneBoundary], output_dir: Path
) -> list[tuple[Path, SceneBoundary]]:
    """Split video into clips using ffmpeg stream copy with random filenames."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []

    for scene in scenes:
        output_file = output_dir / _make_clip_name()
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{scene.start_time:.3f}",
            "-i", str(video_path),
            "-t", f"{scene.duration:.3f}",
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(output_file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            output_files.append((output_file, scene))
        else:
            # Clean up partial file on failure
            output_file.unlink(missing_ok=True)
            logger.warning(
                "ffmpeg failed for scene at %.1fs of %s: %s",
                scene.start_time, video_path.name, result.stderr[:200],
            )

    return output_files


def _get_video_info(video_path: Path) -> tuple[float, float]:
    """Get video FPS and duration in a single ffprobe call. Returns (fps, duration)."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    fps, duration = 24.0, 0.0
    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    r_frame_rate = stream.get("r_frame_rate", "24/1")
                    num, den = map(int, r_frame_rate.split("/"))
                    fps = num / den if den else 24.0
                    break
            duration = float(data.get("format", {}).get("duration", 0))
        except (json.JSONDecodeError, ValueError, ZeroDivisionError):
            logger.warning("Failed to parse video info for %s", video_path)
    else:
        logger.error(
            "ffprobe failed for %s (is ffmpeg installed?): %s",
            video_path, result.stderr[:200],
        )
    return fps, duration


def segment_episode(
    video_path: Path,
    output_dir: Path,
    min_duration: float,
    max_duration: float,
    detector: str = "both",
) -> list[dict]:
    """Segment a single episode into scenes. Returns list of clip metadata."""
    logger.info("Segmenting: %s (detector=%s)", video_path.name, detector)

    fps, duration = _get_video_info(video_path)
    if duration == 0:
        logger.error("Could not get duration for %s", video_path)
        return []

    if detector == "transnet":
        boundaries = detect_with_transnetv2(video_path, fps=fps)
    elif detector == "pyscene":
        boundaries = detect_with_pyscenedetect(video_path)
    else:
        transnet_bounds = detect_with_transnetv2(video_path, fps=fps)
        pyscene_bounds = detect_with_pyscenedetect(video_path)
        boundaries = merge_boundaries(transnet_bounds, pyscene_bounds)

    scenes = boundaries_to_scenes(boundaries, duration, min_duration, max_duration)

    if not scenes:
        logger.warning("No valid scenes found for %s", video_path.name)
        return []

    clip_pairs = split_video(video_path, scenes, output_dir)

    results = []
    for clip_path, scene in clip_pairs:
        results.append(
            {
                "file": clip_path.name,
                "source": video_path.name,
                "start_time": round(scene.start_time, 3),
                "end_time": round(scene.end_time, 3),
                "duration": round(scene.duration, 3),
            }
        )

    logger.info("Segmented %s into %d clips", video_path.name, len(results))
    return results


def segment_episodes(
    input_dir: Path,
    output_dir: Path,
    min_duration: float,
    max_duration: float,
    workers: int,
    episodes: list[Path] | None = None,
    detector: str = "both",
) -> list[dict]:
    """Segment episodes into scenes. Accepts explicit file list or scans directory."""
    output_dir = Path(output_dir).resolve()

    if episodes is None:
        video_extensions = {".mp4", ".mkv", ".avi", ".webm", ".mov"}
        input_dir = Path(input_dir).resolve()
        episodes = sorted(
            f for f in input_dir.iterdir() if f.suffix.lower() in video_extensions
        )

    if not episodes:
        logger.error("No video files found in %s", input_dir)
        return []

    logger.info("Found %d episodes to segment (detector=%s)", len(episodes), detector)

    # Skip ProcessPool overhead for single episode
    if len(episodes) == 1:
        return segment_episode(episodes[0], output_dir, min_duration, max_duration, detector)

    all_metadata = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                segment_episode, ep, output_dir, min_duration, max_duration, detector
            ): ep
            for ep in episodes
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Segmenting"
        ):
            try:
                metadata = future.result()
                all_metadata.extend(metadata)
            except Exception as e:
                ep = futures[future]
                logger.error("Failed to segment %s: %s", ep.name, e)

    return all_metadata


def re_segment_long_clips(
    clips_dir: Path,
    metadata: list[dict],
    threshold: float = 8.0,
    min_duration: float = 2.0,
) -> list[dict]:
    """Second pass: re-segment clips longer than threshold with more sensitive detection.

    Finds clips > threshold seconds, runs PySceneDetect with a lower (more sensitive)
    adaptive threshold to catch subtle scene changes, splits them further,
    and replaces the originals in the metadata.
    Only deletes the original clip if ALL sub-clips were produced successfully.
    """
    clips_dir = Path(clips_dir).resolve()
    long_clips = [m for m in metadata if m.get("duration", 0) > threshold]

    if not long_clips:
        logger.info("No clips longer than %.1fs found, skipping second pass", threshold)
        return metadata

    logger.info(
        "Second pass: re-segmenting %d clips longer than %.1fs",
        len(long_clips), threshold,
    )

    new_metadata = [m for m in metadata if m.get("duration", 0) <= threshold]

    for entry in tqdm(long_clips, desc="Re-segmenting"):
        clip_path = clips_dir / entry["file"]
        if not clip_path.exists():
            logger.warning("Clip not found for re-segmentation: %s", clip_path)
            new_metadata.append(entry)
            continue

        clip_duration = entry.get("duration", 0.0)
        if clip_duration == 0:
            new_metadata.append(entry)
            continue

        # Use more sensitive detection (lower threshold = more cuts detected)
        pyscene_bounds = detect_with_pyscenedetect(clip_path, threshold=1.5)

        if not pyscene_bounds:
            new_metadata.append(entry)
            continue

        # Use a generous max_duration so we don't silently drop large sub-clips
        scenes = boundaries_to_scenes(
            pyscene_bounds, clip_duration, min_duration, max_duration=threshold * 4
        )

        if len(scenes) <= 1:
            new_metadata.append(entry)
            continue

        sub_pairs = split_video(clip_path, scenes, clips_dir)

        if not sub_pairs:
            new_metadata.append(entry)
            continue

        original_start = entry.get("start_time", 0.0)

        for sub_path, scene in sub_pairs:
            new_metadata.append(
                {
                    "file": sub_path.name,
                    "source": entry.get("source", "unknown"),
                    "start_time": round(original_start + scene.start_time, 3),
                    "end_time": round(original_start + scene.end_time, 3),
                    "duration": round(scene.duration, 3),
                    "parent_clip": entry["file"],
                }
            )

        # Only delete original if we actually produced sub-clips
        clip_path.unlink(missing_ok=True)
        logger.info(
            "Re-segmented %s into %d sub-clips", entry["file"], len(sub_pairs)
        )

    logger.info(
        "Second pass complete: %d -> %d clips", len(metadata), len(new_metadata)
    )
    return new_metadata
