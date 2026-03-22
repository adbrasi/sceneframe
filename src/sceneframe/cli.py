import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import cv2
from tqdm import tqdm

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".webm", ".mov", ".wmv", ".flv"}

DEFAULT_MIN_VIDEO_DURATION = 10.0


def _get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using OpenCV. Returns 0.0 on failure."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps > 0 and frame_count > 0:
            return frame_count / fps
        return 0.0
    finally:
        cap.release()


def _find_videos(input_dir: Path, min_duration: float = 0.0) -> list[Path]:
    """Find all video files in a directory, optionally filtering by minimum duration."""
    candidates = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    )
    if min_duration <= 0:
        return candidates

    result = []
    for v in candidates:
        dur = _get_video_duration(v)
        if dur >= min_duration:
            result.append(v)
        else:
            logger.info("Skipping %s (%.1fs < %.1fs min)", v.name, dur, min_duration)
    return result


def _process_single_video(
    video_path: Path,
    output_dir: Path,
    mode: str,
) -> dict:
    """Process a single video: detect scenes, re-segment, extract pairs.

    Runs in a worker process.
    """
    from .detector import detect_scenes, re_detect_long_scenes
    from .extractor import (
        extract_intra_scene_pairs,
        extract_inter_scene_pairs_sequential,
        extract_inter_scene_pairs_sliding,
    )

    video_name = video_path.stem
    result = {"video": video_path.name, "pairs": 0, "scenes": 0, "error": None}

    try:
        scenes = detect_scenes(video_path, show_progress=False)
        if not scenes:
            return result

        scenes = re_detect_long_scenes(video_path, scenes, max_seconds=20.0)
        result["scenes"] = len(scenes)

        video_output = output_dir / video_name

        if mode == "intra":
            pairs = extract_intra_scene_pairs(video_path, scenes, video_output)
        elif mode == "inter-seq":
            pairs = extract_inter_scene_pairs_sequential(video_path, scenes, video_output)
        elif mode == "inter-slide":
            pairs = extract_inter_scene_pairs_sliding(video_path, scenes, video_output)
        else:
            p1 = extract_intra_scene_pairs(video_path, scenes, video_output / "intra")
            p2 = extract_inter_scene_pairs_sequential(video_path, scenes, video_output / "inter-seq")
            p3 = extract_inter_scene_pairs_sliding(video_path, scenes, video_output / "inter-slide")
            pairs = p1 + p2 + p3

        result["pairs"] = pairs

    except Exception as e:
        result["error"] = str(e)
        logger.error("Failed to process %s: %s", video_path.name, e)

    return result


def _print_result(result: dict):
    """Print processing result for a single video."""
    if result["error"]:
        click.echo(f"  ERROR {result['video']}: {result['error']}", err=True)
    elif result["pairs"] == 0:
        click.echo(f"  {result['video']}: no scenes detected")
    else:
        click.echo(f"  {result['video']}: {result['scenes']} scenes -> {result['pairs']} pairs")


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for frame pairs",
)
@click.option(
    "--mode", "-m",
    type=click.Choice(["intra", "inter-seq", "inter-slide", "all"], case_sensitive=False),
    default="all",
    show_default=True,
    help=(
        "Extraction mode: "
        "intra = first+last frame per scene, "
        "inter-seq = first frames of consecutive scene pairs (no overlap), "
        "inter-slide = first frames of consecutive scenes (sliding window), "
        "all = run all three modes"
    ),
)
@click.option(
    "--min-duration",
    type=float,
    default=DEFAULT_MIN_VIDEO_DURATION,
    show_default=True,
    help="Minimum video duration in seconds. Videos shorter than this are skipped.",
)
def main(input_path: Path, output: Path, mode: str, min_duration: float):
    """Extract frame pairs from video scenes for model training.

    INPUT_PATH can be a directory of videos or a single video file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    input_path = Path(input_path).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        dur = _get_video_duration(input_path)
        if dur < min_duration:
            click.echo(f"Video too short ({dur:.1f}s < {min_duration:.1f}s min).", err=True)
            raise SystemExit(1)
        videos = [input_path]
    else:
        videos = _find_videos(input_path, min_duration=min_duration)

    if not videos:
        click.echo("No video files found.", err=True)
        raise SystemExit(1)

    click.echo(f"Found {len(videos)} video(s). Mode: {mode}")

    workers = min(os.cpu_count() or 1, len(videos))

    if len(videos) == 1:
        result = _process_single_video(videos[0], output, mode)
        _print_result(result)
        return

    total_pairs = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_single_video, v, output, mode): v
            for v in videos
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            _print_result(result)
            total_pairs += result["pairs"]

    click.echo(f"\nDone! {total_pairs} total pairs from {len(videos)} videos.")


if __name__ == "__main__":
    main()
