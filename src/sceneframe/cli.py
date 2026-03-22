import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import cv2
from tqdm import tqdm

from .detector import SceneBoundary

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


def _detect_scenes_for_video(video_path: Path) -> tuple[Path, list[SceneBoundary]]:
    """Detect and re-segment scenes for a single video. Runs in worker process."""
    from .detector import detect_scenes, re_detect_long_scenes

    scenes = detect_scenes(video_path, show_progress=False)
    if scenes:
        scenes = re_detect_long_scenes(video_path, scenes, max_seconds=20.0)
    return video_path, scenes


def _extract_pairs(
    video_scenes: list[tuple[Path, list[SceneBoundary]]],
    output_dir: Path,
    mode: str,
) -> int:
    """Extract frame pairs from all videos sequentially with global numbering.

    Output is flat: output_dir/0001_A.jpg, 0001_B.jpg, 0002_A.jpg, ...
    For mode "all", creates subdirs: output_dir/intra/, output_dir/inter-seq/, output_dir/inter-slide/
    """
    from .extractor import (
        extract_intra_scene_pairs,
        extract_inter_scene_pairs_sequential,
        extract_inter_scene_pairs_sliding,
    )

    if mode == "all":
        intra_dir = output_dir / "intra"
        inter_seq_dir = output_dir / "inter-seq"
        inter_slide_dir = output_dir / "inter-slide"

        intra_count = 0
        seq_count = 0
        slide_count = 0

        for video_path, scenes in video_scenes:
            if not scenes:
                continue
            p1 = extract_intra_scene_pairs(video_path, scenes, intra_dir, start_index=intra_count)
            intra_count += p1
            p2 = extract_inter_scene_pairs_sequential(video_path, scenes, inter_seq_dir, start_index=seq_count)
            seq_count += p2
            p3 = extract_inter_scene_pairs_sliding(video_path, scenes, inter_slide_dir, start_index=slide_count)
            slide_count += p3

        return intra_count + seq_count + slide_count

    extract_fn = {
        "intra": extract_intra_scene_pairs,
        "inter-seq": extract_inter_scene_pairs_sequential,
        "inter-slide": extract_inter_scene_pairs_sliding,
    }[mode]

    global_count = 0
    for video_path, scenes in video_scenes:
        if not scenes:
            continue
        pairs = extract_fn(video_path, scenes, output_dir, start_index=global_count)
        global_count += pairs

    return global_count


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

    if input_path.is_file() and input_path.suffix.lower() == ".txt":
        # Read list of directories from text file (one path per line)
        lines = input_path.read_text(encoding="utf-8").splitlines()
        videos = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            folder = Path(line)
            if not folder.is_dir():
                click.echo(f"Skipping (not a directory): {line}", err=True)
                continue
            videos.extend(_find_videos(folder, min_duration=min_duration))
    elif input_path.is_file():
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

    # Phase 1: Detect scenes in parallel
    click.echo("Detecting scenes...")
    video_scenes: list[tuple[Path, list[SceneBoundary]]] = []
    workers = min(os.cpu_count() or 1, len(videos))

    if len(videos) == 1:
        result = _detect_scenes_for_video(videos[0])
        video_scenes.append(result)
        scenes_count = len(result[1])
        click.echo(f"  {result[0].name}: {scenes_count} scenes")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_detect_scenes_for_video, v): v
                for v in videos
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Detecting"):
                try:
                    video_path, scenes = future.result()
                    video_scenes.append((video_path, scenes))
                    click.echo(f"  {video_path.name}: {len(scenes)} scenes")
                except Exception as e:
                    v = futures[future]
                    click.echo(f"  ERROR {v.name}: {e}", err=True)

    total_scenes = sum(len(s) for _, s in video_scenes)
    if total_scenes == 0:
        click.echo("No scenes detected in any video.", err=True)
        raise SystemExit(1)

    # Phase 2: Extract pairs sequentially (global numbering)
    click.echo(f"\nExtracting pairs from {total_scenes} scenes...")
    total_pairs = _extract_pairs(video_scenes, output, mode)

    click.echo(f"\nDone! {total_pairs} pairs from {len(videos)} videos -> {output}")


if __name__ == "__main__":
    main()
