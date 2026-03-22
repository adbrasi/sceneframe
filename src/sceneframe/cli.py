import logging
import os
import re
import sys
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
import cv2
from tqdm import tqdm

from .detector import SceneBoundary

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".webm", ".mov", ".wmv", ".flv"}

DEFAULT_MIN_VIDEO_DURATION = 10.0

STOP_EVENT = threading.Event()
_SIGINT_COUNT = 0
_SIGINT_LOCK = threading.Lock()


def _handle_sigint(_signum, _frame):
    global _SIGINT_COUNT
    with _SIGINT_LOCK:
        _SIGINT_COUNT += 1
        count = _SIGINT_COUNT
    STOP_EVENT.set()
    if count == 1:
        print("\nCtrl+C: cancelling... (press again to force quit)", file=sys.stderr, flush=True)
    else:
        print("\nForce quit.", file=sys.stderr, flush=True)
        os._exit(130)

# Regex to detect Windows-style absolute paths (e.g. C:\, D:\)
_WIN_PATH_RE = re.compile(r"^[A-Za-z]:[\\\/]")


def _resolve_path_from_txt(line: str, txt_parent: Path) -> Path:
    """Resolve a path from a .txt file, handling Windows paths on WSL."""
    if _WIN_PATH_RE.match(line) and sys.platform != "win32":
        drive = line[0].lower()
        rest = line[2:].replace("\\", "/").lstrip("/")
        return Path(f"/mnt/{drive}/{rest}")

    folder = Path(line)
    if not folder.is_absolute():
        folder = txt_parent / folder
    return folder.resolve()


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


def _extract_for_video(
    video_path: Path,
    scenes: list[SceneBoundary],
    output_dir: Path,
    mode: str,
    counters: dict[str, int],
) -> int:
    """Extract frame pairs for a single video, updating global counters in-place."""
    from .extractor import (
        extract_intra_scene_pairs,
        extract_inter_scene_pairs_sequential,
        extract_inter_scene_pairs_sliding,
    )

    if not scenes:
        return 0

    total = 0

    if mode == "all":
        p1 = extract_intra_scene_pairs(video_path, scenes, output_dir / "intra", start_index=counters["intra"])
        counters["intra"] += p1
        p2 = extract_inter_scene_pairs_sequential(video_path, scenes, output_dir / "inter-seq", start_index=counters["inter-seq"])
        counters["inter-seq"] += p2
        p3 = extract_inter_scene_pairs_sliding(video_path, scenes, output_dir / "inter-slide", start_index=counters["inter-slide"])
        counters["inter-slide"] += p3
        total = p1 + p2 + p3
    else:
        extract_fn = {
            "intra": extract_intra_scene_pairs,
            "inter-seq": extract_inter_scene_pairs_sequential,
            "inter-slide": extract_inter_scene_pairs_sliding,
        }[mode]
        pairs = extract_fn(video_path, scenes, output_dir, start_index=counters["main"])
        counters["main"] += pairs
        total = pairs

    return total


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
@click.option(
    "--workers", "-w",
    type=int,
    default=None,
    help="Number of parallel workers for scene detection. Defaults to CPU count - 2.",
)
def main(input_path: Path, output: Path, mode: str, min_duration: float, workers: int | None):
    """Extract frame pairs from video scenes for model training.

    INPUT_PATH can be a directory of videos, a single video file, or a .txt
    file with one directory path per line (lines starting with # are ignored).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Install Ctrl+C handler
    STOP_EVENT.clear()
    global _SIGINT_COUNT
    with _SIGINT_LOCK:
        _SIGINT_COUNT = 0
    signal.signal(signal.SIGINT, _handle_sigint)

    input_path = Path(input_path).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    if input_path.is_file() and input_path.suffix.lower() == ".txt":
        lines = input_path.read_text(encoding="utf-8").splitlines()
        videos = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            folder = _resolve_path_from_txt(line, input_path.parent)
            if not folder.is_dir():
                click.echo(f"Skipping (not a directory): {line} -> {folder}", err=True)
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

    cpu_count = os.cpu_count() or 4
    max_workers = workers if workers else max(1, cpu_count - 2)
    max_workers = min(max_workers, len(videos))

    click.echo(f"Found {len(videos)} video(s). Mode: {mode}. Workers: {max_workers}")

    # Global counters for continuous numbering across videos
    counters = {"intra": 0, "inter-seq": 0, "inter-slide": 0, "main": 0}
    total_pairs = 0
    processed = 0

    # Buffer for detected scenes waiting for extraction (maintains order)
    # We collect results as they finish, but extract in original video order
    pending_results: dict[Path, list[SceneBoundary]] = {}
    next_video_idx = 0  # index into `videos` for the next video to extract

    def _flush_ready():
        """Extract pairs for all videos that are ready in order."""
        nonlocal next_video_idx, total_pairs, processed
        while next_video_idx < len(videos) and videos[next_video_idx] in pending_results:
            video_path = videos[next_video_idx]
            scenes = pending_results.pop(video_path)
            pairs = _extract_for_video(video_path, scenes, output, mode, counters)
            total_pairs += pairs
            processed += 1
            next_video_idx += 1
            if scenes:
                click.echo(f"  [{processed}/{len(videos)}] {video_path.name}: {len(scenes)} scenes -> {pairs} pairs (total: {total_pairs})")
            else:
                click.echo(f"  [{processed}/{len(videos)}] {video_path.name}: no scenes")

    cancelled = False

    if len(videos) == 1:
        video_path, scenes = _detect_scenes_for_video(videos[0])
        pending_results[video_path] = scenes
        _flush_ready()
    else:
        executor = ThreadPoolExecutor(max_workers=max_workers)
        futures = {
            executor.submit(_detect_scenes_for_video, v): v
            for v in videos
        }
        pbar = tqdm(total=len(futures), desc="Processing")
        try:
            for future in as_completed(futures):
                if STOP_EVENT.is_set():
                    break
                pbar.update(1)
                try:
                    video_path, scenes = future.result()
                    pending_results[video_path] = scenes
                    _flush_ready()
                except Exception as e:
                    v = futures[future]
                    click.echo(f"  ERROR {v.name}: {e}", err=True)
                    pending_results[v] = []
                    _flush_ready()
        finally:
            pbar.close()
            cancelled = STOP_EVENT.is_set()
            if cancelled:
                for f in futures:
                    f.cancel()
            executor.shutdown(wait=not cancelled, cancel_futures=cancelled)
            # Flush whatever is ready
            _flush_ready()

    if cancelled:
        click.echo(f"\nCancelled. Saved {total_pairs} pairs from {processed} videos so far -> {output}")
    else:
        click.echo(f"\nDone! {total_pairs} pairs from {len(videos)} videos -> {output}")


if __name__ == "__main__":
    main()
