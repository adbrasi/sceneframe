"""CLI entry point for SceneFrame."""

import logging
import os
import re
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
import cv2
from tqdm import tqdm

from .detector import DEFAULT_WORKERS, SceneBoundary

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


def _init_signal_handler():
    """Install the Ctrl+C handler and reset state."""
    STOP_EVENT.clear()
    global _SIGINT_COUNT
    with _SIGINT_LOCK:
        _SIGINT_COUNT = 0
    signal.signal(signal.SIGINT, _handle_sigint)


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


def _find_videos(input_dir: Path, min_duration: float = 0.0, recursive: bool = True) -> list[Path]:
    """Find all video files in a directory, optionally filtering by minimum duration."""
    if recursive:
        candidates = sorted(
            f for f in input_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        )
    else:
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


def _resolve_videos(input_path: Path, min_duration: float, recursive: bool = True) -> list[Path]:
    """Resolve input path to a list of video files."""
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
            videos.extend(_find_videos(folder, min_duration=min_duration, recursive=recursive))
        return videos
    elif input_path.is_file():
        dur = _get_video_duration(input_path)
        if dur < min_duration:
            click.echo(f"Video too short ({dur:.1f}s < {min_duration:.1f}s min).", err=True)
            raise SystemExit(1)
        return [input_path]
    else:
        return _find_videos(input_path, min_duration=min_duration, recursive=recursive)


def _detect_scenes_for_video(
    video_path: Path,
    engine: str = "pyscenedetect",
    redetect: bool = True,
) -> tuple[Path, list[SceneBoundary]]:
    """Detect and re-segment scenes for a single video. Runs in worker thread."""
    from .detector import detect_scenes, re_detect_long_scenes

    scenes = detect_scenes(video_path, engine=engine, show_progress=False)
    # TransNetV2 already detects dissolves/gradual transitions — re_detect is
    # redundant and expensive (reopens video + runs PySceneDetect CPU per scene).
    if scenes and redetect and engine != "transnetv2":
        scenes = re_detect_long_scenes(video_path, scenes, max_seconds=20.0)
    return video_path, scenes


def _extract_for_video(
    video_path: Path,
    scenes: list[SceneBoundary],
    output_dir: Path,
    mode: str,
    counters: dict[str, int],
    max_pairs: int | None = None,
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
        p1 = extract_intra_scene_pairs(
            video_path, scenes, output_dir / "intra",
            start_index=counters["intra"], max_pairs=max_pairs,
        )
        counters["intra"] += p1
        p2 = extract_inter_scene_pairs_sequential(
            video_path, scenes, output_dir / "inter-seq",
            start_index=counters["inter-seq"], max_pairs=max_pairs,
        )
        counters["inter-seq"] += p2
        p3 = extract_inter_scene_pairs_sliding(
            video_path, scenes, output_dir / "inter-slide",
            start_index=counters["inter-slide"], max_pairs=max_pairs,
        )
        counters["inter-slide"] += p3
        total = p1 + p2 + p3
    else:
        extract_fn = {
            "intra": extract_intra_scene_pairs,
            "inter-seq": extract_inter_scene_pairs_sequential,
            "inter-slide": extract_inter_scene_pairs_sliding,
        }[mode]
        pairs = extract_fn(
            video_path, scenes, output_dir,
            start_index=counters["main"], max_pairs=max_pairs,
        )
        counters["main"] += pairs
        total = pairs

    return total


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """SceneFrame: Extract frame pairs from video scenes for model training."""


# ---------------------------------------------------------------------------
# extract command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for frame pairs.",
)
@click.option(
    "--mode", "-m",
    type=click.Choice(["intra", "inter-seq", "inter-slide", "all"], case_sensitive=False),
    default="all",
    show_default=True,
    help=(
        "Extraction mode: "
        "intra = first+last frame per scene, "
        "inter-seq = consecutive scene pairs (no overlap), "
        "inter-slide = consecutive scenes (sliding window), "
        "all = run all three modes."
    ),
)
@click.option(
    "--min-duration",
    type=float,
    default=DEFAULT_MIN_VIDEO_DURATION,
    show_default=True,
    help="Minimum video duration in seconds.",
)
@click.option(
    "--max-pairs",
    type=int,
    default=None,
    help="Maximum pairs per video (per mode). No limit if not set.",
)
@click.option(
    "--workers", "-w",
    type=int,
    default=None,
    help="Parallel workers for scene detection. Defaults to CPU count - 2.",
)
@click.option(
    "--recursive/--no-recursive",
    default=True,
    show_default=True,
    help="Search for videos recursively in subdirectories.",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    show_default=True,
    help="Skip videos already processed (tracked in processed_videos.log).",
)
@click.option(
    "--engine", "-e",
    type=click.Choice(["pyscenedetect", "transnetv2"], case_sensitive=False),
    default="pyscenedetect",
    show_default=True,
    help="Scene detection engine: pyscenedetect (CPU) or transnetv2 (GPU).",
)
@click.option(
    "--redetect",
    is_flag=True,
    default=False,
    help="Re-segment long scenes (>20s) with AdaptiveDetector. Only applies to pyscenedetect.",
)
def extract(input_path: Path, output: Path, mode: str, min_duration: float, max_pairs: int | None, workers: int | None, recursive: bool, resume: bool, engine: str, redetect: bool):
    """Extract frame pairs from video scenes.

    INPUT_PATH can be a directory, a single video file, or a .txt file with
    one directory path per line (lines starting with # are ignored).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _init_signal_handler()

    input_path = Path(input_path).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    videos = _resolve_videos(input_path, min_duration, recursive=recursive)
    if not videos:
        click.echo("No video files found.", err=True)
        raise SystemExit(1)

    # Resume: filter out already-processed videos
    progress_log = output / "processed_videos.log"
    processed_set: set[str] = set()
    if resume and progress_log.exists():
        processed_set = set(progress_log.read_text(encoding="utf-8").splitlines())
        before = len(videos)
        videos = [v for v in videos if str(v.resolve()) not in processed_set]
        skipped = before - len(videos)
        if skipped:
            click.echo(f"Resume: skipping {skipped} already-processed videos.")
        if not videos:
            click.echo("All videos already processed. Use --no-resume to force reprocessing.")
            raise SystemExit(0)

    cpu_count = os.cpu_count() or 4
    if workers:
        max_workers = workers
    else:
        engine_default = DEFAULT_WORKERS.get(engine)
        max_workers = engine_default if engine_default else max(1, cpu_count - 2)
    max_workers = min(max_workers, len(videos))

    pairs_info = f" (max {max_pairs}/video)" if max_pairs else ""
    click.echo(f"Found {len(videos)} video(s). Mode: {mode}. Engine: {engine}. Workers: {max_workers}{pairs_info}")

    # Scan for existing pair files to avoid label collision on re-runs
    def _find_max_label(directory: Path) -> int:
        """Find the highest existing label number from *_A.jpg files."""
        max_label = 0
        if directory.exists():
            for f in directory.glob("*_A.jpg"):
                m = re.match(r"^(\d+)_A\.jpg$", f.name)
                if m:
                    max_label = max(max_label, int(m.group(1)))
        return max_label

    if mode == "all":
        counters = {
            "intra": _find_max_label(output / "intra"),
            "inter-seq": _find_max_label(output / "inter-seq"),
            "inter-slide": _find_max_label(output / "inter-slide"),
            "main": 0,
        }
    else:
        counters = {
            "intra": 0,
            "inter-seq": 0,
            "inter-slide": 0,
            "main": _find_max_label(output),
        }
    total_pairs = 0
    processed = 0

    pending_results: dict[Path, list[SceneBoundary]] = {}
    next_video_idx = 0

    def _flush_ready():
        nonlocal next_video_idx, total_pairs, processed
        while next_video_idx < len(videos) and videos[next_video_idx] in pending_results:
            video_path = videos[next_video_idx]
            scenes = pending_results.pop(video_path)
            pairs = _extract_for_video(video_path, scenes, output, mode, counters, max_pairs)
            total_pairs += pairs
            processed += 1
            next_video_idx += 1
            # Log processed video for resume
            with open(progress_log, "a", encoding="utf-8") as f:
                f.write(str(video_path.resolve()) + "\n")
            if scenes:
                click.echo(f"  [{processed}/{len(videos)}] {video_path.name}: {len(scenes)} scenes -> {pairs} pairs (total: {total_pairs})")
            else:
                click.echo(f"  [{processed}/{len(videos)}] {video_path.name}: no scenes")

    cancelled = False

    if len(videos) == 1:
        video_path, scenes = _detect_scenes_for_video(videos[0], engine=engine, redetect=redetect)
        pending_results[video_path] = scenes
        _flush_ready()
    else:
        executor = ThreadPoolExecutor(max_workers=max_workers)
        futures = {
            executor.submit(_detect_scenes_for_video, v, engine, redetect): v
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
            _flush_ready()

    if cancelled:
        click.echo(f"\nCancelled. Saved {total_pairs} pairs from {processed} videos so far -> {output}")
    else:
        click.echo(f"\nDone! {total_pairs} pairs from {len(videos)} videos -> {output}")


# ---------------------------------------------------------------------------
# clean command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--no-solid", is_flag=True, help="Skip solid-color removal.")
@click.option("--no-duplicates", is_flag=True, help="Skip duplicate removal.")
@click.option(
    "--nsfw/--no-nsfw",
    default=False,
    show_default=True,
    help="Enable NSFW filter (requires torch + transformers).",
)
@click.option(
    "--keep-nsfw/--remove-nsfw",
    default=True,
    show_default=True,
    help="Keep NSFW images (reverse filter) or remove them.",
)
@click.option("--nsfw-confidence", type=float, default=0.5, show_default=True, help="NSFW classification confidence threshold.")
@click.option("--nsfw-batch-size", type=int, default=64, show_default=True, help="Batch size for NSFW inference.")
@click.option("--nsfw-device", type=str, default=None, help="Device for NSFW model (cuda/cpu). Auto-detects if not set.")
@click.option("--similarity", type=float, default=0.96, show_default=True, help="Min cosine similarity for duplicate detection (0-1). Higher = stricter.")
@click.option("--solid-threshold", type=float, default=12.0, show_default=True, help="Max std-dev per channel to consider solid color.")
@click.option("--workers", "-w", type=int, default=16, show_default=True, help="Parallel workers for image processing.")
@click.option("--dry-run", is_flag=True, help="Show what would be removed without deleting.")
def clean(
    directory: Path,
    no_solid: bool,
    no_duplicates: bool,
    nsfw: bool,
    keep_nsfw: bool,
    nsfw_confidence: float,
    nsfw_batch_size: int,
    nsfw_device: str | None,
    similarity: float,
    solid_threshold: float,
    workers: int,
    dry_run: bool,
):
    """Clean image pairs: remove solid colors, duplicates, and optionally filter by NSFW.

    DIRECTORY should contain image pairs named NNNNNN_A.jpg / NNNNNN_B.jpg.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    from .cleaner import clean_directory

    directory = Path(directory).resolve()

    if dry_run:
        click.echo("[DRY RUN] No files will be deleted.")

    stats = clean_directory(
        directory,
        remove_solid=not no_solid,
        remove_dups=not no_duplicates,
        nsfw=nsfw,
        keep_nsfw=keep_nsfw,
        nsfw_confidence=nsfw_confidence,
        nsfw_batch_size=nsfw_batch_size,
        nsfw_device=nsfw_device,
        similarity=similarity,
        solid_threshold=solid_threshold,
        workers=workers,
        dry_run=dry_run,
    )

    click.echo(f"\n--- Cleaning summary ---")
    click.echo(f"  Solid color pairs removed: {stats['solid_removed']}")
    click.echo(f"  Duplicate pairs removed:   {stats['duplicates_removed']}")
    if nsfw:
        click.echo(f"  NSFW filtered pairs:       {stats['nsfw_removed']}")
    click.echo(f"  Orphan pairs removed:      {stats['orphans_removed']}")
    click.echo(f"  Total removed:             {stats['total_removed']}")
    click.echo(f"  Remaining pairs:           {stats['remaining']}")


# ---------------------------------------------------------------------------
# control command (depth + canny)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--percentage", "-p", type=float, default=100.0, show_default=True, help="Total % of _B images that receive a control image (0-100).")
@click.option("--depth", type=float, default=100.0, show_default=True, help="% of selected images that get depth maps. Must sum to 100 with --canny and --image-base.")
@click.option("--canny", type=float, default=0.0, show_default=True, help="% of selected images that get canny edges. Must sum to 100 with --depth and --image-base.")
@click.option("--image-base", type=float, default=0.0, show_default=True, help="% of selected images that get a copy as _image_base.jpg. Must sum to 100 with --depth and --canny.")
@click.option("--image-base-source", type=click.Choice(["A", "B"], case_sensitive=False), default="A", show_default=True, help="Which image to copy for image_base: A or B.")
@click.option("--batch-size", "-b", type=int, default=32, show_default=True, help="Batch size for depth GPU inference. 32 for 32GB VRAM, 64+ for 96GB.")
@click.option("--device", type=str, default=None, help="Device for depth inference (cuda/cpu). Auto-detects if not set.")
@click.option(
    "--model", "-m",
    type=str,
    default="large",
    show_default=True,
    help="Depth model preset (small/base/large) or full HuggingFace model ID.",
)
@click.option("--canny-low", type=int, default=100, show_default=True, help="Canny lower threshold.")
@click.option("--canny-high", type=int, default=200, show_default=True, help="Canny upper threshold.")
@click.option("--workers", "-w", type=int, default=16, show_default=True, help="Parallel workers for canny/image_base I/O.")
@click.option("--seed", type=int, default=None, help="Random seed for reproducible subset selection.")
def control(
    directory: Path,
    percentage: float,
    depth: float,
    canny: float,
    image_base: float,
    image_base_source: str,
    batch_size: int,
    device: str | None,
    model: str,
    canny_low: int,
    canny_high: int,
    workers: int,
    seed: int | None,
):
    """Generate control images from _B images using depth, canny, and/or image_base.

    DIRECTORY should contain image pairs named NNNNNN_A.jpg / NNNNNN_B.jpg.

    First, --percentage selects how many _B images are processed.
    Then, --depth, --canny, and --image-base split that selection (must sum to 100).

    \b
    Control types:
      depth      → depth map saved as _C.jpg (GPU, Depth Anything V2)
      canny      → canny edges saved as _C.jpg (CPU, OpenCV)
      image-base → copy of _A saved as _image_base.jpg

    \b
    Examples:
      sceneframe control ./output                                            # 100% depth
      sceneframe control ./output -p 50 --depth 50 --canny 50               # 25% depth + 25% canny
      sceneframe control ./output -p 100 --depth 25 --canny 25 --image-base 50
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not (0 < percentage <= 100):
        click.echo("Error: --percentage must be between 0 and 100.", err=True)
        raise SystemExit(1)

    if abs(depth + canny + image_base - 100) > 0.01:
        click.echo(
            f"Error: --depth ({depth}) + --canny ({canny}) + --image-base ({image_base}) must equal 100.",
            err=True,
        )
        raise SystemExit(1)

    from .depth import _get_candidates, generate_canny_maps, generate_depth_maps, generate_image_base

    import random

    directory = Path(directory).resolve()
    all_candidates = _get_candidates(directory)

    if not all_candidates:
        click.echo("No _B images without existing _C found.")
        raise SystemExit(0)

    # Step 1: select total pool
    if seed is not None:
        random.seed(seed)

    if percentage < 100.0:
        pool_size = max(1, int(len(all_candidates) * percentage / 100.0))
        pool = sorted(random.sample(all_candidates, pool_size))
    else:
        pool = list(all_candidates)

    # Step 2: split pool into depth, canny, and image_base
    depth_count = round(len(pool) * depth / 100.0)
    canny_count = round(len(pool) * canny / 100.0)
    # Clamp to prevent overflow when independent rounds sum > pool size
    canny_count = min(canny_count, len(pool) - depth_count)
    # image_base gets the remainder (pool[depth_count + canny_count:])
    random.shuffle(pool)
    depth_candidates = sorted(pool[:depth_count])
    canny_candidates = sorted(pool[depth_count:depth_count + canny_count])
    base_candidates = sorted(pool[depth_count + canny_count:])

    parts = []
    if depth_candidates:
        parts.append(f"{len(depth_candidates)} depth")
    if canny_candidates:
        parts.append(f"{len(canny_candidates)} canny")
    if base_candidates:
        parts.append(f"{len(base_candidates)} image_base")

    click.echo(
        f"Found {len(all_candidates)} _B images → "
        f"selecting {len(pool)} ({percentage:.0f}%): "
        + " + ".join(parts)
    )

    depth_saved = 0
    canny_saved = 0
    base_saved = 0

    if depth_candidates:
        depth_saved = generate_depth_maps(
            depth_candidates,
            batch_size=batch_size,
            device=device,
            model=model,
        )

    if canny_candidates:
        canny_saved = generate_canny_maps(
            canny_candidates,
            low_threshold=canny_low,
            high_threshold=canny_high,
            workers=workers,
        )

    if base_candidates:
        base_saved = generate_image_base(
            base_candidates, source=image_base_source, workers=workers,
        )

    total = depth_saved + canny_saved + base_saved
    click.echo(f"\nDone! Generated {depth_saved} depth + {canny_saved} canny + {base_saved} image_base = {total} control images -> {directory}")


# ---------------------------------------------------------------------------
# Backwards-compatible entry point: `sceneframe` without subcommand = extract
# ---------------------------------------------------------------------------

# Keep `main` as an alias so `python -m sceneframe.cli` still works
main = cli


if __name__ == "__main__":
    cli()
