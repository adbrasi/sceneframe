import logging
import shutil
import sys
import tempfile
import threading
import queue
from pathlib import Path

import click

from .config import (
    DEFAULT_CLASSIFY_WORKERS,
    DEFAULT_MAX_DURATION,
    DEFAULT_MIN_DURATION,
    DEFAULT_MODEL,
    DEFAULT_RESEGMENT_THRESHOLD,
    DEFAULT_SEGMENT_WORKERS,
    get_api_key,
    load_categories,
    load_system_prompt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _is_slow_mount(path: Path) -> bool:
    """Check if path is on a Windows mount (slow I/O in WSL2)."""
    resolved = str(path.resolve())
    return resolved.startswith("/mnt/")


def _resolve_input(input_path: Path) -> tuple[Path, list[Path]]:
    """Resolve input to a directory and list of video files.

    Accepts a directory (all videos in it) or a single video file.
    Returns (effective_dir, video_list).
    """
    video_extensions = {".mp4", ".mkv", ".avi", ".webm", ".mov"}
    input_path = Path(input_path).resolve()

    if input_path.is_file():
        if input_path.suffix.lower() not in video_extensions:
            raise click.BadParameter(f"Not a supported video file: {input_path.name}")
        return input_path.parent, [input_path]

    videos = sorted(
        f for f in input_path.iterdir() if f.suffix.lower() in video_extensions
    )
    return input_path, videos


def _copy_single_video(video_path: Path) -> tuple[Path, bool]:
    """Copy a single video to local filesystem if on slow mount. Returns (local_path, was_copied)."""
    if not _is_slow_mount(video_path):
        return video_path, False

    file_size = video_path.stat().st_size
    free = shutil.disk_usage(tempfile.gettempdir()).free
    if file_size > free * 0.9:
        click.echo(
            f"Warning: not enough temp space for {video_path.name} "
            f"({free // (1024*1024)} MB free, need {file_size // (1024*1024)} MB). "
            f"Processing from original path.",
            err=True,
        )
        return video_path, False

    local_dir = Path(tempfile.mkdtemp(prefix="dataset_maker_"))
    local_path = local_dir / video_path.name
    click.echo(
        f"Copying {video_path.name} ({file_size // (1024*1024)} MB) to {local_dir}..."
    )
    shutil.copy2(video_path, local_path)
    return local_path, True


def _cleanup_single_video(local_path: Path, was_copied: bool) -> None:
    """Remove temp copy of a single video."""
    if was_copied:
        shutil.rmtree(local_path.parent, ignore_errors=True)


def _video_stem(video_path: Path) -> str:
    """Get a clean folder name from a video filename."""
    return video_path.stem


def _load_completed(output_dir: Path) -> set[str]:
    """Load list of already-completed video stems for resume."""
    completed_file = output_dir / "completed.txt"
    if not completed_file.exists():
        return set()
    return {line.strip() for line in completed_file.read_text().splitlines() if line.strip()}


def _mark_completed(output_dir: Path, video_stem: str) -> None:
    """Append a video stem to the completed list."""
    completed_file = output_dir / "completed.txt"
    with open(completed_file, "a") as f:
        f.write(video_stem + "\n")


@click.group()
@click.version_option(version="0.3.0")
def main():
    """Dataset Maker - Create labeled video datasets from anime/series episodes."""
    pass


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for clips",
)
@click.option(
    "--segment-workers",
    default=DEFAULT_SEGMENT_WORKERS,
    show_default=True,
    help="Parallel segmentation workers",
)
@click.option(
    "--min-duration",
    default=DEFAULT_MIN_DURATION,
    show_default=True,
    help="Minimum clip duration (seconds)",
)
@click.option(
    "--max-duration",
    default=DEFAULT_MAX_DURATION,
    show_default=True,
    help="Maximum clip duration (seconds)",
)
@click.option(
    "--resegment-threshold",
    default=DEFAULT_RESEGMENT_THRESHOLD,
    show_default=True,
    help="Re-segment clips longer than this (seconds). 0 to disable.",
)
@click.option(
    "--detector",
    type=click.Choice(["transnet", "pyscene", "both"], case_sensitive=False),
    default="both",
    show_default=True,
    help="Scene detection method.",
)
def segment(input_path, output, segment_workers, min_duration, max_duration, resegment_threshold, detector):
    """Segment episodes into scene clips. INPUT_PATH can be a directory or a single video file."""
    from .segmenter import segment_episodes, re_segment_long_clips
    from .organizer import save_metadata

    input_dir, videos = _resolve_input(input_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    completed = _load_completed(output)
    pending = [v for v in videos if _video_stem(v) not in completed]

    if completed:
        click.echo(f"Resuming: {len(completed)} videos already done, {len(pending)} remaining")

    if not pending:
        click.echo("All videos already segmented.")
        return

    for i, video in enumerate(pending, 1):
        click.echo(f"\n=== [{i}/{len(pending)}] Segmenting: {video.name} ===")
        local_path, was_copied = _copy_single_video(video)
        try:
            video_out = output / _video_stem(video)
            metadata = segment_episodes(
                local_path.parent, video_out, min_duration, max_duration,
                segment_workers, episodes=[local_path], detector=detector,
            )
            if not metadata:
                click.echo(f"No clips generated for {video.name}.", err=True)
                continue

            if resegment_threshold > 0:
                metadata = re_segment_long_clips(
                    video_out, metadata, threshold=resegment_threshold, min_duration=min_duration
                )

            metadata_path = video_out / "segments_metadata.jsonl"
            save_metadata(metadata_path, metadata)
            _mark_completed(output, _video_stem(video))
            click.echo(f"Segmented {video.name} into {len(metadata)} clips.")
        finally:
            _cleanup_single_video(local_path, was_copied)


@main.command()
@click.argument("clips_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for organized dataset",
)
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    show_default=True,
    help="OpenRouter model ID",
)
@click.option(
    "--classify-workers",
    default=DEFAULT_CLASSIFY_WORKERS,
    show_default=True,
    help="Parallel classification workers",
)
@click.option(
    "--api-key",
    default=None,
    help="OpenRouter API key (or set OPENROUTER_API_KEY)",
)
@click.option(
    "--categories-file",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Path to categories.json",
)
@click.option(
    "--system-prompt",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Path to system_prompt.txt",
)
def classify(clips_dir, output, model, classify_workers, api_key, categories_file, system_prompt):
    """Classify clips and organize into labeled dataset."""
    from .classifier import classify_clips
    from .organizer import organize_dataset, load_metadata

    clips_dir = Path(clips_dir).resolve()
    output = Path(output).resolve()

    api_key = get_api_key(api_key)
    categories = load_categories(categories_file)
    prompt = load_system_prompt(system_prompt)

    metadata_path = clips_dir / "segments_metadata.jsonl"
    existing_metadata = (
        load_metadata(metadata_path) if metadata_path.exists() else None
    )

    results = classify_clips(
        clips_dir, api_key, model, categories, prompt, classify_workers, existing_metadata
    )
    if results:
        organize_dataset(clips_dir, output, results)
        click.echo(f"Classified and organized {len(results)} clips into {output}")
    else:
        click.echo("No clips classified.", err=True)
        sys.exit(1)


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for final dataset",
)
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    show_default=True,
    help="OpenRouter model ID",
)
@click.option(
    "--segment-workers",
    default=DEFAULT_SEGMENT_WORKERS,
    show_default=True,
    help="Parallel segmentation workers",
)
@click.option(
    "--classify-workers",
    default=DEFAULT_CLASSIFY_WORKERS,
    show_default=True,
    help="Parallel classification workers",
)
@click.option(
    "--min-duration",
    default=DEFAULT_MIN_DURATION,
    show_default=True,
    help="Minimum clip duration (seconds)",
)
@click.option(
    "--max-duration",
    default=DEFAULT_MAX_DURATION,
    show_default=True,
    help="Maximum clip duration (seconds)",
)
@click.option(
    "--resegment-threshold",
    default=DEFAULT_RESEGMENT_THRESHOLD,
    show_default=True,
    help="Re-segment clips longer than this (seconds). 0 to disable.",
)
@click.option(
    "--api-key",
    default=None,
    help="OpenRouter API key (or set OPENROUTER_API_KEY)",
)
@click.option(
    "--categories-file",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Path to categories.json",
)
@click.option(
    "--system-prompt",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Path to system_prompt.txt",
)
@click.option(
    "--detector",
    type=click.Choice(["transnet", "pyscene", "both"], case_sensitive=False),
    default="both",
    show_default=True,
    help="Scene detection method.",
)
def run(
    input_path, output, model,
    segment_workers, classify_workers,
    min_duration, max_duration, resegment_threshold,
    api_key, categories_file, system_prompt, detector,
):
    """Run full pipeline: segment -> classify -> organize.

    Each video gets its own folder. Progress is saved per video so you can
    resume after a crash. Segmentation of the next video runs in parallel
    with classification of the current one (GPU + network don't compete).
    """
    from .segmenter import segment_episodes, re_segment_long_clips
    from .classifier import classify_clips
    from .organizer import organize_dataset, save_metadata

    input_dir, videos = _resolve_input(input_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    api_key = get_api_key(api_key)
    categories = load_categories(categories_file)
    prompt = load_system_prompt(system_prompt)

    # Resume: skip already completed videos
    completed = _load_completed(output)
    pending = [v for v in videos if _video_stem(v) not in completed]

    if completed:
        click.echo(f"Resuming: {len(completed)} videos already done, {len(pending)} remaining")

    if not pending:
        click.echo("All videos already processed.")
        return

    # Pipeline: segment next video while classifying current one
    # Queue holds (video_path, video_output_dir, clips_dir, seg_metadata)
    seg_queue = queue.Queue(maxsize=1)
    seg_error = [None]

    def _segmentation_producer():
        """Runs in background thread: segments videos one by one."""
        try:
            for i, video in enumerate(pending):
                try:
                    stem = _video_stem(video)
                    video_out = output / stem
                    clips_dir = video_out / "_clips"

                    # Copy single video from /mnt/ if needed
                    local_path, was_copied = _copy_single_video(video)
                    try:
                        click.echo(f"\n[Segment {i+1}/{len(pending)}] {video.name}")
                        seg_metadata = segment_episodes(
                            local_path.parent, clips_dir, min_duration, max_duration,
                            segment_workers, episodes=[local_path], detector=detector,
                        )

                        if seg_metadata and resegment_threshold > 0:
                            seg_metadata = re_segment_long_clips(
                                clips_dir, seg_metadata,
                                threshold=resegment_threshold,
                                min_duration=min_duration,
                            )

                        if seg_metadata:
                            save_metadata(clips_dir / "segments_metadata.jsonl", seg_metadata)

                        seg_queue.put((video, video_out, clips_dir, seg_metadata))
                    finally:
                        _cleanup_single_video(local_path, was_copied)
                except Exception as e:
                    logger.error("Segmentation failed for %s: %s", video.name, e)
                    seg_queue.put((video, None, None, None))
        finally:
            seg_queue.put(None)  # Sentinel always sent, even on KeyboardInterrupt

    # Start segmentation in background thread
    seg_thread = threading.Thread(target=_segmentation_producer, daemon=True)
    seg_thread.start()

    total_clips = 0
    processed = 0

    while True:
        item = seg_queue.get()
        if item is None:
            break

        video, video_out, clips_dir, seg_metadata = item
        stem = _video_stem(video)
        processed += 1

        if not seg_metadata:
            click.echo(f"[{processed}/{len(pending)}] No clips for {video.name}, skipping.")
            _mark_completed(output, stem)
            continue

        click.echo(f"\n{'='*60}")
        click.echo(f"[{processed}/{len(pending)}] Classifying: {video.name} ({len(seg_metadata)} clips)")
        click.echo(f"{'='*60}")

        # Classify
        classified = classify_clips(
            clips_dir, api_key, model, categories, prompt, classify_workers, seg_metadata
        )

        if not classified:
            click.echo(f"Classification failed for {video.name}, clips kept in {clips_dir}", err=True)
            continue

        # Organize into per-video category folders
        organize_dataset(clips_dir, video_out, classified, move=True)

        # Cleanup intermediate clips dir
        shutil.rmtree(clips_dir, ignore_errors=True)

        _mark_completed(output, stem)
        total_clips += len(classified)
        click.echo(f"[{processed}/{len(pending)}] Done: {len(classified)} clips → {video_out}")

    seg_thread.join()

    click.echo(f"\nDone! {total_clips} total clips from {processed} videos at {output}")


if __name__ == "__main__":
    main()
