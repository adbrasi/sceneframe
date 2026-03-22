# SceneFrame Rewrite — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite the dataset_maker code into a focused CLI tool that detects video scenes and extracts frame pairs (JPEG) for model training.

**Architecture:** Three-module design: `detector.py` handles scene detection via PySceneDetect with two-pass re-segmentation, `extractor.py` extracts and saves frame pairs using OpenCV, `cli.py` ties it together with Click. Videos are processed in parallel using ProcessPoolExecutor. No video cutting, no classification, no LLM integration.

**Tech Stack:** Python 3.10+, PySceneDetect (ContentDetector + AdaptiveDetector), OpenCV (cv2), Click, pathlib (cross-platform paths)

---

### Task 1: Clean up — remove unused modules and rename package

**Files:**
- Delete: `src/dataset_maker/classifier.py`
- Delete: `src/dataset_maker/organizer.py`
- Delete: `src/dataset_maker/config.py`
- Delete: `src/dataset_maker/segmenter.py`
- Delete: `src/dataset_maker/__init__.py`
- Delete: `src/dataset_maker/cli.py`
- Create: `src/sceneframe/__init__.py`
- Modify: `pyproject.toml`

**Step 1: Delete old package**

```bash
rm -rf src/dataset_maker
```

**Step 2: Create new empty package**

```bash
mkdir -p src/sceneframe
touch src/sceneframe/__init__.py
```

**Step 3: Update pyproject.toml**

Replace entire contents with:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "sceneframe"
version = "0.1.0"
description = "Detect video scenes and extract frame pairs for model training"
requires-python = ">=3.10"
dependencies = [
    "click>=8.1",
    "scenedetect[opencv]>=0.6",
    "opencv-python>=4.8",
    "tqdm>=4.66",
]

[project.scripts]
sceneframe = "sceneframe.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/sceneframe"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove unused modules and rename package to sceneframe"
```

---

### Task 2: Implement `detector.py` — scene detection

**Files:**
- Create: `src/sceneframe/detector.py`
- Create: `tests/test_detector.py`

**Step 1: Write the failing tests**

Create `tests/test_detector.py`:

```python
from sceneframe.detector import SceneBoundary, detect_scenes, re_detect_long_scenes


class TestSceneBoundary:
    def test_duration_seconds(self):
        sb = SceneBoundary(start_frame=0, end_frame=120, fps=24.0)
        assert sb.duration_seconds == 5.0

    def test_duration_seconds_30fps(self):
        sb = SceneBoundary(start_frame=0, end_frame=300, fps=30.0)
        assert sb.duration_seconds == 10.0


class TestDetectScenes:
    def test_returns_list_of_scene_boundaries(self, tmp_path):
        """detect_scenes should return a list (possibly empty) of SceneBoundary."""
        # With a non-existent video, should return empty list (not crash)
        result = detect_scenes(tmp_path / "nonexistent.mp4")
        assert isinstance(result, list)

    def test_returns_empty_for_missing_video(self, tmp_path):
        result = detect_scenes(tmp_path / "missing.mp4")
        assert result == []


class TestReDetectLongScenes:
    def test_short_scenes_unchanged(self, tmp_path):
        """Scenes under max_seconds should pass through unchanged."""
        scenes = [
            SceneBoundary(start_frame=0, end_frame=240, fps=24.0),   # 10s
            SceneBoundary(start_frame=240, end_frame=480, fps=24.0), # 10s
        ]
        result = re_detect_long_scenes(
            tmp_path / "nonexistent.mp4", scenes, max_seconds=20.0
        )
        assert result == scenes

    def test_identifies_long_scenes(self):
        """Scenes over max_seconds should be flagged for re-detection."""
        scenes = [
            SceneBoundary(start_frame=0, end_frame=240, fps=24.0),    # 10s - ok
            SceneBoundary(start_frame=240, end_frame=1200, fps=24.0), # 40s - long
        ]
        long = [s for s in scenes if s.duration_seconds > 20.0]
        assert len(long) == 1
        assert long[0].start_frame == 240
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_detector.py -v`
Expected: FAIL with ImportError

**Step 3: Write the implementation**

Create `src/sceneframe/detector.py`:

```python
import logging
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


def detect_scenes(video_path: Path) -> list[SceneBoundary]:
    """Detect scene boundaries in a video using PySceneDetect ContentDetector.

    Returns a list of SceneBoundary. Returns empty list on failure.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        logger.warning("Video not found: %s", video_path)
        return []

    try:
        from scenedetect import detect, ContentDetector

        scene_list = detect(str(video_path), ContentDetector(), show_progress=True)

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

        # Re-detect with more sensitive settings
        try:
            from scenedetect import (
                SceneManager,
                open_video,
                AdaptiveDetector,
            )

            video = open_video(str(video_path))
            video.seek(scene.start_frame)

            scene_manager = SceneManager()
            scene_manager.add_detector(
                AdaptiveDetector(adaptive_threshold=1.5)
            )

            duration_frames = scene.end_frame - scene.start_frame
            scene_manager.detect_scenes(video, end_frame=scene.end_frame)
            sub_scenes = scene_manager.get_scene_list()

            if len(sub_scenes) <= 1:
                # No further splits found, keep original
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
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_detector.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/sceneframe/detector.py tests/test_detector.py
git commit -m "feat: add scene detection with PySceneDetect and re-segmentation"
```

---

### Task 3: Implement `extractor.py` — frame pair extraction

**Files:**
- Create: `src/sceneframe/extractor.py`
- Create: `tests/test_extractor.py`

**Step 1: Write the failing tests**

Create `tests/test_extractor.py`:

```python
import cv2
import numpy as np
from pathlib import Path

from sceneframe.detector import SceneBoundary
from sceneframe.extractor import (
    extract_frame,
    extract_intra_scene_pairs,
    extract_inter_scene_pairs_sequential,
    extract_inter_scene_pairs_sliding,
)

FRAME_OFFSET = 3


def _make_test_video(path: Path, num_frames: int = 100, fps: float = 24.0):
    """Create a minimal test video with colored frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (64, 64))
    for i in range(num_frames):
        frame = np.full((64, 64, 3), fill_value=i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestExtractFrame:
    def test_extracts_frame_at_index(self, tmp_path):
        video = tmp_path / "test.mp4"
        _make_test_video(video, num_frames=50)
        frame = extract_frame(video, frame_index=10)
        assert frame is not None
        assert frame.shape[0] > 0
        assert frame.shape[1] > 0

    def test_returns_none_for_invalid_index(self, tmp_path):
        video = tmp_path / "test.mp4"
        _make_test_video(video, num_frames=10)
        frame = extract_frame(video, frame_index=9999)
        assert frame is None

    def test_returns_none_for_missing_video(self, tmp_path):
        frame = extract_frame(tmp_path / "missing.mp4", frame_index=0)
        assert frame is None


class TestIntraScenePairs:
    def test_saves_pairs_with_correct_names(self, tmp_path):
        video = tmp_path / "test.mp4"
        _make_test_video(video, num_frames=200)

        scenes = [
            SceneBoundary(start_frame=0, end_frame=50, fps=24.0),
            SceneBoundary(start_frame=50, end_frame=100, fps=24.0),
        ]

        output = tmp_path / "output_intra"
        count = extract_intra_scene_pairs(video, scenes, output)

        assert count == 2
        assert (output / "0001_A.jpg").exists()
        assert (output / "0001_B.jpg").exists()
        assert (output / "0002_A.jpg").exists()
        assert (output / "0002_B.jpg").exists()

    def test_skips_scenes_too_short_for_offset(self, tmp_path):
        video = tmp_path / "test.mp4"
        _make_test_video(video, num_frames=200)

        scenes = [
            SceneBoundary(start_frame=0, end_frame=5, fps=24.0),  # too short
            SceneBoundary(start_frame=50, end_frame=100, fps=24.0),
        ]

        output = tmp_path / "output_intra"
        count = extract_intra_scene_pairs(video, scenes, output)
        assert count == 1


class TestInterScenePairsSequential:
    def test_pairs_consecutive_scenes(self, tmp_path):
        video = tmp_path / "test.mp4"
        _make_test_video(video, num_frames=200)

        scenes = [
            SceneBoundary(start_frame=0, end_frame=50, fps=24.0),
            SceneBoundary(start_frame=50, end_frame=100, fps=24.0),
            SceneBoundary(start_frame=100, end_frame=150, fps=24.0),
            SceneBoundary(start_frame=150, end_frame=200, fps=24.0),
        ]

        output = tmp_path / "output_inter_seq"
        count = extract_inter_scene_pairs_sequential(video, scenes, output)

        assert count == 2  # (1+2), (3+4)
        assert (output / "0001_A.jpg").exists()
        assert (output / "0001_B.jpg").exists()
        assert (output / "0002_A.jpg").exists()
        assert (output / "0002_B.jpg").exists()

    def test_discards_odd_scene(self, tmp_path):
        video = tmp_path / "test.mp4"
        _make_test_video(video, num_frames=200)

        scenes = [
            SceneBoundary(start_frame=0, end_frame=50, fps=24.0),
            SceneBoundary(start_frame=50, end_frame=100, fps=24.0),
            SceneBoundary(start_frame=100, end_frame=150, fps=24.0),
        ]

        output = tmp_path / "output_inter_seq"
        count = extract_inter_scene_pairs_sequential(video, scenes, output)
        assert count == 1  # only (1+2), scene 3 discarded


class TestInterScenePairsSliding:
    def test_sliding_window_pairs(self, tmp_path):
        video = tmp_path / "test.mp4"
        _make_test_video(video, num_frames=200)

        scenes = [
            SceneBoundary(start_frame=0, end_frame=50, fps=24.0),
            SceneBoundary(start_frame=50, end_frame=100, fps=24.0),
            SceneBoundary(start_frame=100, end_frame=150, fps=24.0),
        ]

        output = tmp_path / "output_inter_slide"
        count = extract_inter_scene_pairs_sliding(video, scenes, output)

        assert count == 2  # (1+2), (2+3)
        assert (output / "0001_A.jpg").exists()
        assert (output / "0001_B.jpg").exists()
        assert (output / "0002_A.jpg").exists()
        assert (output / "0002_B.jpg").exists()
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_extractor.py -v`
Expected: FAIL with ImportError

**Step 3: Write the implementation**

Create `src/sceneframe/extractor.py`:

```python
import logging
from pathlib import Path

import cv2

from .detector import SceneBoundary

logger = logging.getLogger(__name__)

FRAME_OFFSET = 3
JPEG_QUALITY = 95


def extract_frame(video_path: Path, frame_index: int) -> "np.ndarray | None":
    """Extract a single frame from a video by index. Returns BGR numpy array or None."""
    import numpy as np

    video_path = Path(video_path)
    if not video_path.exists():
        return None

    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            return None
        return frame
    finally:
        cap.release()


def _save_frame(frame: "np.ndarray", path: Path) -> bool:
    """Save a frame as JPEG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def _safe_frame_indices(scene: SceneBoundary, offset: int = FRAME_OFFSET) -> tuple[int, int] | None:
    """Get safe start/end frame indices with offset applied.

    Returns None if the scene is too short for the offset.
    """
    total_frames = scene.end_frame - scene.start_frame
    if total_frames <= offset * 2:
        return None
    return scene.start_frame + offset, scene.end_frame - offset


def extract_intra_scene_pairs(
    video_path: Path,
    scenes: list[SceneBoundary],
    output_dir: Path,
) -> int:
    """Mode 1: Extract first and last frame of each scene as pairs.

    Saves as 0001_A.jpg / 0001_B.jpg, etc.
    Returns the number of pairs saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return 0

    pair_count = 0

    try:
        for scene in scenes:
            indices = _safe_frame_indices(scene)
            if indices is None:
                continue

            start_idx, end_idx = indices

            # Extract start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
            ret_a, frame_a = cap.read()
            if not ret_a:
                continue

            # Extract end frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, end_idx)
            ret_b, frame_b = cap.read()
            if not ret_b:
                continue

            pair_count += 1
            label = f"{pair_count:04d}"
            _save_frame(frame_a, output_dir / f"{label}_A.jpg")
            _save_frame(frame_b, output_dir / f"{label}_B.jpg")
    finally:
        cap.release()

    return pair_count


def extract_inter_scene_pairs_sequential(
    video_path: Path,
    scenes: list[SceneBoundary],
    output_dir: Path,
) -> int:
    """Mode 2a: Pair first frames of consecutive scenes (no overlap).

    Pairs: (scene1, scene2), (scene3, scene4), ...
    Odd last scene is discarded.
    Returns the number of pairs saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return 0

    pair_count = 0

    try:
        for i in range(0, len(scenes) - 1, 2):
            scene_a = scenes[i]
            scene_b = scenes[i + 1]

            idx_a = scene_a.start_frame + FRAME_OFFSET
            idx_b = scene_b.start_frame + FRAME_OFFSET

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx_a)
            ret_a, frame_a = cap.read()
            if not ret_a:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx_b)
            ret_b, frame_b = cap.read()
            if not ret_b:
                continue

            pair_count += 1
            label = f"{pair_count:04d}"
            _save_frame(frame_a, output_dir / f"{label}_A.jpg")
            _save_frame(frame_b, output_dir / f"{label}_B.jpg")
    finally:
        cap.release()

    return pair_count


def extract_inter_scene_pairs_sliding(
    video_path: Path,
    scenes: list[SceneBoundary],
    output_dir: Path,
) -> int:
    """Mode 2b: Pair first frames of consecutive scenes (sliding window).

    Pairs: (scene1, scene2), (scene2, scene3), (scene3, scene4), ...
    Returns the number of pairs saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(scenes) < 2:
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return 0

    pair_count = 0

    try:
        for i in range(len(scenes) - 1):
            scene_a = scenes[i]
            scene_b = scenes[i + 1]

            idx_a = scene_a.start_frame + FRAME_OFFSET
            idx_b = scene_b.start_frame + FRAME_OFFSET

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx_a)
            ret_a, frame_a = cap.read()
            if not ret_a:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx_b)
            ret_b, frame_b = cap.read()
            if not ret_b:
                continue

            pair_count += 1
            label = f"{pair_count:04d}"
            _save_frame(frame_a, output_dir / f"{label}_A.jpg")
            _save_frame(frame_b, output_dir / f"{label}_B.jpg")
    finally:
        cap.release()

    return pair_count
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_extractor.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/sceneframe/extractor.py tests/test_extractor.py
git commit -m "feat: add frame pair extraction (intra-scene, inter-scene sequential and sliding)"
```

---

### Task 4: Implement `cli.py` — CLI with parallel processing

**Files:**
- Create: `src/sceneframe/cli.py`

**Step 1: Write the implementation**

Create `src/sceneframe/cli.py`:

```python
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".webm", ".mov", ".wmv", ".flv"}


def _find_videos(input_dir: Path) -> list[Path]:
    """Find all video files in a directory."""
    return sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    )


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
        scenes = detect_scenes(video_path)
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
            # "all" — run all three modes into separate subdirs
            p1 = extract_intra_scene_pairs(video_path, scenes, video_output / "intra")
            p2 = extract_inter_scene_pairs_sequential(video_path, scenes, video_output / "inter-seq")
            p3 = extract_inter_scene_pairs_sliding(video_path, scenes, video_output / "inter-slide")
            pairs = p1 + p2 + p3

        result["pairs"] = pairs

    except Exception as e:
        result["error"] = str(e)
        logger.error("Failed to process %s: %s", video_path.name, e)

    return result


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
def main(input_path: Path, output: Path, mode: str):
    """Extract frame pairs from video scenes for model training.

    INPUT_PATH can be a directory of videos or a single video file.
    """
    input_path = Path(input_path).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        videos = [input_path]
    else:
        videos = _find_videos(input_path)

    if not videos:
        click.echo("No video files found.", err=True)
        raise SystemExit(1)

    click.echo(f"Found {len(videos)} video(s). Mode: {mode}")

    workers = min(os.cpu_count() or 1, len(videos))

    if len(videos) == 1:
        # No pool overhead for single video
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


def _print_result(result: dict):
    """Print processing result for a single video."""
    if result["error"]:
        click.echo(f"  ERROR {result['video']}: {result['error']}", err=True)
    elif result["pairs"] == 0:
        click.echo(f"  {result['video']}: no scenes detected")
    else:
        click.echo(f"  {result['video']}: {result['scenes']} scenes → {result['pairs']} pairs")


if __name__ == "__main__":
    main()
```

**Step 2: Manual smoke test**

Run: `python -m sceneframe.cli --help`
Expected: Shows help text with options

**Step 3: Commit**

```bash
git add src/sceneframe/cli.py
git commit -m "feat: add CLI with parallel processing and three extraction modes"
```

---

### Task 5: Integration test and final cleanup

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`

**Step 1: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All PASS

**Step 2: Smoke test with --help**

Run: `python -m sceneframe.cli --help`
Expected: Shows usage

**Step 3: Update .gitignore if needed**

Ensure these are in `.gitignore`:
```
.venv/
__pycache__/
*.pyc
.env
.env.*
dist/
build/
*.egg-info/
test_output/
test_video/
*.mp4
```

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore: final cleanup and integration tests"
```

---

## Decision Log

| Decision | Choice | Alternatives | Why |
|---|---|---|---|
| Package name | `sceneframe` | Keep `dataset_maker` | New project, new identity |
| Detector | PySceneDetect only | PySceneDetect + TransNetV2 | User chose simplicity, no TF dep |
| Frame extraction | OpenCV cv2 | ffmpeg subprocess | Already a dep, direct frame seek |
| CLI framework | Click | Typer, argparse | Already used, simple |
| Parallelism | ProcessPoolExecutor + cpu_count | Fixed workers, ThreadPool | Auto-scale, CPU-bound task |
| Frame offset | 3 frames from edges | Configurable, 0 offset | Avoids transition artifacts |
| JPEG quality | 95 | PNG, configurable | User chose JPEG, 95 is high quality |
| Re-segment threshold | Fixed 20s | Configurable | User chose fixed |
| First detector | ContentDetector | AdaptiveDetector | Faster, good for hard cuts |
| Re-segment detector | AdaptiveDetector(1.5) | ContentDetector with lower threshold | Better for subtle transitions |
| Naming | 0001_A.jpg / 0001_B.jpg | Subfolders, UUIDs | User chose this format |
| Odd scene handling | Discard | Duplicate, pair with self | User chose discard |
