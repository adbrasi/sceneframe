import cv2
import numpy as np
from pathlib import Path

from sceneframe.detector import SceneBoundary
from sceneframe.extractor import (
    extract_frame,
    extract_intra_scene_pairs,
    extract_inter_scene_pairs_sequential,
    extract_inter_scene_pairs_sliding,
    _safe_frame_indices,
    FRAME_OFFSET,
)


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


class TestSafeFrameIndices:
    def test_returns_offset_indices(self):
        scene = SceneBoundary(start_frame=0, end_frame=50, fps=24.0)
        result = _safe_frame_indices(scene)
        assert result == (3, 46)  # start+3, end-1-3

    def test_returns_none_for_too_short_scene(self):
        scene = SceneBoundary(start_frame=0, end_frame=6, fps=24.0)
        result = _safe_frame_indices(scene)
        assert result is None

    def test_returns_none_when_start_equals_end(self):
        # offset*2 + 1 = 7, so 7 frames is the minimum
        scene = SceneBoundary(start_frame=0, end_frame=7, fps=24.0)
        result = _safe_frame_indices(scene)
        assert result is None

    def test_minimum_valid_scene(self):
        # 8 frames: start+3=3, end-1-3=4, different frames
        scene = SceneBoundary(start_frame=0, end_frame=8, fps=24.0)
        result = _safe_frame_indices(scene)
        assert result == (3, 4)

    def test_end_frame_is_exclusive(self):
        """end_frame is the first frame of the next scene (exclusive).
        The last frame of this scene is end_frame - 1.
        With offset=3, we read end_frame - 1 - 3."""
        scene = SceneBoundary(start_frame=100, end_frame=200, fps=24.0)
        result = _safe_frame_indices(scene)
        assert result == (103, 196)


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
        assert (output / "000001_A.jpg").exists()
        assert (output / "000001_B.jpg").exists()
        assert (output / "000002_A.jpg").exists()
        assert (output / "000002_B.jpg").exists()

    def test_skips_scenes_too_short_for_offset(self, tmp_path):
        video = tmp_path / "test.mp4"
        _make_test_video(video, num_frames=200)

        scenes = [
            SceneBoundary(start_frame=0, end_frame=5, fps=24.0),
            SceneBoundary(start_frame=50, end_frame=100, fps=24.0),
        ]

        output = tmp_path / "output_intra"
        count = extract_intra_scene_pairs(video, scenes, output)
        assert count == 1

    def test_empty_scenes_returns_zero(self, tmp_path):
        video = tmp_path / "test.mp4"
        _make_test_video(video, num_frames=50)

        output = tmp_path / "output_intra"
        count = extract_intra_scene_pairs(video, [], output)
        assert count == 0


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

        assert count == 2
        assert (output / "000001_A.jpg").exists()
        assert (output / "000001_B.jpg").exists()
        assert (output / "000002_A.jpg").exists()
        assert (output / "000002_B.jpg").exists()

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
        assert count == 1

    def test_empty_scenes_returns_zero(self, tmp_path):
        video = tmp_path / "test.mp4"
        _make_test_video(video, num_frames=50)

        output = tmp_path / "output_inter_seq"
        count = extract_inter_scene_pairs_sequential(video, [], output)
        assert count == 0

    def test_single_scene_returns_zero(self, tmp_path):
        video = tmp_path / "test.mp4"
        _make_test_video(video, num_frames=50)

        scenes = [SceneBoundary(start_frame=0, end_frame=50, fps=24.0)]

        output = tmp_path / "output_inter_seq"
        count = extract_inter_scene_pairs_sequential(video, scenes, output)
        assert count == 0

    def test_skips_short_scenes(self, tmp_path):
        video = tmp_path / "test.mp4"
        _make_test_video(video, num_frames=200)

        scenes = [
            SceneBoundary(start_frame=0, end_frame=2, fps=24.0),   # too short
            SceneBoundary(start_frame=2, end_frame=50, fps=24.0),  # ok but paired with short
            SceneBoundary(start_frame=50, end_frame=100, fps=24.0),
            SceneBoundary(start_frame=100, end_frame=150, fps=24.0),
        ]

        output = tmp_path / "output_inter_seq"
        count = extract_inter_scene_pairs_sequential(video, scenes, output)
        # Pair (0,1): scene 0 is too short -> skipped
        # Pair (2,3): both ok -> 1 pair
        assert count == 1


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

        assert count == 2
        assert (output / "000001_A.jpg").exists()
        assert (output / "000001_B.jpg").exists()
        assert (output / "000002_A.jpg").exists()
        assert (output / "000002_B.jpg").exists()

    def test_single_scene_returns_zero(self, tmp_path):
        video = tmp_path / "test.mp4"
        _make_test_video(video, num_frames=50)

        scenes = [SceneBoundary(start_frame=0, end_frame=50, fps=24.0)]

        output = tmp_path / "output_inter_slide"
        count = extract_inter_scene_pairs_sliding(video, scenes, output)
        assert count == 0

    def test_empty_scenes_returns_zero(self, tmp_path):
        video = tmp_path / "test.mp4"
        _make_test_video(video, num_frames=50)

        output = tmp_path / "output_inter_slide"
        count = extract_inter_scene_pairs_sliding(video, [], output)
        assert count == 0
