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
            SceneBoundary(start_frame=0, end_frame=5, fps=24.0),
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

        assert count == 2
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
        assert (output / "0001_A.jpg").exists()
        assert (output / "0001_B.jpg").exists()
        assert (output / "0002_A.jpg").exists()
        assert (output / "0002_B.jpg").exists()
