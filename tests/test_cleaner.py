import cv2
import numpy as np
from pathlib import Path

from sceneframe.cleaner import (
    is_solid_color,
    _compute_feature_vector,
    scan_pairs,
    find_solid_color_labels,
    find_duplicate_labels,
    find_orphan_labels,
    clean_directory,
)

JPEG_QUALITY = 95


def _save_test_image(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def _make_solid_image(color=(0, 0, 0), size=(64, 64)) -> np.ndarray:
    """Create a solid color image."""
    return np.full((*size, 3), color, dtype=np.uint8)


def _make_noisy_image(seed=42, size=(64, 64)) -> np.ndarray:
    """Create a random noisy image (not solid)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (*size, 3), dtype=np.uint8)


class TestIsSolidColor:
    def test_black_image_is_solid(self):
        img = _make_solid_image((0, 0, 0))
        assert is_solid_color(img) is True

    def test_white_image_is_solid(self):
        img = _make_solid_image((255, 255, 255))
        assert is_solid_color(img) is True

    def test_green_image_is_solid(self):
        img = _make_solid_image((0, 255, 0))
        assert is_solid_color(img) is True

    def test_noisy_image_is_not_solid(self):
        img = _make_noisy_image()
        assert is_solid_color(img) is False

    def test_nearly_solid_with_noise(self):
        """An image that is mostly one color with slight noise."""
        img = _make_solid_image((128, 128, 128))
        # Add very small noise
        rng = np.random.RandomState(0)
        noise = rng.randint(-3, 4, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        assert is_solid_color(img) is True

    def test_none_image_is_solid(self):
        assert is_solid_color(None) is True


class TestFeatureVector:
    def test_identical_images_same_vector(self):
        img = _make_noisy_image(seed=1)
        v1 = _compute_feature_vector(img)
        v2 = _compute_feature_vector(img)
        np.testing.assert_array_almost_equal(v1, v2)

    def test_different_images_different_vector(self):
        v1 = _compute_feature_vector(_make_noisy_image(seed=1))
        v2 = _compute_feature_vector(_make_noisy_image(seed=2))
        similarity = np.dot(v1, v2)
        assert similarity < 0.99  # different images should not be identical

    def test_vector_is_normalized(self):
        img = _make_noisy_image(seed=1)
        v = _compute_feature_vector(img)
        norm = np.linalg.norm(v)
        assert abs(norm - 1.0) < 1e-4

    def test_vector_length(self):
        img = _make_noisy_image()
        v = _compute_feature_vector(img)
        # 64*64 = 4096 features
        assert len(v) == 64 * 64

    def test_nearly_identical_images_high_similarity(self):
        """An image with tiny noise added should have high cosine similarity."""
        img1 = _make_noisy_image(seed=1, size=(128, 128))
        img2 = img1.copy()
        # Add very small perturbation (like an eye blink)
        img2[50:55, 50:55] = np.clip(img2[50:55, 50:55].astype(np.int16) + 10, 0, 255).astype(np.uint8)
        v1 = _compute_feature_vector(img1)
        v2 = _compute_feature_vector(img2)
        similarity = np.dot(v1, v2)
        assert similarity > 0.96  # should be caught as duplicate


class TestScanPairs:
    def test_finds_pairs(self, tmp_path):
        img = _make_noisy_image()
        _save_test_image(tmp_path / "000001_A.jpg", img)
        _save_test_image(tmp_path / "000001_B.jpg", img)
        _save_test_image(tmp_path / "000002_A.jpg", img)
        _save_test_image(tmp_path / "000002_B.jpg", img)

        pairs = scan_pairs(tmp_path)
        assert len(pairs) == 2
        assert "A" in pairs["000001"]
        assert "B" in pairs["000001"]

    def test_ignores_non_pair_files(self, tmp_path):
        img = _make_noisy_image()
        _save_test_image(tmp_path / "000001_A.jpg", img)
        _save_test_image(tmp_path / "random.jpg", img)
        (tmp_path / "readme.txt").write_text("hello")

        pairs = scan_pairs(tmp_path)
        assert len(pairs) == 1


class TestFindSolidColorLabels:
    def test_detects_solid_pairs(self, tmp_path):
        black = _make_solid_image((0, 0, 0))
        normal = _make_noisy_image()

        _save_test_image(tmp_path / "000001_A.jpg", black)
        _save_test_image(tmp_path / "000001_B.jpg", normal)
        _save_test_image(tmp_path / "000002_A.jpg", normal)
        _save_test_image(tmp_path / "000002_B.jpg", normal)

        labels = find_solid_color_labels(tmp_path, workers=1)
        assert "000001" in labels
        assert "000002" not in labels


class TestFindDuplicateLabels:
    def test_detects_duplicates(self, tmp_path):
        img1 = _make_noisy_image(seed=1)
        img2 = _make_noisy_image(seed=2)

        # Pair 1 and 3 have the same A image → duplicate
        _save_test_image(tmp_path / "000001_A.jpg", img1)
        _save_test_image(tmp_path / "000001_B.jpg", img2)
        _save_test_image(tmp_path / "000002_A.jpg", img2)
        _save_test_image(tmp_path / "000002_B.jpg", img1)
        _save_test_image(tmp_path / "000003_A.jpg", img1)  # same as 000001_A
        _save_test_image(tmp_path / "000003_B.jpg", img2)

        labels = find_duplicate_labels(tmp_path, similarity=0.96, workers=1)
        # 000003 should be marked as duplicate of 000001
        assert "000003" in labels
        assert "000001" not in labels

    def test_nearly_identical_detected(self, tmp_path):
        """Frames with tiny differences (eye blink level) should be caught."""
        img1 = _make_noisy_image(seed=1, size=(128, 128))
        img2 = img1.copy()
        # Small perturbation — like an eye blink in a small region
        img2[50:55, 50:55] = np.clip(img2[50:55, 50:55].astype(np.int16) + 10, 0, 255).astype(np.uint8)

        _save_test_image(tmp_path / "000001_A.jpg", img1)
        _save_test_image(tmp_path / "000001_B.jpg", img1)
        _save_test_image(tmp_path / "000002_A.jpg", img2)
        _save_test_image(tmp_path / "000002_B.jpg", img2)

        labels = find_duplicate_labels(tmp_path, similarity=0.96, workers=1)
        assert "000002" in labels
        assert "000001" not in labels

    def test_no_duplicates(self, tmp_path):
        for i in range(3):
            img = _make_noisy_image(seed=i * 100)
            _save_test_image(tmp_path / f"{i + 1:06d}_A.jpg", img)
            _save_test_image(tmp_path / f"{i + 1:06d}_B.jpg", img)

        labels = find_duplicate_labels(tmp_path, similarity=0.96, workers=1)
        assert len(labels) == 0


class TestFindOrphanLabels:
    def test_finds_orphans(self, tmp_path):
        img = _make_noisy_image()
        _save_test_image(tmp_path / "000001_A.jpg", img)
        # No 000001_B → orphan
        _save_test_image(tmp_path / "000002_A.jpg", img)
        _save_test_image(tmp_path / "000002_B.jpg", img)

        orphans = find_orphan_labels(tmp_path)
        assert "000001" in orphans
        assert "000002" not in orphans


class TestCleanDirectory:
    def test_full_pipeline(self, tmp_path):
        black = _make_solid_image((0, 0, 0))
        normal1 = _make_noisy_image(seed=1)
        normal2 = _make_noisy_image(seed=2)

        # Pair 1: solid color A → should be removed
        _save_test_image(tmp_path / "000001_A.jpg", black)
        _save_test_image(tmp_path / "000001_B.jpg", normal1)

        # Pair 2: normal → should survive
        _save_test_image(tmp_path / "000002_A.jpg", normal1)
        _save_test_image(tmp_path / "000002_B.jpg", normal2)

        # Pair 3: duplicate of pair 2 → should be removed
        _save_test_image(tmp_path / "000003_A.jpg", normal1)
        _save_test_image(tmp_path / "000003_B.jpg", normal2)

        stats = clean_directory(tmp_path, workers=1)

        assert stats["solid_removed"] == 1
        assert stats["duplicates_removed"] >= 1
        assert stats["remaining"] >= 1

        # Pair 1 files should be deleted
        assert not (tmp_path / "000001_A.jpg").exists()
        assert not (tmp_path / "000001_B.jpg").exists()

        # Pair 2 should remain
        assert (tmp_path / "000002_A.jpg").exists()
        assert (tmp_path / "000002_B.jpg").exists()

    def test_dry_run_does_not_delete(self, tmp_path):
        black = _make_solid_image((0, 0, 0))
        normal = _make_noisy_image()

        _save_test_image(tmp_path / "000001_A.jpg", black)
        _save_test_image(tmp_path / "000001_B.jpg", normal)

        stats = clean_directory(tmp_path, workers=1, dry_run=True)

        assert stats["solid_removed"] == 1
        # Files should still exist
        assert (tmp_path / "000001_A.jpg").exists()
        assert (tmp_path / "000001_B.jpg").exists()


class TestMaxPairs:
    """Test max_pairs parameter on extraction functions."""

    def test_intra_max_pairs(self, tmp_path):
        from sceneframe.extractor import extract_intra_scene_pairs

        video = tmp_path / "test.mp4"
        _make_test_video(video, num_frames=200)

        scenes = [
            SceneBoundary(start_frame=0, end_frame=50, fps=24.0),
            SceneBoundary(start_frame=50, end_frame=100, fps=24.0),
            SceneBoundary(start_frame=100, end_frame=150, fps=24.0),
        ]

        output = tmp_path / "output"
        count = extract_intra_scene_pairs(video, scenes, output, max_pairs=2)
        assert count == 2

    def test_inter_seq_max_pairs(self, tmp_path):
        from sceneframe.extractor import extract_inter_scene_pairs_sequential

        video = tmp_path / "test.mp4"
        _make_test_video(video, num_frames=400)

        scenes = [
            SceneBoundary(start_frame=i * 50, end_frame=(i + 1) * 50, fps=24.0)
            for i in range(8)
        ]

        output = tmp_path / "output"
        count = extract_inter_scene_pairs_sequential(video, scenes, output, max_pairs=1)
        assert count == 1

    def test_inter_slide_max_pairs(self, tmp_path):
        from sceneframe.extractor import extract_inter_scene_pairs_sliding

        video = tmp_path / "test.mp4"
        _make_test_video(video, num_frames=400)

        scenes = [
            SceneBoundary(start_frame=i * 50, end_frame=(i + 1) * 50, fps=24.0)
            for i in range(8)
        ]

        output = tmp_path / "output"
        count = extract_inter_scene_pairs_sliding(video, scenes, output, max_pairs=2)
        assert count == 2


def _make_test_video(path: Path, num_frames: int = 100, fps: float = 24.0):
    """Create a minimal test video."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (64, 64))
    for i in range(num_frames):
        frame = np.full((64, 64, 3), fill_value=i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


# Re-import for max_pairs tests
from sceneframe.detector import SceneBoundary
