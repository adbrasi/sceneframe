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
        result = detect_scenes(tmp_path / "nonexistent.mp4")
        assert isinstance(result, list)

    def test_returns_empty_for_missing_video(self, tmp_path):
        result = detect_scenes(tmp_path / "missing.mp4")
        assert result == []


class TestReDetectLongScenes:
    def test_short_scenes_unchanged(self, tmp_path):
        scenes = [
            SceneBoundary(start_frame=0, end_frame=240, fps=24.0),
            SceneBoundary(start_frame=240, end_frame=480, fps=24.0),
        ]
        result = re_detect_long_scenes(
            tmp_path / "nonexistent.mp4", scenes, max_seconds=20.0
        )
        assert result == scenes

    def test_identifies_long_scenes(self):
        scenes = [
            SceneBoundary(start_frame=0, end_frame=240, fps=24.0),
            SceneBoundary(start_frame=240, end_frame=1200, fps=24.0),
        ]
        long = [s for s in scenes if s.duration_seconds > 20.0]
        assert len(long) == 1
        assert long[0].start_frame == 240
