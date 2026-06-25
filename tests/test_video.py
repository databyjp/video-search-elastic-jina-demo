"""Tests for video splitting and audio extraction."""

import pytest
import av

from omnimodal_search.video import cut_video, extract_audio

VIDEO = "data/videos/6q-DZyWD_VE.mp4"
START, END = 10.0, 20.0  # 10-second window well within the clip


def get_duration(path) -> float:
    with av.open(str(path)) as container:
        return float(container.duration) / 1_000_000  # microseconds → seconds


@pytest.fixture(autouse=True)
def require_video():
    pytest.importorskip("av")
    if not __import__("pathlib").Path(VIDEO).exists():
        pytest.skip(f"Test video not found: {VIDEO}")


def test_cut_video_creates_file(tmp_path):
    out = tmp_path / "clip.mp4"
    cut_video(VIDEO, str(out), START, END)
    assert out.exists() and out.stat().st_size > 0


def test_cut_video_duration(tmp_path):
    out = tmp_path / "clip.mp4"
    cut_video(VIDEO, str(out), START, END)
    # Stream copy snaps to keyframes, so allow up to 2s of drift
    assert abs(get_duration(out) - (END - START)) < 2.0


def test_extract_audio_creates_file(tmp_path):
    out = tmp_path / "audio.wav"
    extract_audio(VIDEO, str(out), START, END)
    assert out.exists() and out.stat().st_size > 0


def test_extract_audio_duration(tmp_path):
    out = tmp_path / "audio.wav"
    extract_audio(VIDEO, str(out), START, END)
    # Re-encoded audio is sample-accurate
    assert abs(get_duration(out) - (END - START)) < 0.1
