import subprocess
from pathlib import Path
from scenedetect import detect, ContentDetector


def find_scenes(video_path: str):
    return detect(video_path, ContentDetector())


def cut_video(video_path: str, out_path: str, start_sec: float, end_sec: float):
    """Extract a sub-clip. -ss before -i for keyframe-safe seeking with -c copy."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start_sec),   # fast seek BEFORE input
            "-i", video_path,
            "-to", str(end_sec),     # absolute stop time in the source
            "-c", "copy",
            out_path,
        ],
        check=True, capture_output=True,
    )


def extract_audio(video_path: str, out_wav: str, start_sec: float, end_sec: float):
    """Extract audio track as 16 kHz mono WAV."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start_sec),   # keep in sync with video cut
            "-i", video_path,
            "-to", str(end_sec),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            out_wav,
        ],
        check=True, capture_output=True,
    )
