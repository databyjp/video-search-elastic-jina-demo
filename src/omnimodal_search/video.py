import subprocess
from pathlib import Path
from scenedetect import detect, ContentDetector


def find_scenes(video_path: str):
    return detect(video_path, ContentDetector())


def cut_video(video_path: str, out_path: str, start_sec: float, end_sec: float):
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
        "-ss", str(start_sec), "-to", str(end_sec),
        "-c", "copy", out_path],
        check=True, capture_output=True,
    )


def extract_audio(video_path: str, out_wav: str, start_sec: float, end_sec: float):
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
        "-ss", str(start_sec), "-to", str(end_sec),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        out_wav],
        check=True, capture_output=True,
    )
