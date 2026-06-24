import subprocess
from scenedetect import detect, ContentDetector


def find_scenes(video_path: str):
    return detect(video_path, ContentDetector())


def cut_video(video_path: str, out_path: str, start_sec: float, end_sec: float):
    """Extract a sub-clip"""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_sec),  # fast seek BEFORE input
            "-i",
            video_path,
            "-t",
            str(
                end_sec - start_sec
            ),  # duration (output timestamps start at 0 after fast seek)
            "-c",
            "copy",
            out_path,
        ],
        check=True,
        capture_output=True,
    )


def extract_audio(video_path: str, out_wav: str, start_sec: float, end_sec: float):
    """Extract audio track as 16 kHz mono WAV."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_sec),  # keep in sync with video cut
            "-i",
            video_path,
            "-t",
            str(
                end_sec - start_sec
            ),  # duration (output timestamps start at 0 after fast seek)
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            out_wav,
        ],
        check=True,
        capture_output=True,
    )
