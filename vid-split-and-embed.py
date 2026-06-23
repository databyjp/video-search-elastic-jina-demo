"""
Scene detection, slicing, and multimodal embedding with jina-embeddings-v5-omni-small.

Prerequisites:
    pip install scenedetect sentence-transformers av
    ffmpeg must be installed on your system

Usage:
    python vector-search-howto-omni-video-scene-embed.py <path_to_video>
"""

import os
import tempfile
import subprocess
from pathlib import Path

from scenedetect import detect, ContentDetector
from sentence_transformers import SentenceTransformer


# --- 1. Load model ---------------------------------------------------------
model = SentenceTransformer(
    "jinaai/jina-embeddings-v5-omni-small-retrieval",
    trust_remote_code=True,
)
print(f"Model loaded — vector size: {model.get_embedding_dimension()}")


# --- 2. Helpers ------------------------------------------------------------
def extract_audio(video_path, out_wav, start_sec=None, end_sec=None):
    """Extract audio track as 16 kHz mono WAV."""
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1"]
    if start_sec is not None:
        cmd += ["-ss", str(start_sec)]
    if end_sec is not None:
        cmd += ["-to", str(end_sec)]
    cmd.append(out_wav)
    subprocess.run(cmd, check=True, capture_output=True)


def cut_video(video_path, out_path, start_sec, end_sec):
    """Extract a sub-clip (no re-encode)."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-ss", str(start_sec),
            "-to", str(end_sec),
            "-c", "copy",
            out_path,
        ],
        check=True,
        capture_output=True,
    )


# --- 3. Process ------------------------------------------------------------
video_path = "/Users/jphwang/code/content/202606-omnimodal-search-live/data/videos/6q-DZyWD_VE.mp4"
video_name = Path(video_path).stem
print(f"\n🔍 Detecting scenes in: {video_path}")

scene_list = detect(video_path, ContentDetector())
print(f"Found {len(scene_list)} scene(s)")

with tempfile.TemporaryDirectory() as tmpdir:
    for i, scene in enumerate(scene_list):
        start_sec = scene[0].seconds
        end_sec = scene[1].seconds
        duration = end_sec - start_sec

        if duration < 1.0:
            print(f"  Skipping scene {i+1} (too short: {duration:.2f}s)")
            continue

        print(f"\n  Scene {i+1}: {start_sec:.2f}s – {end_sec:.2f}s  ({duration:.2f}s)")

        scene_video = os.path.join(tmpdir, f"{video_name}_scene_{i}.mp4")
        scene_audio = os.path.join(tmpdir, f"{video_name}_scene_{i}.wav")

        # Cut the scene and extract its audio
        cut_video(video_path, scene_video, start_sec, end_sec)
        extract_audio(video_path, scene_audio, start_sec, end_sec)

        # --- Embed video + audio (fused) ---------------------------------
        print("    Embedding video ...")
        emb_video = model.encode_document(scene_video)

        print("    Embedding audio ...")
        emb_audio = model.encode_document(scene_audio)

        print("    Fusing video + audio ...")
        emb_fused = model.encode_document((scene_video, scene_audio))

        print(f"    Video  vector: {emb_video.shape}  (first 5: {emb_video.tolist()[:5]})")
        print(f"    Audio  vector: {emb_audio.shape}  (first 5: {emb_audio.tolist()[:5]})")
        print(f"    Fused  vector: {emb_fused.shape}  (first 5: {emb_fused.tolist()[:5]})")

print("\n✅ Done.")
