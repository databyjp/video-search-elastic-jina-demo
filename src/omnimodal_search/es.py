from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os
import tempfile
from pathlib import Path

from omnimodal_search.video import find_scenes, cut_video, extract_audio
from omnimodal_search.embedding import get_model

load_dotenv("elastic-start-local/.env")


def get_client() -> Elasticsearch:
    return Elasticsearch(
        os.getenv("ES_LOCAL_URL"),
        api_key=os.getenv("ES_LOCAL_API_KEY"),
    )


def create_index(client: Elasticsearch, name: str, dims: int):
    return client.indices.create(
        index=name,
        mappings={
            "properties": {
                "content_type": {"type": "keyword"},
                "title": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine"
                },
                "modality": {"type": "keyword"},  # 'fused', 'text', 'video', etc.
                "video_id": {"type": "keyword"},
                "scene_index": {"type": "integer"},
                "start_sec": {"type": "float"},
                "end_sec": {"type": "float"},
                "clip_duration": {"type": "float"},
                "source_path": {"type": "keyword"},
            }
        }
    )


def index_videos(client: Elasticsearch, index: str, video_dir: str | Path) -> int:
    """Index all video scenes from a directory. Returns number of documents indexed."""
    model = get_model()
    video_files = sorted(Path(video_dir).glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No .mp4 files found in {video_dir}")

    count = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        for video_path in video_files:
            video_id = video_path.stem
            print(f"🔍 {video_id} — detecting scenes ...")
            scene_list = find_scenes(str(video_path))
            print(f"  Found {len(scene_list)} scene(s)")

            for i, scene in enumerate(scene_list):
                start_sec = scene[0].get_seconds()
                end_sec = scene[1].get_seconds()
                duration = end_sec - start_sec
                if duration < 1.0:
                    continue

                scene_video = os.path.join(tmpdir, f"{video_id}_s{i}.mp4")
                scene_audio = os.path.join(tmpdir, f"{video_id}_s{i}.wav")

                cut_video(str(video_path), scene_video, start_sec, end_sec)
                extract_audio(str(video_path), scene_audio, start_sec, end_sec)

                base = {
                    "content_type": "video",
                    "video_id": video_id,
                    "scene_index": i,
                    "start_sec": round(start_sec, 2),
                    "end_sec": round(end_sec, 2),
                    "clip_duration": round(duration, 2),
                    "source_path": str(video_path),
                }

                client.index(index=index, document={
                    **base,
                    "modality": "fused",
                    "title": f"Fused scene {i} from {video_id}",
                    "embedding": model.encode_document((scene_video, scene_audio)).tolist(), # Fused embedding
                })
                count += 1

    client.indices.refresh(index=index)
    return count
