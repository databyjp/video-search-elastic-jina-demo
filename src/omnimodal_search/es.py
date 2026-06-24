from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from tqdm import tqdm
import json
import os
import tempfile
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", message=".*Qwen3VL requires frame timestamps.*")

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
                    "similarity": "cosine",
                },
                "modality": {"type": "keyword"},  # 'fused', 'text', 'video', etc.
                "video_id": {"type": "keyword"},
                "scene_index": {"type": "integer"},
                "start_sec": {"type": "float"},
                "end_sec": {"type": "float"},
                "clip_duration": {"type": "float"},
                "source_path": {"type": "keyword"},
            }
        },
    )


def _scene_cache(video_path: Path) -> Path:
    return video_path.with_suffix(".scenes.json")


def _video_exists(client: Elasticsearch, index: str, video_id: str, video_path: Path) -> bool:
    """Return True if all expected scenes for this video are already indexed."""
    cache = _scene_cache(video_path)
    if not cache.exists():
        return False
    expected = len(json.loads(cache.read_text()))
    resp = client.search(index=index, query={"term": {"video_id": video_id}}, size=0)
    return resp["hits"]["total"]["value"] == expected


def _get_scenes(video_path: Path) -> list[tuple[float, float]]:
    """Return (start_sec, end_sec) pairs for all usable scenes, using a JSON cache."""
    cache = _scene_cache(video_path)
    if cache.exists():
        return [tuple(s) for s in json.loads(cache.read_text())]

    scene_list = find_scenes(str(video_path))
    scenes = [
        (s[0].get_seconds(), s[1].get_seconds())
        for s in scene_list
        if s[1].get_seconds() - s[0].get_seconds() >= 1.0
    ]
    cache.write_text(json.dumps(scenes))
    return scenes


def index_videos(
    client: Elasticsearch,
    index: str,
    video_dir: str | Path,
) -> int:
    """Index all video scenes from a directory. Returns number of documents indexed."""
    model = get_model()
    video_files = sorted(Path(video_dir).glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No .mp4 files found in {video_dir}")

    count = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        for video_path in video_files:
            video_id = video_path.stem

            cached = _scene_cache(video_path).exists()
            print(f"🔍 {video_id} — {'loading scenes from cache' if cached else 'detecting scenes'} ...")
            scenes = _get_scenes(video_path)
            print(f"  Found {len(scenes)} scene(s)")

            for i, (start_sec, end_sec) in enumerate(pbar := tqdm(scenes, desc=f"  {video_id}", unit="scene")):
                duration = end_sec - start_sec

                scene_video = os.path.join(tmpdir, f"{video_id}_s{i}.mp4")
                scene_audio = os.path.join(tmpdir, f"{video_id}_s{i}.wav")

                pbar.set_postfix(step="cutting")
                cut_video(str(video_path), scene_video, start_sec, end_sec)
                extract_audio(str(video_path), scene_audio, start_sec, end_sec)

                pbar.set_postfix(step="embedding")
                embedding = model.encode_document((scene_video, scene_audio)).tolist()

                pbar.set_postfix(step="indexing")
                client.index(
                    index=index,
                    document={
                        "content_type": "video",
                        "video_id": video_id,
                        "scene_index": i,
                        "start_sec": round(start_sec, 2),
                        "end_sec": round(end_sec, 2),
                        "clip_duration": round(duration, 2),
                        "source_path": str(video_path),
                        "modality": "fused",
                        "title": f"Fused scene {i} from {video_id}",
                        "embedding": embedding,
                    },
                )
                count += 1

    client.indices.refresh(index=index)
    return count
