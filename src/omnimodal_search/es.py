from elasticsearch import Elasticsearch, NotFoundError
from dotenv import load_dotenv
from tqdm import tqdm
import hashlib
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
                # video fields
                "video_id": {"type": "keyword"},
                "scene_index": {"type": "integer"},
                "start_sec": {"type": "float"},
                "end_sec": {"type": "float"},
                "clip_duration": {"type": "float"},
                "source_path": {"type": "keyword"},
                # blog fields
                "slug": {"type": "keyword"},
                "url": {"type": "keyword"},
                "description": {"type": "text"},
                "published_at": {"type": "date", "format": "EEE, dd MMM yyyy HH:mm:ss z"},
                "categories": {"type": "keyword"},
                "hash": {"type": "keyword"},
            }
        },
    )


def _scene_cache(video_path: Path) -> Path:
    return video_path.with_suffix(".scenes.json")


def _video_hash(video_path: Path) -> str:
    """Fast fingerprint: SHA-256 of file size + first 64 KB of content."""
    h = hashlib.sha256()
    h.update(str(video_path.stat().st_size).encode())
    with video_path.open("rb") as f:
        h.update(f.read(64 * 1024))
    return h.hexdigest()


def _is_fully_indexed(client: Elasticsearch, index: str, video_id: str, video_path: Path) -> bool:
    """Return True if the video is indexed and the source file hasn't changed."""
    cache = _scene_cache(video_path)
    if not cache.exists():
        return False
    data = json.loads(cache.read_text())
    if data.get("hash") != _video_hash(video_path):
        return False
    resp = client.search(index=index, query={"term": {"video_id": video_id}}, size=0)
    return resp["hits"]["total"]["value"] == len(data["scenes"])


def _get_scenes(video_path: Path) -> list[tuple[float, float]]:
    """Return (start_sec, end_sec) pairs for all usable scenes, using a JSON cache."""
    cache = _scene_cache(video_path)
    current_hash = _video_hash(video_path)
    if cache.exists():
        data = json.loads(cache.read_text())
        if data.get("hash") == current_hash:
            return [tuple(s) for s in data["scenes"]]

    scene_list = find_scenes(str(video_path))
    scenes = [
        (s[0].get_seconds(), s[1].get_seconds())
        for s in scene_list
        if s[1].get_seconds() - s[0].get_seconds() >= 1.0
    ]
    cache.write_text(json.dumps({"hash": current_hash, "scenes": scenes}))
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

            if _is_fully_indexed(client, index, video_id, video_path):
                print(f"⏭️  {video_id} — up to date, skipping")
                continue

            print(f"🔍 {video_id} — detecting scenes ...")
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


def _blog_hash(blog_path: Path) -> str:
    """Full content hash of the blog JSON file."""
    return hashlib.sha256(blog_path.read_bytes()).hexdigest()


def _blog_is_indexed(client: Elasticsearch, index: str, slug: str, current_hash: str) -> bool:
    """Return True if this blog is already indexed with the current file content."""
    try:
        doc = client.get(index=index, id=slug)
        return doc["_source"].get("hash") == current_hash
    except NotFoundError:
        return False


def index_blogs(
    client: Elasticsearch,
    index: str,
    blog_dir: str | Path,
) -> int:
    """Index blog posts from a directory of JSON files. Returns number of documents indexed."""
    model = get_model()
    blog_files = sorted(Path(blog_dir).glob("*.json"))
    if not blog_files:
        raise FileNotFoundError(f"No .json files found in {blog_dir}")

    count = 0
    for blog_path in tqdm(blog_files, desc="Blogs", unit="blog"):
        blog = json.loads(blog_path.read_text())
        current_hash = _blog_hash(blog_path)

        if _blog_is_indexed(client, index, blog["slug"], current_hash):
            continue

        text = f"{blog['title']}\n\n{blog['text']}"
        embedding = model.encode_document(text).tolist()

        client.index(
            index=index,
            id=blog["slug"],
            document={
                "content_type": "blog",
                "title": blog["title"],
                "slug": blog["slug"],
                "url": blog["url"],
                "description": blog.get("description"),
                "published_at": blog.get("published_at"),
                "categories": blog.get("categories", []),
                "modality": "text",
                "hash": current_hash,
                "embedding": embedding,
            },
        )
        count += 1

    client.indices.refresh(index=index)
    return count
