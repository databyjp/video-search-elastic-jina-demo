import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()

JINA_URL = "https://api.jina.ai/v1/embeddings"
DEFAULT_MODEL = "jina-embeddings-v5-omni-small"

# ---------------------------------------------------------------------------
# Config — edit this
# ---------------------------------------------------------------------------
# VIDEO_URL = "https://databyjp.s3.us-west-2.amazonaws.com/temp/AurP_IjYcvw_s1.mp4"
VIDEO_URL = "https://databyjp.s3.us-west-2.amazonaws.com/temp/AurP_IjYcvw_s6.mp4"


def embed_video_jina(url: str, model: str = DEFAULT_MODEL) -> dict:
    """Call the Jina API to embed a video URL. Returns the JSON response."""
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        print(
            "❌ JINA_API_KEY not found in environment. Set it in .env or export it.",
            file=sys.stderr,
        )
        sys.exit(1)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = {
        "model": model,
        "task": "retrieval.passage",
        "normalized": True,
        "input": [{"video": url}],
    }

    print(f"Calling Jina API for embedding ...\n  URL: {url}")
    start = time.perf_counter()
    response = requests.post(JINA_URL, headers=headers, json=data, timeout=300)
    elapsed = time.perf_counter() - start

    if not response.ok:
        print(f"❌ Jina API error {response.status_code}: {response.text}", file=sys.stderr)
        sys.exit(1)

    result = response.json()
    embedding = result["data"][0]["embedding"]

    print(f"\n⏱️  Jina API embedding time: {elapsed:.4f}s")
    print(f"📐 Embedding length: {len(embedding)}")
    print(f"🔢 First 5 dims: {embedding[:5]}")

    return result


if __name__ == "__main__":
    embed_video_jina(VIDEO_URL)
