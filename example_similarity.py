"""Embed a video and a text query, then compute cosine similarity."""

import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from omnimodal_search.embedding import get_model

load_dotenv()

# ---------------------------------------------------------------------------
# Config — edit these
# ---------------------------------------------------------------------------
VIDEO_PATH = Path("data/clips/6q-DZyWD_VE_s0.mp4")  # Jen holding a Kindle
# VIDEO_PATH = Path("data/clips/AurP_IjYcvw_s6.mp4")  # Cake recipe
TEXT_QUERIES = [
    "Presenter with kindle",
    "A cake recipe tutorial",
    "Picture of cats",
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    print("Loading model (not timed) ...")
    model = get_model()

    # --- Embed the video (as a document) ---
    print(f"\n🎬 Embedding video: {VIDEO_PATH}")
    t0 = time.perf_counter()
    video_emb = model.encode_document(str(VIDEO_PATH))
    t_video = time.perf_counter() - t0
    print(f"   ⏱️  {t_video:.4f}s  |  shape: {video_emb.shape}")

    # --- Embed each text query and compute similarity ---
    print()
    for query in TEXT_QUERIES:
        t0 = time.perf_counter()
        text_emb = model.encode_query(query)
        t_text = time.perf_counter() - t0
        cos = cosine_similarity(video_emb, text_emb)
        es_score = (1 + cos) / 2
        print(f"📝 \"{query}\"")
        print(f"   ⏱️  {t_text:.4f}s  |  🎯 Similarity score: {es_score:.4f}\n")


if __name__ == "__main__":
    main()
