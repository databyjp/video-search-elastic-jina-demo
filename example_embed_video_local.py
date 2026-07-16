"""Embed a local video with the local jina-omni model and time just the embedding."""

import time
from pathlib import Path

from dotenv import load_dotenv

from omnimodal_search.embedding import get_model

load_dotenv()

# ---------------------------------------------------------------------------
# Config — edit these paths
# ---------------------------------------------------------------------------
VIDEO_PATH = Path("data/clips/AurP_IjYcvw_s6.mp4")


def main():
    # Load the model first so we don't time model loading
    print("Loading model (this is NOT timed) ...")
    model = get_model()

    # Prepare input
    input_data = str(VIDEO_PATH)
    print(f"Embedding video: {VIDEO_PATH}")

    # Time JUST the embedding
    print("Running encode_document() ...")
    start = time.perf_counter()
    embedding = model.encode_document(input_data)
    elapsed = time.perf_counter() - start

    print(f"\n⏱️  Embedding time: {elapsed:.4f}s")
    print(f"📐 Embedding shape: {embedding.shape}")
    print(f"🔢 First 5 dims: {embedding[:5].tolist()}")


if __name__ == "__main__":
    main()
