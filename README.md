# Omnimodal Search

Multimodal search demo that indexes video scenes and blog posts into Elasticsearch using [`jina-embeddings-v5-omni-small`](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small). Each video is split into scenes; frames and audio are embedded together. A FastAPI app lets you search across both modalities with a single query.

## How it works

1. **Ingest** — `ingest.py` detects scene cuts, slices clips, embeds frames + audio (and transcripts) for each scene, then indexes everything — along with blog posts — into an Elasticsearch index (`elastic-wizard`).
2. **Search UI** — `app.py` serves a web UI that embeds your query and runs a kNN search over the index.

## Setup

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker, `ffmpeg`

```bash
# 1. Install dependencies
uv sync

# 2. Start Elasticsearch (Docker)
cd elastic-start-local && ./start.sh && cd ..

# 3. Configure connection (copy .env from elastic-start-local output)
cp elastic-start-local/.env .env
```

## Usage

```bash
# Ingest videos + blogs into Elasticsearch
# (place .mp4 files in data/videos/ and blog JSON files in data/blogs/ first)
uv run python ingest.py

# Run the search UI
uv run uvicorn app:app --reload
# → open http://localhost:8000
```

Place blog posts as text/markdown files in `data/blogs/` before running `ingest.py`.
