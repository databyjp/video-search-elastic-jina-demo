# Video + Multimodal Search

Multimodal search demo that indexes video scenes and blog posts into Elasticsearch using [`jina-embeddings-v5-omni-small`](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small). Each video is split into scenes; frames and audio are embedded together. A FastAPI app lets you search across both modalities with a single query.

## How it works

1. **Ingest** — `ingest.py` detects scene cuts, slices clips, embeds frames + audio (and transcripts) for each scene, then indexes everything — along with blog posts — into an Elasticsearch index (`elastic-wizard`).
2. **Search UI** — `app.py` serves a web UI that embeds your query and runs a kNN search over the index.

## Setup

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker, `ffmpeg`

```bash
# 1. Install dependencies
uv sync

# 2. Start Elasticsearch via the Elastic start-local script
# https://www.elastic.co/docs/deploy-manage/deploy/self-managed/local-development-installation-quickstart
curl -fsSL https://elastic.co/start-local | sh
```

This creates an `elastic-start-local/` directory with a `.env` file containing your credentials — the app reads it directly from there.

## Environment variables

| Variable | Used by | Source |
|---|---|---|
| `ES_LOCAL_URL` | App + ingestion | Auto-loaded from `elastic-start-local/.env` |
| `ES_LOCAL_API_KEY` | App + ingestion | Auto-loaded from `elastic-start-local/.env` |
| `JINA_API_KEY` | `example_embed_video_jina_api_url.py` | Your `.env` file |
| `GENERAL_LITELLM_API_KEY` | Voice search (transcript mode) | Any OpenAI-compatible API key |
| `LITELLM_BASE_URL` | Voice search (transcript mode) | Any OpenAI-compatible base URL (e.g. `https://api.openai.com/v1`) |

## Usage

```bash
# Ingest videos + blogs into Elasticsearch
# (place .mp4 files in data/videos/ and blog JSON files in data/blogs/ first)
uv run python ingest.py

# Run the search UI
uv run uvicorn app:app --reload
# → open http://localhost:8000
```

### Search modes

- **Text** — type a query in the search box
- **Voice (direct)** — click the mic button; audio is embedded directly with the omni model (default)
- **Voice (transcript)** — uncheck "Direct audio embedding" to transcribe with Whisper, extract a query via LLM, then search as text (requires `GENERAL_LITELLM_API_KEY` / `LITELLM_BASE_URL`)
- **Image** — upload an image to embed and search against video scenes
- **Modality filter** — use the dropdown to restrict results to transcript-only or video+audio embeddings

### Example queries

- Visual: "presenter holding a Kindle", "food recipe"
- Transcript: "bm25 explained", "how to set up a Jina model on Elastic Inference Service"

## Example scripts

| Script | What it does |
|---|---|
| `example_embed_video_local.py` | Embed a local video file using the local model and print the embedding shape + timing. |
| `example_embed_video_jina_api_url.py` | Embed a video via the Jina cloud API (by URL). Requires `JINA_API_KEY`. |
| `example_similarity.py` | Embed a video and multiple text queries locally, then compute cosine similarity and the equivalent Elasticsearch score (`(1 + cos) / 2`) for each. Useful for sanity-checking relevance outside of ES. |

Run any example with:

```bash
uv run python example_similarity.py
```

## Data

Place `.mp4` video files in `data/videos/` before running `ingest.py`. Blog posts (`.json` files in `data/blogs/`) are also indexed if present.
