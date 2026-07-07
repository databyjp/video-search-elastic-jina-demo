"""Local search UI — run with: uvicorn app:app --reload"""

import os
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import AsyncOpenAI

from omnimodal_search.es import get_client
from omnimodal_search.embedding import get_model
from omnimodal_search.video import transcribe_audio

INDEX = "elastic-wizard"

Path("data/clips").mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.mount("/videos", StaticFiles(directory="data/videos"), name="videos")
app.mount("/clips", StaticFiles(directory="data/clips"), name="clips")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

client = get_client()
model = get_model()
openai = AsyncOpenAI(
    api_key=os.getenv("GENERAL_LITELLM_API_KEY"),
    base_url=os.getenv("LITELLM_BASE_URL")
)


def fmt_ts(seconds: float) -> str:
    """Format seconds as M:SS.s — e.g. 321.1 → 5:21.1"""
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:04.1f}"


templates.env.filters["ts"] = fmt_ts


def _search_hits(query_vector: list[float], modality_filter: str | None = None) -> list[dict]:
    """kNN search, deduplicated to one result per scene."""
    es_term = {"term": {"content_type": "video"}}
    if modality_filter and modality_filter != "both":
        internal = "fused" if modality_filter == "video_plus_audio" else modality_filter
        es_filter = {"bool": {"must": [es_term, {"term": {"modality": internal}}]}}
    else:
        es_filter = es_term
    resp = client.search(
        index=INDEX,
        knn={
            "field": "embedding",
            "query_vector": query_vector,
            "k": 20,
            "num_candidates": 100,
            "filter": es_filter,
        },
    )
    seen = {}
    for h in resp["hits"]["hits"]:
        key = (h["_source"]["video_id"], h["_source"]["scene_index"])
        if key not in seen or h["_score"] > seen[key]["score"]:
            seen[key] = h["_source"] | {"score": h["_score"]}
    hits = list(seen.values())[:5]
    video_ids = {h["video_id"] for h in hits}
    scene_counts = _get_scene_counts(video_ids)
    for h in hits:
        h["scene_count"] = scene_counts.get(h["video_id"], 0)
    return hits


def _get_scene_counts(video_ids: set[str]) -> dict[str, int]:
    """Return number of indexed scenes per video (scene_index is 0-based)."""
    resp = client.search(
        index=INDEX,
        query={"terms": {"video_id": list(video_ids)}},
        aggs={
            "by_video": {
                "terms": {"field": "video_id", "size": 100},
                "aggs": {"max_index": {"max": {"field": "scene_index"}}},
            }
        },
        size=0,
    )
    return {
        b["key"]: int(b["max_index"]["value"]) + 1
        for b in resp["aggregations"]["by_video"]["buckets"]
    }


async def extract_query(transcript: str) -> str:
    """Use an LLM to extract a concise search query from a spoken transcript."""
    resp = await openai.chat.completions.create(
        model="llm-gateway/gpt-5.4-nano",
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract a concise search query from the user's spoken request. "
                    "Return only the search terms, nothing else."
                ),
            },
            {"role": "user", "content": transcript},
        ],
        max_tokens=50,
    )
    return resp.choices[0].message.content.strip()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...), modality: str = Form("both")):
    hits = _search_hits(model.encode_query(query).tolist(), modality_filter=modality)
    return templates.TemplateResponse(request, "results.html", {"hits": hits})


@app.post("/voice-search")
async def voice_search(request: Request, audio: UploadFile = File(...), modality: str = Form("both")):
    # 1. Save upload and transcribe
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        transcript = transcribe_audio(tmp_path)
    finally:
        os.unlink(tmp_path)

    # 2. Extract clean query from conversational speech
    query = await extract_query(transcript)

    # 3. Search and render results HTML
    hits = _search_hits(model.encode_query(query).tolist(), modality_filter=modality)
    html = templates.env.get_template("results.html").render(hits=hits)
    return JSONResponse({"transcript": transcript, "query": query, "html": html})


@app.post("/voice-search-direct")
async def voice_search_direct(audio: UploadFile = File(...), modality: str = Form("both")):
    """Embed the audio directly — no transcription or LLM step."""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(await audio.read())
        webm_path = tmp.name
    wav_path = webm_path.replace(".webm", ".wav")
    try:
        # Convert to wav — model treats .webm as video (no video stream → crash)
        subprocess.run(
            ["ffmpeg", "-y", "-i", webm_path, "-ar", "16000", "-ac", "1", wav_path],
            check=True, capture_output=True,
        )
        query_vector = model.encode_query(wav_path).tolist()
    finally:
        os.unlink(webm_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)

    hits = _search_hits(query_vector, modality_filter=modality)
    html = templates.env.get_template("results.html").render(hits=hits)
    return JSONResponse({"query": "(direct audio embedding)", "html": html})
