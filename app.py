"""Local search UI — run with: uvicorn app:app --reload"""

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from omnimodal_search.es import get_client
from omnimodal_search.embedding import get_model

INDEX = "elastic-wizard"

app = FastAPI()
app.mount("/videos", StaticFiles(directory="data/videos"), name="videos")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

client = get_client()
model = get_model()


def fmt_ts(seconds: float) -> str:
    """Format seconds as M:SS.s — e.g. 321.1 → 5:21.1"""
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:04.1f}"


templates.env.filters["ts"] = fmt_ts


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    qv = model.encode_query(query).tolist()
    resp = client.search(
        index=INDEX,
        knn={
            "field": "embedding",
            "query_vector": qv,
            "k": 20,
            "num_candidates": 100,
            "filter": {"term": {"content_type": "video"}},
        },
    )

    # Deduplicate by scene, keeping the highest-scoring modality per scene
    seen = {}
    for h in resp["hits"]["hits"]:
        key = (h["_source"]["video_id"], h["_source"]["scene_index"])
        if key not in seen or h["_score"] > seen[key]["score"]:
            seen[key] = h["_source"] | {"score": h["_score"]}

    hits = list(seen.values())[:5]
    return templates.TemplateResponse(request, "results.html", {"hits": hits})
