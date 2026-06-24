"""Search indexed video scenes with a text query."""

from omnimodal_search.es import get_client
from omnimodal_search.embedding import get_model

INDEX = "elastic-wizard"
QUERY = "How to set up security in Elasticsearch"

client = get_client()
model = get_model()

print(f"Query: {QUERY!r}\n")

qv = model.encode_query(QUERY).tolist()

resp = client.search(
    index=INDEX,
    knn={
        "field": "embedding",
        "query_vector": qv,
        "k": 5,
        "num_candidates": 50,
        # "filter": {"term": {"modality": "fused"}},
    },
)

for h in resp["hits"]["hits"]:
    src = h["_source"]
    yt = f"https://www.youtube.com/watch?v={src['video_id']}&t={int(src['start_sec'])}s"
    print(f"  {h['_score']:.4f}  [{src['video_id']}]  {src['start_sec']:.1f}s–{src['end_sec']:.1f}s")
    print(f"           → {yt}")
