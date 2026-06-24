"""Index video scenes and run a text search."""
from pathlib import Path

from omnimodal_search.es import get_client, create_index, index_videos
from omnimodal_search.embedding import get_model

VIDEO_DIR = Path("data/videos")
INDEX = "elastic-wizard"

# Connect to Elasticsearch
client = get_client()
print("ES cluster:", client.info()["cluster_name"])

# Load model to get the correct dims
model = get_model()
dims = model.get_embedding_dimension()
print(f"Model dims: {dims}")

# Reset + create index
if client.indices.exists(index=INDEX):
    client.indices.delete(index=INDEX)
    print(f"Dropped old index `{INDEX}`")

create_index(client, INDEX, dims=dims)
print(f"Created index `{INDEX}`")

# Index videos
print()
n_docs = index_videos(client, INDEX, VIDEO_DIR)
print(f"\nIndexed {n_docs} documents total")

# --- Search ---
query = "How to set up security in Elasticsearch"
print(f"\n{'='*50}")
print(f"Query: {query!r}")
print(f"{'='*50}")

qv = model.encode_query(query).tolist()

# Search fused only
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
print("\n[Search results]")
for h in resp["hits"]["hits"]:
    src = h["_source"]
    yt = f"https://www.youtube.com/watch?v={src['video_id']}&t={int(src['start_sec'])}s"
    print(f"  {h['_score']:.4f}  [{src['video_id']}]  {src['start_sec']:.1f}s–{src['end_sec']:.1f}s")
    print(f"           → {yt}")
