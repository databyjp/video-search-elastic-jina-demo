"""Ingest video scenes into Elasticsearch."""

from pathlib import Path

from omnimodal_search.es import get_client, create_index, index_videos
from omnimodal_search.embedding import get_model

VIDEO_DIR = Path("data/videos")
INDEX = "elastic-wizard"

client = get_client()
print("ES cluster:", client.info()["cluster_name"])

model = get_model()
dims = model.get_embedding_dimension()
print(f"Model dims: {dims}")

# Reset index
if client.indices.exists(index=INDEX):
    client.indices.delete(index=INDEX)
    print(f"Dropped old index `{INDEX}`")

create_index(client, INDEX, dims=dims)
print(f"Created index `{INDEX}`\n")

# Index videos
n_docs = index_videos(client, INDEX, VIDEO_DIR)
print(f"\nIndexed {n_docs} documents total")
