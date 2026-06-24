"""Ingest video scenes into Elasticsearch."""

from pathlib import Path

from omnimodal_search.es import get_client, create_index, index_videos, index_blogs
from omnimodal_search.embedding import get_model

VIDEO_DIR = Path("data/videos")
BLOG_DIR = Path("data/blogs")
INDEX = "elastic-wizard"

client = get_client()
print("ES cluster:", client.info()["cluster_name"])

model = get_model()
dims = model.get_embedding_dimension()
print(f"Model dims: {dims}")

# Reset index
if client.indices.exists(index=INDEX):
    delete_index = input("Index exists. Delete and start again? (y for yes): ")
    if delete_index.lower() == "y":
        client.indices.delete(index=INDEX)
        print(f"Dropped old index `{INDEX}`")

if not client.indices.exists(index=INDEX):
    create_index(client, INDEX, dims=dims)
    print(f"Created index `{INDEX}`\n")

# Index videos
v_docs = index_videos(client, INDEX, VIDEO_DIR)
print(f"\nIndexed {v_docs} video documents")

# Index blogs
b_docs = index_blogs(client, INDEX, BLOG_DIR)
print(f"Indexed {b_docs} blog documents")
