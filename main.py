from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os

load_dotenv("elastic-start-local/.env")

client = Elasticsearch(
    os.getenv("ES_LOCAL_URL"),
    api_key=os.getenv("ES_LOCAL_API_KEY"),
)
print(client.info())

INDEX = "elastic-wizard"

# RESET INDEX
client.indices.delete(index=INDEX, ignore_unavailable=True)
client.indices.create(
    index=INDEX,
    mappings={
        "properties": {
            "content_type": {"type": "keyword"},
            "title": {"type": "text"},
            "embedding": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
)

if client.indices.exists(index=INDEX):
    print(f"Index '{INDEX}' successfully created")
