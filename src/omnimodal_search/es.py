from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os

load_dotenv("elastic-start-local/.env")


def get_client() -> Elasticsearch:
    return Elasticsearch(
        os.getenv("ES_LOCAL_URL"),
        api_key=os.getenv("ES_LOCAL_API_KEY"),
    )


def create_index(client: Elasticsearch, name: str, dims: int):
    return client.indices.create(
        index=name,
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
