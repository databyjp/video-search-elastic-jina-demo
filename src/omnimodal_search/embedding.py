from functools import lru_cache
from sentence_transformers import SentenceTransformer

@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(
        "jinaai/jina-embeddings-v5-omni-small-retrieval",
        trust_remote_code=True,
    )
