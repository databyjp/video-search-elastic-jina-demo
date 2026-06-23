from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os

load_dotenv("elastic-start-local/.env")

client = Elasticsearch(
    os.getenv("ES_LOCAL_URL"),
    api_key=os.getenv("ES_LOCAL_API_KEY"),
)
print(client.info())
