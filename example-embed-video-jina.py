import requests
import os

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv("JINA_API_KEY")}"
}
data = {
    "model": "jina-embeddings-v5-omni-small",
    "task": "retrieval.passage",
    "normalized": True,
    "input": [
        {
            "video": "https://storage.googleapis.com/jina-public/example-video-clip.mp4"  # path to your video
        }
    ]
}

response = requests.post("https://api.jina.ai/v1/embeddings", headers=headers, json=data)

print(response.json())
