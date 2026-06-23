from sentence_transformers import SentenceTransformer

model = SentenceTransformer("jinaai/jina-embeddings-v5-omni-nano-retrieval", trust_remote_code=True)

print(f"Model loaded - vector size: {model.get_embedding_dimension()}")

# URLs, local paths (with or without extension), PIL.Image, np.ndarray,
# torch.Tensor, bytes, and BytesIO are all accepted directly.
# For retrieval, you must use encode_query(...) for search inputs and encode_document(...) for things you index. This applies to every modality (text, image, video, audio).
q_vec = model.encode_query("Which planet is known as the Red Planet?")
d_vec = model.encode_document("Mars is often referred to as the Red Planet due to its reddish appearance.")

print(len(d_vec.tolist()))
print(d_vec.tolist()[:10])

# # The Query:/Document: distinction applies to EVERY modality, not just text —
# # pass the image / video / audio (URL, path, or object) straight to
# # encode_query() / encode_document() to encode it as that retrieval side:
# img_as_document = model.encode_document("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg")
# img_as_query    = model.encode_query("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg")
# i_vec = model.encode("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg")
# v_vec = model.encode("https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4")  # needs `pip install av`
# a_vec = model.encode("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")  # needs `pip install librosa soundfile`

# # Fused multimodal — a tuple becomes ONE embedding in a single forward pass:
# emb = model.encode(("Winter boots, waterproof leather upper",
#                     "https://.../boot.jpg",
#                     "https://.../boot.mp4"))
