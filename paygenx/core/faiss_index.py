import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384

# Use cosine similarity
index = faiss.IndexFlatIP(dimension)
qa_store = []

def embed_text(text):
    vec = model.encode([text])[0].astype(np.float32)
    return vec / np.linalg.norm(vec)

def add_to_index(question, answer):
    vec = embed_text(question)
    index.add(np.array([vec]))
    qa_store.append((question, answer, vec))

# def search_similar(question, top_k=1, threshold=0.85):
#     if index.ntotal == 0:
#         return None
#     vec = embed_text(question)
#     D, I = index.search(np.array([vec]), top_k)
#     if D[0][0] < threshold:
#         return None
#     q, a, _ = qa_store[I[0][0]]
#     return q, a


def search_similar(question, top_k=3, threshold=0.85):
    if index.ntotal == 0:
        return None
    vec = embed_text(question)
    D, I = index.search(np.array([vec]), top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if score < threshold:
            continue
        q, a, _ = qa_store[idx]
        results.append((q, a, idx))  # ✅ Return 3 values per result

    return results if results else None



# 🔹 NEW: Save to training file
def append_to_training_data(question, answer, file_path="training_data.jsonl"):
    data = {"input": question, "output": answer}
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")
