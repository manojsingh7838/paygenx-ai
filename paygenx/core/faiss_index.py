# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("all-MiniLM-L6-v2")

# dimension = 384
# index = faiss.IndexFlatL2(dimension)
# qa_store = []

# def embed_text(text):
#     return model.encode([text])[0].astype(np.float32)

# def add_to_index(question, answer):
#     vec = embed_text(question)
#     index.add(np.array([vec]))
#     qa_store.append((question, answer, vec))

# def search_similar(question, top_k=1, threshold=0.8):
#     if index.ntotal == 0:
#         return None
#     vec = embed_text(question)
#     D, I = index.search(np.array([vec]), top_k)
#     if D[0][0] < (1 - threshold):
#         return None
#     q, a, _ = qa_store[I[0][0]]
#     return q, a




# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import os
# import pickle

# # Load sentence transformer model
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # Dimension of the model embeddings
# dimension = 384  # Make sure this matches your model

# # Initialize FAISS index and store
# index = faiss.IndexFlatL2(dimension)
# qa_store = []

# # File paths for saving
# INDEX_FILE = "faiss_index.idx"
# STORE_FILE = "qa_store.pkl"

# # --------------------------
# # Helper Functions
# # --------------------------

# def embed_text(text):
#     embedding = model.encode([text])[0]
#     return np.array(embedding).astype("float32")

# def add_to_index(question, answer):
#     vec = embed_text(question)
#     index.add(np.array([vec]))
#     qa_store.append((question, answer, vec))
#     save_faiss_index()

# def search_similar(question, threshold=0.8):
#     if index.ntotal == 0:
#         return None

#     vec = embed_text(question)
#     D, I = index.search(np.array([vec]), k=1)
#     best_score = D[0][0]
#     best_idx = I[0][0]

#     if best_score > (1 - threshold):  # L2 distance; lower is better
#         matched_question, matched_answer, _ = qa_store[best_idx]
#         return {
#             "matched_question": matched_question,
#             "matched_answer": matched_answer,
#             "distance": float(best_score),
#         }

#     return None

# # --------------------------
# # Persistence
# # --------------------------

# def save_faiss_index():
#     faiss.write_index(index, INDEX_FILE)
#     with open(STORE_FILE, "wb") as f:
#         pickle.dump(qa_store, f)

# def load_faiss_index():
#     global index, qa_store
#     if os.path.exists(INDEX_FILE):
#         index_loaded = faiss.read_index(INDEX_FILE)
#         if index_loaded.d == dimension:
#             index = index_loaded
#     if os.path.exists(STORE_FILE):
#         with open(STORE_FILE, "rb") as f:
#             qa_store = pickle.load(f)

# # --------------------------
# # Optional: Clear for dev
# # --------------------------

# def clear_index():
#     global index, qa_store
#     index = faiss.IndexFlatL2(dimension)
#     qa_store = []
#     save_faiss_index()

# # --------------------------
# # Load index on module start
# # --------------------------

# load_faiss_index()



# import os
# import pickle
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer

# # Load lightweight multilingual model for embedding
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # Embedding dimension
# dimension = 384  # Do NOT change unless you change the model

# # FAISS index
# index = faiss.IndexFlatL2(dimension)
# qa_store = []  # Stores tuples: (question, answer, embedding)

# # File paths (render supports persistent disk in /persist)
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# INDEX_FILE = os.path.join(BASE_DIR, "faiss_index.idx")
# STORE_FILE = os.path.join(BASE_DIR, "qa_store.pkl")

# # -----------------------
# # Embedding Function
# # -----------------------
# def embed_text(text):
#     embedding = model.encode([text])[0]
#     return np.array(embedding).astype("float32")

# # -----------------------
# # Add to Index
# # -----------------------
# def add_to_index(question, answer):
#     vec = embed_text(question)
#     index.add(np.array([vec]))
#     qa_store.append((question, answer, vec))
#     save_faiss_index()

# # -----------------------
# # Search Similar Question
# # -----------------------
# def search_similar(question, threshold=0.8):
#     if index.ntotal == 0:
#         return None

#     vec = embed_text(question)
#     D, I = index.search(np.array([vec]), k=1)
#     best_score = D[0][0]
#     best_idx = I[0][0]

#     if best_score < (1 - threshold):  # Lower is better for L2 distance
#         matched_question, matched_answer, _ = qa_store[best_idx]
#         return {
#             "matched_question": matched_question,
#             "matched_answer": matched_answer,
#             "distance": float(best_score),
#         }

#     return None

# # -----------------------
# # Persistence
# # -----------------------
# def save_faiss_index():
#     faiss.write_index(index, INDEX_FILE)
#     with open(STORE_FILE, "wb") as f:
#         pickle.dump(qa_store, f)

# def load_faiss_index():
#     global index, qa_store
#     if os.path.exists(INDEX_FILE):
#         index_loaded = faiss.read_index(INDEX_FILE)
#         if index_loaded.d == dimension:
#             index = index_loaded
#     if os.path.exists(STORE_FILE):
#         with open(STORE_FILE, "rb") as f:
#             qa_store = pickle.load(f)

# # -----------------------
# # Clear Index (Dev Only)
# # -----------------------
# def clear_index():
#     global index, qa_store
#     index = faiss.IndexFlatL2(dimension)
#     qa_store = []
#     save_faiss_index()

# # Load everything on module import
# load_faiss_index()


import os
import pickle
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

# Load model lazily to reduce memory pressure during import
_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
    return _model

# Embedding dimension of the selected lightweight model
dimension = 384

# FAISS index setup
index = faiss.IndexFlatL2(dimension)
qa_store = []

# Persistent storage paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "faiss_index.idx")
STORE_FILE = os.path.join(BASE_DIR, "qa_store.pkl")

# --------------------------
# Embed text
# --------------------------
def embed_text(text):
    model = get_model()
    embedding = model.encode([text], convert_to_numpy=True)[0]
    return embedding.astype("float32")

# --------------------------
# Add to FAISS and store
# --------------------------
def add_to_index(question, answer):
    vec = embed_text(question)
    index.add(np.array([vec]))
    qa_store.append((question, answer, vec))
    save_faiss_index()

# --------------------------
# Search similar
# --------------------------
def search_similar(question, threshold=0.8):
    if index.ntotal == 0:
        return None

    vec = embed_text(question)
    D, I = index.search(np.array([vec]), k=1)
    best_score = D[0][0]
    best_idx = I[0][0]

    if best_score < (1 - threshold):
        matched_question, matched_answer, _ = qa_store[best_idx]
        return {
            "matched_question": matched_question,
            "matched_answer": matched_answer,
            "distance": float(best_score),
        }
    return None

# --------------------------
# Save index and store
# --------------------------
def save_faiss_index():
    try:
        faiss.write_index(index, INDEX_FILE)
        with open(STORE_FILE, "wb") as f:
            pickle.dump(qa_store, f)
    except Exception as e:
        print(f"Error saving index: {e}")

# --------------------------
# Load index and store
# --------------------------
def load_faiss_index():
    global index, qa_store
    if os.path.exists(INDEX_FILE):
        try:
            index_loaded = faiss.read_index(INDEX_FILE)
            if index_loaded.d == dimension:
                index = index_loaded
        except Exception as e:
            print(f"Failed to load FAISS index: {e}")
    
    if os.path.exists(STORE_FILE):
        try:
            with open(STORE_FILE, "rb") as f:
                qa_store = pickle.load(f)
        except Exception as e:
            print(f"Failed to load QA store: {e}")

# --------------------------
# Dev only: clear everything
# --------------------------
def clear_index():
    global index, qa_store
    index = faiss.IndexFlatL2(dimension)
    qa_store = []
    save_faiss_index()

# Load once
load_faiss_index()

