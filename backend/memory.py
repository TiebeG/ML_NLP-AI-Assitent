# backend/memory.py
# Streamlit Cloud + Local SAFE long-term memory storage using ChromaDB

import os
import chromadb
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------
# ENV DETECTION (Cloud vs Local)
# ----------------------------------------------------
if os.path.exists("/mount/data"):
    MEMORY_DIR = "/mount/data/memory_db"          # Streamlit Cloud (writable)
else:
    MEMORY_DIR = os.path.join(os.getcwd(), "local_memory_db")  # Local dev

os.makedirs(MEMORY_DIR, exist_ok=True)

# ----------------------------------------------------
# EMBEDDING MODEL
# ----------------------------------------------------
_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------------------------------
# CHROMADB CLIENT (PERSISTENT & WRITABLE)
# ----------------------------------------------------
chroma = chromadb.PersistentClient(path=MEMORY_DIR)

memory_collection = chroma.get_or_create_collection(
    name="long_term_memory",
    metadata={"hnsw:space": "cosine"},
)

# ----------------------------------------------------
# HELPERS
# ----------------------------------------------------
def _embed(text: str):
    return _model.encode([text])[0].tolist()


# ----------------------------------------------------
# STORE MEMORY (WRITE)
# ----------------------------------------------------
def store_memory(text: str) -> bool:
    if not text or not text.strip():
        return False

    try:
        memory_collection.add(
            embeddings=[_embed(text)],
            documents=[text],
            ids=[str(abs(hash(text)))[:16]],
        )
        return True
    except Exception as e:
        # Never crash the app because of memory
        print(f"[MEMORY WRITE ERROR] {e}")
        return False


# ----------------------------------------------------
# RECALL MEMORY (READ)
# ----------------------------------------------------
def recall_memory(query: str, k: int = 3) -> list[str]:
    if not query or not query.strip():
        return []

    try:
        results = memory_collection.query(
            query_embeddings=[_embed(query)],
            n_results=k,
        )
        docs = results.get("documents", [[]])
        return docs[0] if docs and docs[0] else []
    except Exception as e:
        print(f"[MEMORY READ ERROR] {e}")
        return []
