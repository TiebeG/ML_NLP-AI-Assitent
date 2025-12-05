# backend/memory.py

import chromadb
from sentence_transformers import SentenceTransformer

# Simple, shared embedding model for memory
_model = SentenceTransformer("all-MiniLM-L6-v2")

# Persistent client (will create folder "memory_db")
_chroma = chromadb.PersistentClient(path="memory_db")

_memory_collection = _chroma.get_or_create_collection(
    name="long_term_memory",
    metadata={"hnsw:space": "cosine"},
)


def _embed(text: str):
    return _model.encode([text])[0].tolist()


def store_memory(text: str) -> bool:
    """
    Store a piece of long-term memory.
    """
    if not text or not text.strip():
        return False

    embedding = _embed(text)
    _memory_collection.add(
        embeddings=[embedding],
        documents=[text],
        ids=[str(abs(hash(text)))[:16]],
    )
    return True


def recall_memory(query: str, k: int = 3) -> list[str]:
    """
    Retrieve up to k relevant memories for the given query.
    """
    if not query or not query.strip():
        return []

    embedding = _embed(query)
    results = _memory_collection.query(
        query_embeddings=[embedding],
        n_results=k,
    )
    docs = results.get("documents", [[]])
    if not docs or not docs[0]:
        return []
    return docs[0]
