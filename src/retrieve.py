"""
Phase 2 — Retrieval function: embed a query, search the FAISS index, return
the top-k matching chunks with their source metadata.
"""

import json

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

PROCESSED = "data/processed"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

_model = None
_index = None
_metadata = None
_texts = None


def _load():
    global _model, _index, _metadata, _texts
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
        _index = faiss.read_index(f"{PROCESSED}/index.faiss")
        _metadata = pd.read_json(f"{PROCESSED}/metadata.jsonl", lines=True)
        _texts = pd.read_json(f"{PROCESSED}/texts.jsonl", lines=True).set_index("id")["text"]


def retrieve(query: str, k: int = 5):
    _load()
    query_vec = _model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)
    scores, indices = _index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        row = _metadata.iloc[idx]
        results.append({
            "score": float(score),
            "source": row["source"],
            "category": row["category"],
            "text": _texts.loc[row["id"]],
        })
    return results


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "which category has the highest cancellation rate"
    for r in retrieve(query, k=5):
        print(f"[{r['score']:.3f}] ({r['source']}, {r['category']}) {r['text'][:150]}")
