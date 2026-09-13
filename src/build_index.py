"""
Phase 2 — Retrieval Pipeline: embed both corpora and build a FAISS index.

Free/local stack: sentence-transformers for embeddings, FAISS for the vector
store. Both corpora (review_chunks and category_month_summaries) are embedded
into ONE combined index, tagged with a `source` field, so retrieval can pull
from either depending on what the question needs.
"""

import json

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PROCESSED = "data/processed"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def load_corpus():
    summaries = pd.read_csv(f"{PROCESSED}/category_month_summaries.csv")
    summaries["source"] = "category_month_summary"
    summaries["id"] = "summary_" + summaries.index.astype(str)

    reviews = pd.read_csv(f"{PROCESSED}/review_chunks.csv")
    reviews["source"] = "review"
    reviews["id"] = "review_" + reviews["review_id"].astype(str)

    combined = pd.concat([
        summaries[["id", "source", "text", "category", "year_month"]],
        reviews[["id", "source", "text", "category"]].assign(year_month=None),
    ], ignore_index=True)
    return combined


def main():
    corpus = load_corpus()
    print(f"Embedding {len(corpus)} chunks with {MODEL_NAME}...")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        corpus["text"].tolist(), show_progress_bar=True, convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, f"{PROCESSED}/index.faiss")
    corpus.drop(columns=["text"]).to_json(f"{PROCESSED}/metadata.jsonl", orient="records", lines=True)
    corpus[["id", "text"]].to_json(f"{PROCESSED}/texts.jsonl", orient="records", lines=True)

    print(f"Index built: {index.ntotal} vectors, dim={embeddings.shape[1]}")
    print(f"Saved to {PROCESSED}/index.faiss, metadata.jsonl, texts.jsonl")


if __name__ == "__main__":
    main()
