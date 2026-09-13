# Chat With Your Data

A RAG-based analyst assistant that lets you ask natural-language questions about the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and get grounded, data-backed answers — instead of writing SQL/pandas by hand.

**Fully free and local**: no API keys, no cloud costs. Embeddings and generation both run on your own machine.

## The problem

Most "chat with your data" tutorials embed raw table rows and hope semantic search can answer analytical questions. It can't: "which category has the highest cancellation rate" isn't a similarity match, it's a `GROUP BY` + `COUNT` + sort — the answer doesn't exist in any single row, only after aggregation. This project is built around that constraint rather than around it.

## Architecture

```
                    ┌─────────────────────┐
                    │   Olist raw CSVs     │
                    │ (orders, items,      │
                    │  reviews, products)  │
                    └──────────┬───────────┘
                               │  src/data_prep.py
              ┌────────────────┴────────────────┐
              ▼                                  ▼
   category_month_summaries.csv        review_chunks.csv
   (pre-aggregated stats, one            (real customer review
    sentence per category×month)          text, Portuguese)
              │                                  │
              └────────────────┬─────────────────┘
                               │  src/build_index.py
                               ▼
              paraphrase-multilingual-MiniLM-L12-v2
                     (local embedding model)
                               │
                               ▼
                        FAISS vector index
                       (42,248 chunks, 384-dim)
                               │  src/retrieve.py (top-k search)
                               ▼
                    ┌──────────────────────┐
   question  ───────►   Prompt template     │
                    │  (context + guardrails)│
                    └──────────┬────────────┘
                               │  src/generate.py
                               ▼
                    llama3.2:3b via Ollama
                     (local generation)
                               │
                               ▼
                         Streamlit UI
                          (src/app.py)
```

**Two separate corpora, embedded together but tagged by source:**
1. **`category_month_summaries`** — orders/items pre-aggregated into one natural-language sentence per (category × month): order count, revenue, cancellation rate, avg review score. Retrieval surfaces the right slice of numbers; the LLM reads them, it doesn't compute them.
2. **`review_chunks`** — real free-text customer reviews (Portuguese), filtered to non-empty comments. This is the genuine semantic-search use case: "what are people unhappy about."

## Stack

| Component | Choice | Why |
|---|---|---|
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers) | Free, local, and — critically — multilingual. An English-only model scored ~0.28 similarity on relevant Portuguese reviews; switching to a multilingual model raised true matches to ~0.75. |
| Vector store | FAISS (flat, inner-product on normalized vectors) | Free, no server to run, fine at this scale (~42K vectors). |
| Generation | `llama3.2:3b` via Ollama | Free, local, no API key. Small enough to run on a laptop CPU. |
| Interface | Streamlit | Fast to build, standard for analytics demos. |

## Example Q&A (real output from this system)

**Q: What do customers complain about in the furniture category?**
> The customer complains about: the product's finish being incompatible with the price... The packaging being fragile and causing damage during shipping... [and a second review about] the product not being available in stock.

Grounded correctly, translated from Portuguese, cited to specific reviews.

**Q: Who is the CEO of Olist and what is their annual salary?**
> I don't have enough data to answer this question.

Correct refusal — this isn't in the dataset, and the model didn't invent an answer.

**Q: Which product category has the highest cancellation rate overall?**
> Based on the sample retrieved, the 'auto' category has the highest cancellation rate among the categories shown, with 0.86% of orders canceled. However, I don't have enough data to answer this question confidently...

This is the most important example in this README. The retrieved sample is *not* guaranteed to contain the true global maximum — semantic search finds text similar to the question, not the numerically largest value. Early testing showed the model would otherwise confidently state a sample result as global fact. The prompt was tightened specifically to catch this: see [Limitations](#limitations).

## Limitations

- **Aggregation questions ("highest", "overall") are answered from a retrieved sample, not a full table scan.** Pure RAG cannot reliably rank across an entire dataset — this needs a text-to-SQL fallback (see Roadmap) or a re-ranking step over the complete aggregated table, not just the top-k retrieved rows.
- **Reviews are in Portuguese**, translated by the LLM at answer time. Nuance can be lost, and translation quality depends on the small local model.
- **Small local model (3B params) makes occasional arithmetic mistakes** when comparing multiple numbers in context — it correctly avoided fabricating data but still misread which of two numbers was larger in one test. A larger model or an actual code-based comparison would fix this.
- **Multi-category orders are collapsed to one "primary" category** (the highest-priced line item) during data prep, to avoid double-counting an order across categories in the aggregates. A small simplification, not a hidden bug.
- **No "returns" field exists in this dataset** — only `order_status` (delivered/shipped/canceled/etc). Questions are framed around cancellation rate and review score as the closest real proxies for customer dissatisfaction.
- **CPU-only local generation is slow** (10-30s per answer) compared to a hosted API. Acceptable for a demo/portfolio project, not for production-scale query volume.

## Setup

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
winget install Ollama.Ollama
ollama pull llama3.2:3b
```

Place the Olist CSVs (`olist_orders_dataset.csv`, `olist_order_items_dataset.csv`, `olist_order_reviews_dataset.csv`, `olist_products_dataset.csv`, `product_category_name_translation.csv`) in `data/raw/` — download from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (not included in this repo).

```bash
.venv\Scripts\python src\data_prep.py      # build the two corpora
.venv\Scripts\python src\build_index.py    # embed + build the FAISS index
.venv\Scripts\streamlit run src\app.py     # launch the app
```

## Roadmap

- **Text-to-SQL fallback** for aggregation/ranking questions: detect when a question needs a full-table computation, generate SQL against the structured tables instead of relying on retrieval, and return the exact computed answer.

## Resume framing

> Built a retrieval-augmented generation (RAG) assistant over a 100K-row e-commerce dataset, combining semantic search over free-text customer reviews with pre-aggregated structured summaries to avoid the common failure mode of naive RAG on tabular data. Diagnosed and fixed a cross-lingual retrieval bug (English queries against Portuguese text) by switching embedding models, and engineered prompt guardrails against a subtler hallucination mode — the model presenting a retrieved sample as if it were a complete-dataset result. Fully local/free stack (sentence-transformers, FAISS, Ollama, Streamlit).
