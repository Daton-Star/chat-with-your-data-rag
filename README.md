# Chat With Your Data

A hybrid RAG + text-to-SQL analyst assistant that lets you ask natural-language questions about the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and get grounded, data-backed answers — instead of writing SQL/pandas by hand.

**Fully free and local**: no API keys, no cloud costs. Embeddings and generation both run on your own machine.

## The problem

Most "chat with your data" tutorials embed raw table rows and hope semantic search can answer analytical questions. It can't: "which category has the highest cancellation rate" isn't a similarity match, it's a `GROUP BY` + `COUNT` + sort — the answer doesn't exist in any single row, only after aggregation. This project is built around that constraint: aggregation/ranking questions are routed to a **text-to-SQL** path that computes an exact answer, while qualitative questions ("what do customers complain about") go through **semantic RAG** over review text.

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
              │ src/build_db.py                  │ src/build_index.py
              ▼                                  ▼
        SQLite (olist.db)          paraphrase-multilingual-MiniLM-L12-v2
   category_month_stats +                  (local embedding model)
   category_overall_stats                          │
              │                                     ▼
              │                            FAISS vector index
              │                           (42,248 chunks, 384-dim)
              │                                     │
              │                    ┌────────────────────────────┐
   question ──┴───────────────────►│   Router (LLM classifier)   │
                                    │  SQL  vs  SEMANTIC           │
                                    └───────────┬─────────────────┘
                              ┌─────────────────┴──────────────────┐
                              ▼ SQL                                 ▼ SEMANTIC
                    src/text_to_sql.py                    src/retrieve.py (top-k)
                    generate SQL → validate                        │
                    (SELECT-only) → execute                        ▼
                              │                          Prompt template + guardrails
                              └─────────────┬──────────────────────┘
                                            ▼  src/generate.py / src/hybrid.py
                                  llama3.2:3b via Ollama
                                   (local generation)
                                            │
                                            ▼
                                      Streamlit UI
                                       (src/app.py)
```

**Two separate corpora for the semantic path, embedded together but tagged by source:**
1. **`category_month_summaries`** — orders/items pre-aggregated into one natural-language sentence per (category × month): order count, revenue, cancellation rate, avg review score.
2. **`review_chunks`** — real free-text customer reviews (Portuguese), filtered to non-empty comments. This is the genuine semantic-search use case: "what are people unhappy about."

**Two pre-aggregated SQL tables for the exact-answer path:**
3. **`category_month_stats`** — same aggregation as above, but as a real SQL table.
4. **`category_overall_stats`** — the same stats aggregated across *all* time per category. This exists specifically because a small local model asked to compute "overall" by combining monthly rows picked a month with 1 order and 1 cancellation (100% rate) as the "highest" — technically correct SQL, wrong answer. Giving it a pre-aggregated table for the common case removes that failure mode entirely (see [Example Q&A](#example-qa-real-output-from-this-system)).

## Stack

| Component | Choice | Why |
|---|---|---|
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers) | Free, local, and — critically — multilingual. An English-only model scored ~0.28 similarity on relevant Portuguese reviews; switching to a multilingual model raised true matches to ~0.75. |
| Vector store | FAISS (flat, inner-product on normalized vectors) | Free, no server to run, fine at this scale (~42K vectors). |
| Generation | `llama3.2:3b` via Ollama | Free, local, no API key. Small enough to run on a laptop CPU. |
| SQL store | SQLite | Free, zero-config, single file. The LLM writes SELECT-only queries validated before execution against a read-only connection. |
| Interface | Streamlit | Fast to build, standard for analytics demos. |

## Example Q&A (real output from this system)

**Q: What do customers complain about in the furniture category?**
> The customer complains about: the product's finish being incompatible with the price... The packaging being fragile and causing damage during shipping... [and a second review about] the product not being available in stock.

Grounded correctly, translated from Portuguese, cited to specific reviews.

**Q: Who is the CEO of Olist and what is their annual salary?**
> I don't have enough data to answer this question.

Correct refusal — this isn't in the dataset, and the model didn't invent an answer.

**Q: Which product category has the highest cancellation rate overall?**
> The product category with the highest cancellation rate overall is 'diapers_and_hygiene'.

*(Generated SQL: `SELECT category FROM category_overall_stats ORDER BY cancellation_rate_pct DESC LIMIT 1`)*

This is the most important example in this README, because it tells a two-part story:
1. **Pure RAG got this wrong.** Early testing had this exact question answered by semantic retrieval, which confidently stated `'auto'` (0.86% cancellation) as the answer — a real result, but not the true maximum, because retrieval only sees a top-k sample, not the full 1,271-row table. The prompt was tightened to at least caveat this ("based on the sample retrieved...") rather than overclaim.
2. **Naive text-to-SQL got this wrong too, differently.** The first version of the SQL path found `'telephony'` at "100% cancellation" — technically a correct query result, but from a single month with only 1-2 total orders. Statistically meaningless, but a syntactically perfect SQL query will happily return it. The fix was adding `category_overall_stats`, a properly pre-aggregated table, so the model didn't have to get a multi-month rollup right on its own. Verified independently with pandas: `diapers_and_hygiene` at 3.70% is the correct answer.

The lesson: neither RAG nor text-to-SQL is automatically safe for aggregation questions — both need the underlying data shaped to match the question's actual grain before the LLM touches it.

## Limitations

- **The router is itself an LLM call and can misclassify a question** — e.g. sending a qualitative question down the SQL path (which would fail safely and fall back to RAG) or a numeric question down the semantic path. No accuracy number is claimed for routing precision; this is a demo-scale project, not a benchmarked classifier.
- **Text-to-SQL is only as good as the tables it's given.** This project's fix (a pre-aggregated `category_overall_stats` table) works because the failure mode was anticipated and tested for. A genuinely novel aggregation question outside these two tables would fall back to the raw `orders`/`order_category` tables, where a 3B model is far more likely to write an incorrect join.
- **Reviews are in Portuguese**, translated by the LLM at answer time. Nuance can be lost, and translation quality depends on the small local model.
- **Small local model (3B params) makes occasional arithmetic mistakes** in the semantic (non-SQL) path when comparing multiple numbers in retrieved text — it correctly avoided fabricating data but still misread which of two numbers was larger in one test.
- **Multi-category orders are collapsed to one "primary" category** (the highest-priced line item) during data prep, to avoid double-counting an order across categories in the aggregates. A small simplification, not a hidden bug.
- **No "returns" field exists in this dataset** — only `order_status` (delivered/shipped/canceled/etc). Questions are framed around cancellation rate and review score as the closest real proxies for customer dissatisfaction.
- **CPU-only local generation is slow** (10-30s per answer, more for SQL questions since routing + generation + summarizing is 2-3 sequential LLM calls) compared to a hosted API. Acceptable for a demo/portfolio project, not for production-scale query volume.

## Setup

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
winget install Ollama.Ollama
ollama pull llama3.2:3b
```

Place the Olist CSVs (`olist_orders_dataset.csv`, `olist_order_items_dataset.csv`, `olist_order_reviews_dataset.csv`, `olist_products_dataset.csv`, `product_category_name_translation.csv`) in `data/raw/` — download from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (not included in this repo).

```bash
.venv\Scripts\python src\data_prep.py      # build the two RAG corpora
.venv\Scripts\python src\build_index.py    # embed + build the FAISS index
.venv\Scripts\python src\build_db.py       # build the SQLite database for text-to-SQL
.venv\Scripts\streamlit run src\app.py     # launch the app (uses src/hybrid.py to route)
```

## Roadmap

- **Query result re-ranking / larger model** for the semantic path's occasional arithmetic slips.
- **Broader SQL schema coverage** (raw order/item-level questions beyond the two pre-aggregated tables) with corresponding testing for join correctness.

## Resume framing

> Built a hybrid RAG + text-to-SQL analyst assistant over a 100K-row e-commerce dataset. Designed a router that sends aggregation/ranking questions to a validated text-to-SQL path and qualitative questions to semantic retrieval over free-text reviews — avoiding the common failure mode of naive RAG on tabular data. Diagnosed and fixed two distinct wrong-answer modes on the same question: RAG overclaiming from a retrieved sample, and text-to-SQL returning a statistically meaningless result from a small-sample month; verified the final answer independently with pandas. Fully local/free stack (sentence-transformers, FAISS, SQLite, Ollama, Streamlit) — no API keys or cloud costs.
