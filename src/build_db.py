"""
Phase 4 — Build a SQLite database for the text-to-SQL fallback.

Reuses the exact same aggregation logic from data_prep.py so the SQL path and
the RAG path agree on definitions (e.g. "primary category" per order). The
key design choice: `category_month_stats` is materialized as a real table,
not just left as raw normalized orders/items/products. A small local LLM is
far more reliable writing `SELECT category FROM category_month_stats ORDER BY
cancellation_rate_pct DESC` than reconstructing a 3-table join with a GROUP BY
from scratch. This is the same "semantic layer" idea production text-to-SQL
systems use.
"""

import sqlite3

from data_prep import attach_category, build_category_month_summaries, load_raw

OUT = "data/processed"
DB_PATH = f"{OUT}/olist.db"


def main():
    orders, items, reviews, products, category_translation = load_raw()
    items, primary = attach_category(items, products, category_translation)
    summaries = build_category_month_summaries(orders, items, primary, reviews)

    conn = sqlite3.connect(DB_PATH)

    orders[["order_id", "order_status", "order_purchase_timestamp", "order_estimated_delivery_date"]].to_sql(
        "orders", conn, if_exists="replace", index=False
    )
    primary.rename(columns={"price": "top_item_price"}).to_sql(
        "order_category", conn, if_exists="replace", index=False
    )
    reviews[["review_id", "order_id", "review_score"]].to_sql(
        "reviews", conn, if_exists="replace", index=False
    )
    summaries.drop(columns=["text"]).to_sql(
        "category_month_stats", conn, if_exists="replace", index=False
    )

    overall = summaries.groupby("category").agg(
        total_orders=("total_orders", "sum"),
        total_revenue=("total_revenue", "sum"),
        canceled_orders=("canceled_orders", "sum"),
    ).reset_index()
    overall["cancellation_rate_pct"] = (overall["canceled_orders"] / overall["total_orders"] * 100).round(2)
    overall["avg_review_score"] = (
        (summaries["avg_review_score"] * summaries["total_orders"]).groupby(summaries["category"]).sum()
        / overall.set_index("category")["total_orders"]
    ).round(2).values
    overall.to_sql("category_overall_stats", conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()
    print(f"Built {DB_PATH}")
    print("Tables: orders, order_category, reviews, category_month_stats, category_overall_stats")
    print(f"category_month_stats: {len(summaries)} rows, category_overall_stats: {len(overall)} rows")


if __name__ == "__main__":
    main()
