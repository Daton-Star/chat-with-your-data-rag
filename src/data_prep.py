"""
Phase 1 — Data Prep for "Chat With Your Data".

Builds two separate corpora for retrieval:

1. review_chunks: real free-text customer reviews (genuine semantic RAG use case).
2. category_month_summaries: structured orders/items pre-aggregated into
   natural-language sentences (so retrieval surfaces the right numbers instead
   of the LLM trying to compute aggregates itself).

Naive row-level embedding of `orders`/`order_items` is deliberately avoided —
similarity search can't answer "highest cancellation rate by category" from
individual order rows, since the answer only exists after a GROUP BY.
"""

import pandas as pd

RAW = "data/raw"
OUT = "data/processed"


def load_raw():
    orders = pd.read_csv(f"{RAW}/olist_orders_dataset.csv", parse_dates=[
        "order_purchase_timestamp", "order_estimated_delivery_date"
    ])
    items = pd.read_csv(f"{RAW}/olist_order_items_dataset.csv")
    reviews = pd.read_csv(f"{RAW}/olist_order_reviews_dataset.csv")
    products = pd.read_csv(f"{RAW}/olist_products_dataset.csv")
    category_translation = pd.read_csv(f"{RAW}/product_category_name_translation.csv")
    return orders, items, reviews, products, category_translation


def attach_category(items, products, category_translation):
    # An order can span multiple categories (multiple line items). For both
    # corpora below we attribute each order to a single primary category —
    # the one with the highest line-item price — rather than double-counting
    # the order once per category. This is a simplification worth calling
    # out explicitly in the README limitations section.
    items = items.merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
    items = items.merge(category_translation, on="product_category_name", how="left")
    items["product_category_name_english"] = items["product_category_name_english"].fillna("unknown")

    primary = (
        items.sort_values("price", ascending=False)
        .drop_duplicates(subset="order_id", keep="first")[["order_id", "product_category_name_english", "price"]]
        .rename(columns={"product_category_name_english": "category"})
    )
    return items, primary


def build_category_month_summaries(orders, items, primary, reviews):
    order_category = orders.merge(primary[["order_id", "category"]], on="order_id", how="inner")
    order_category["year_month"] = order_category["order_purchase_timestamp"].dt.to_period("M").astype(str)

    review_scores = reviews.groupby("order_id")["review_score"].mean().rename("review_score")
    order_category = order_category.merge(review_scores, on="order_id", how="left")

    revenue = items.groupby("order_id")["price"].sum().rename("order_revenue")
    order_category = order_category.merge(revenue, on="order_id", how="left")

    grouped = order_category.groupby(["category", "year_month"])
    summary = grouped.agg(
        total_orders=("order_id", "count"),
        total_revenue=("order_revenue", "sum"),
        canceled_orders=("order_status", lambda s: (s == "canceled").sum()),
        avg_review_score=("review_score", "mean"),
    ).reset_index()
    summary["cancellation_rate_pct"] = (summary["canceled_orders"] / summary["total_orders"] * 100).round(2)
    summary["total_revenue"] = summary["total_revenue"].round(2)
    summary["avg_review_score"] = summary["avg_review_score"].round(2)

    def to_sentence(row):
        return (
            f"In {row['year_month']}, the '{row['category']}' category had {row['total_orders']} orders "
            f"totaling R${row['total_revenue']:.2f} in revenue, a {row['cancellation_rate_pct']}% cancellation rate "
            f"({row['canceled_orders']} canceled orders), and an average review score of {row['avg_review_score']:.2f} out of 5."
        )

    summary["text"] = summary.apply(to_sentence, axis=1)
    return summary


def build_review_chunks(reviews, orders, primary):
    reviews = reviews.dropna(subset=["review_comment_message"])
    reviews = reviews.merge(orders[["order_id", "order_purchase_timestamp"]], on="order_id", how="left")
    reviews = reviews.merge(primary[["order_id", "category"]], on="order_id", how="left")
    reviews["category"] = reviews["category"].fillna("unknown")

    def to_text(row):
        title = f"{row['review_comment_title']}. " if pd.notna(row["review_comment_title"]) else ""
        return f"{title}{row['review_comment_message']}"

    reviews["text"] = reviews.apply(to_text, axis=1)
    return reviews[[
        "review_id", "order_id", "category", "review_score",
        "order_purchase_timestamp", "text",
    ]]


def main():
    orders, items, reviews, products, category_translation = load_raw()
    items, primary = attach_category(items, products, category_translation)

    summaries = build_category_month_summaries(orders, items, primary, reviews)
    summaries.to_csv(f"{OUT}/category_month_summaries.csv", index=False)

    review_chunks = build_review_chunks(reviews, orders, primary)
    review_chunks.to_csv(f"{OUT}/review_chunks.csv", index=False)

    print(f"category_month_summaries: {len(summaries)} rows -> {OUT}/category_month_summaries.csv")
    print(f"review_chunks: {len(review_chunks)} rows -> {OUT}/review_chunks.csv")
    print("\nSample summary sentence:")
    print(summaries["text"].iloc[0])
    print("\nSample review chunk:")
    print(review_chunks["text"].iloc[0])


if __name__ == "__main__":
    main()
