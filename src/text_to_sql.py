"""
Phase 4 — Text-to-SQL fallback for aggregation/ranking questions that pure
RAG can't reliably answer (see Phase 3 findings: retrieval returns a sample,
not the full dataset, so "highest X overall" can't be trusted from top-k
chunks alone).
"""

import re
import sqlite3

import ollama

DB_PATH = "data/processed/olist.db"
MODEL_NAME = "llama3.2:3b"

SCHEMA = """\
Table category_overall_stats (one row per product category, aggregated across ALL time):
  category TEXT, total_orders INTEGER, total_revenue REAL, canceled_orders INTEGER,
  cancellation_rate_pct REAL, avg_review_score REAL
  Use this for any question about a category's performance "overall", "in total",
  or without a specific time period — it's already correctly aggregated across
  every month, so you never need to combine rows yourself.

Table category_month_stats (one row per product category per calendar month):
  category TEXT, year_month TEXT (format 'YYYY-MM'), total_orders INTEGER,
  total_revenue REAL, canceled_orders INTEGER, cancellation_rate_pct REAL,
  avg_review_score REAL
  Use this ONLY when the question specifies a particular month or asks about a
  trend over time. A single month's rate can be based on very few orders and
  be misleading (e.g. 1 canceled out of 1 order = 100%) — do not use this table
  to answer "overall" questions.

Table orders:
  order_id TEXT, order_status TEXT, order_purchase_timestamp TEXT,
  order_estimated_delivery_date TEXT

Table order_category (each order's primary/highest-value category):
  order_id TEXT, category TEXT, top_item_price REAL

Table reviews:
  review_id TEXT, order_id TEXT, review_score INTEGER (1-5)
"""

SQL_SYSTEM_PROMPT = f"""You write SQLite SELECT queries to answer data questions.

Schema:
{SCHEMA}

Rules:
- Output ONLY the SQL query. No explanation, no markdown code fences, no commentary.
- Only SELECT statements. Never modify data.
- Prefer category_month_stats for anything about rates, revenue, or order counts by category — it's already aggregated.
- If the question can't be answered with this schema, output exactly: NO_QUERY"""

FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|attach|pragma|create)\b", re.IGNORECASE)


def generate_sql(question: str) -> str:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SQL_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    sql = response["message"]["content"].strip()
    sql = re.sub(r"^```sql\s*|\s*```$", "", sql, flags=re.IGNORECASE).strip()
    return sql


def is_safe_select(sql: str) -> bool:
    stripped = sql.strip().rstrip(";")
    if not stripped:
        return False
    if ";" in stripped:
        return False
    if not stripped.upper().startswith("SELECT"):
        return False
    if FORBIDDEN.search(stripped):
        return False
    return True


def execute_sql(sql: str):
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        return columns, rows
    finally:
        conn.close()


def answer_with_sql(question: str) -> dict:
    sql = generate_sql(question)

    if sql == "NO_QUERY" or not is_safe_select(sql):
        return {"question": question, "sql": sql, "error": "Could not generate a safe query for this question."}

    try:
        columns, rows = execute_sql(sql)
    except sqlite3.Error as e:
        return {"question": question, "sql": sql, "error": str(e)}

    result_text = ", ".join(columns) + "\n" + "\n".join(str(r) for r in rows[:20])
    summary_prompt = (
        f"QUESTION: {question}\n\nSQL QUERY RUN: {sql}\n\nQUERY RESULT:\n{result_text}\n\n"
        "Answer the question in one or two sentences using only these exact results. "
        "Do not add numbers that aren't in the result."
    )
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": summary_prompt}],
    )

    return {
        "question": question,
        "sql": sql,
        "columns": columns,
        "rows": rows,
        "answer": response["message"]["content"],
    }


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "Which product category has the highest cancellation rate overall?"
    result = answer_with_sql(question)
    print(f"Q: {result['question']}\n")
    print(f"SQL: {result['sql']}\n")
    if "error" in result:
        print(f"ERROR: {result['error']}")
    else:
        print(f"Result rows: {result['rows'][:10]}\n")
        print(f"A: {result['answer']}")
