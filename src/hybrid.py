"""
Phase 4 — Hybrid router: classify each question as needing an exact
aggregate/ranking computation (-> text-to-SQL) or a qualitative/semantic
lookup (-> RAG). This is the fix for the exact weakness found in Phase 3:
pure RAG answering "highest X overall" from a retrieved sample instead of
the full dataset.
"""

import ollama

from generate import answer as rag_answer
from text_to_sql import answer_with_sql

MODEL_NAME = "llama3.2:3b"

ROUTER_PROMPT = """Classify the question as SQL or SEMANTIC.

SQL = requires computing/ranking/comparing a statistic across the full dataset
(highest, lowest, total, average, count, percentage, "overall", "which category has the most/least").

SEMANTIC = asking about opinions, complaints, qualitative descriptions, or specific facts
that would appear in text (what do customers say, are people happy, complaints about X).

Answer with exactly one word: SQL or SEMANTIC."""


def route(question: str) -> str:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    verdict = response["message"]["content"].strip().upper()
    return "sql" if "SQL" in verdict else "semantic"


def answer(question: str, k: int = 5) -> dict:
    mode = route(question)

    if mode == "sql":
        result = answer_with_sql(question)
        if "error" not in result:
            return {
                "question": question,
                "mode": "sql",
                "answer": result["answer"],
                "sql": result["sql"],
                "sources": [],
            }
        # SQL generation/execution failed — fall back to RAG rather than dead-ending.
        mode = "semantic"

    rag_result = rag_answer(question, k=k)
    return {
        "question": question,
        "mode": "semantic",
        "answer": rag_result["answer"],
        "sql": None,
        "sources": rag_result["sources"],
    }


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "Which product category has the highest cancellation rate overall?"
    result = answer(question)
    print(f"Q: {result['question']}")
    print(f"Mode: {result['mode']}")
    if result["sql"]:
        print(f"SQL: {result['sql']}")
    print(f"\nA: {result['answer']}")
