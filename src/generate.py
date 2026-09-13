"""
Phase 3 — Generation Layer: combine retrieved context with the question into
a prompt, and generate a grounded answer with a local Ollama model.

The prompt explicitly instructs the model to say it doesn't have enough data
rather than guess, and to only use numbers/facts that appear in the retrieved
context — it should never compute or infer a statistic that isn't literally
present in a chunk.
"""

import ollama

from retrieve import retrieve

MODEL_NAME = "llama3.2:3b"

SYSTEM_PROMPT = """You are a data analyst assistant answering questions about an e-commerce dataset.

Rules:
- Only use facts, numbers, and quotes that literally appear in the CONTEXT below.
- Never calculate, estimate, or infer a number that isn't directly stated in the context.
- If the context doesn't contain enough information to answer confidently, say exactly: "I don't have enough data to answer this question." Do not guess.
- The context you're given is a small retrieved SAMPLE, not the full dataset. For any question asking for the "highest", "lowest", "overall", "total", or a full ranking across ALL categories or time periods, you cannot determine the true answer from a sample — say so explicitly (e.g. "Based on the sample retrieved, X had the highest rate among the categories shown, but this may not reflect the true maximum across the full dataset.") rather than stating a sample result as if it were the global answer.
- Some context chunks are customer reviews written in Portuguese — translate their meaning into English in your answer.
- Cite which category/time period or review your answer is based on."""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_lines = []
    for c in chunks:
        label = "Aggregated stat" if c["source"] == "category_month_summary" else "Customer review"
        context_lines.append(f"[{label} - {c['category']}] {c['text']}")
    context = "\n".join(context_lines)
    return f"CONTEXT:\n{context}\n\nQUESTION: {question}"


def answer(question: str, k: int = 5) -> dict:
    chunks = retrieve(question, k=k)
    prompt = build_prompt(question, chunks)

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return {
        "question": question,
        "answer": response["message"]["content"],
        "sources": chunks,
    }


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What do customers complain about in the furniture category?"
    result = answer(question)
    print(f"Q: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    print("Sources used:")
    for s in result["sources"]:
        print(f"  [{s['score']:.3f}] ({s['source']}, {s['category']}) {s['text'][:100]}")
