"""
Phase 5 — Streamlit interface for "Chat With Your Data".

Run with: streamlit run src/app.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

from hybrid import answer

st.set_page_config(page_title="Chat With Your Data", page_icon="\U0001F4CA", layout="centered")

st.title("\U0001F4CA Chat With Your Data")
st.caption(
    "Ask questions about the Olist Brazilian e-commerce dataset. "
    "Answers are grounded in retrieved data — the assistant will say so if it doesn't have enough evidence."
)

with st.sidebar:
    st.header("About")
    st.markdown(
        "- **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` (local, free)\n"
        "- **Generation:** `llama3.2:3b` via Ollama (local, free)\n"
        "- **Vector store:** FAISS\n"
        "- **Corpora:** customer reviews (Portuguese) + pre-aggregated category/month stats\n"
        "- **Hybrid routing:** aggregation/ranking questions are routed to a text-to-SQL "
        "path (exact answer from SQLite) instead of semantic retrieval"
    )
    st.header("Try asking")
    examples = [
        "What do customers complain about in the furniture category?",
        "Which category has the highest cancellation rate overall?",
        "Are customers happy with delivery times for electronics?",
        "Who is the CEO of Olist?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["pending_question"] = ex

    st.header("Known limitations")
    st.markdown(
        "- Reviews are in Portuguese; the model translates but may lose nuance.\n"
        "- Aggregation questions are routed to SQL for an exact answer, but the router "
        "itself is an LLM call and can occasionally misclassify a question.\n"
        "- The 3B local model can make small arithmetic mistakes comparing numbers in the semantic path."
    )

if "messages" not in st.session_state:
    st.session_state["messages"] = []

def render_evidence(msg):
    if msg.get("sql"):
        st.caption("Routed to: SQL (exact computation)")
        st.code(msg["sql"], language="sql")
    elif msg.get("sources"):
        st.caption("Routed to: semantic retrieval")
        with st.expander("Sources used"):
            for s in msg["sources"]:
                st.markdown(f"**[{s['score']:.3f}]** ({s['source']}, {s['category']}) — {s['text'][:200]}")


for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_evidence(msg)

pending = st.session_state.pop("pending_question", None)
question = st.chat_input("Ask a question about the data...") or pending

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Routing question and generating answer..."):
            result = answer(question)
        st.markdown(result["answer"])
        render_evidence(result)

    st.session_state["messages"].append({
        "role": "assistant",
        "content": result["answer"],
        "sql": result.get("sql"),
        "sources": result.get("sources"),
    })
