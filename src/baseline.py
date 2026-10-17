"""Baseline: plain vector RAG. Find the k chunks closest in meaning, answer only from them, cite chunk ids.
This is the system the graph version must beat, so it gets the same model, prompt rules and citation check.

Run: uv run python src/baseline.py "Which supplier do NVIDIA and AMD share?"
"""

import os
import sys
import time

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from pydantic import BaseModel

from extract import PRICE_PER_M
from llm import ask_structured, embed

load_dotenv()

SYSTEM = """You answer questions about NVIDIA, AMD and Intel using ONLY the numbered passages given.
- After every claim, cite the passage id in square brackets, e.g. [NVDA-10K-2026-02-25-6637efa5bd47].
- List every id you cited in `citations`.
- If the passages don't contain the answer, set answerable=false and say so. Never use outside knowledge."""


class Answer(BaseModel):
    answer: str
    citations: list[str]
    answerable: bool


def retrieve(conn, question: str, k: int) -> list[tuple[str, str]]:
    vector = np.array(embed([question])[0])
    return conn.execute("SELECT chunk_id, text FROM chunks ORDER BY embedding <=> %s LIMIT %s", (vector, k)).fetchall()


def answer(question: str, k: int = 5) -> dict:
    start = time.perf_counter()
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        passages = retrieve(conn, question, k)
    context = "\n\n".join(f"[{chunk_id}]\n{text}" for chunk_id, text in passages)
    reply = ask_structured(SYSTEM, f"Passages:\n{context}\n\nQuestion: {question}", Answer)
    parsed = reply["parsed"]

    # The model can write an id that looks real but wasn't given to it. A set difference catches that.
    retrieved = {chunk_id for chunk_id, _ in passages}
    bad = set(parsed.citations) - retrieved
    citations_ok = not bad and (bool(parsed.citations) or not parsed.answerable)  # a claim needs a source
    cost = (reply["input_tokens"] * PRICE_PER_M["input"] + reply["output_tokens"] * PRICE_PER_M["output"]) / 1e6
    return {
        "question": question, "answer": parsed.answer, "citations": parsed.citations,
        "answerable": parsed.answerable, "citations_ok": citations_ok, "bad_citations": sorted(bad),
        "retrieved": [chunk_id for chunk_id, _ in passages],
        "latency_s": round(time.perf_counter() - start, 2), "cost_usd": round(cost, 5),
    }


if __name__ == "__main__":
    questions = sys.argv[1:] or [
        "Which supplier do NVIDIA and AMD share?",
        "What does Intel say about its foundry plans?",
        "What was Apple's revenue last year?",  # out of scope: should say it can't answer
    ]
    for q in questions:
        r = answer(q)
        print(f"\nQ: {r['question']}\nA: {r['answer']}")
        print(f"   answerable={r['answerable']}  cited={len(r['citations'])}  citations_ok={r['citations_ok']}  "
              f"{r['latency_s']}s  ${r['cost_usd']}")
        print(f"   retrieved: {', '.join(c.split('-')[0] + '…' + c[-6:] for c in r['retrieved'])}")
