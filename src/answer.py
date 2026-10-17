"""The full pipeline: route -> evidence (graph facts and/or text chunks) -> cited answer -> citation check.

Run: uv run python src/answer.py "Which supplier do NVIDIA and AMD share?"
"""

import os
import sys
import time

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from pydantic import BaseModel

from baseline import retrieve
from extract import PRICE_PER_M
from graph_query import Graph
from llm import ask_structured
from router import route

load_dotenv()
MAX_GRAPH_CHUNKS = 8  # source chunks fetched for graph facts (the first facts' chunks first)
NOT_FOUND = "The filings don't contain enough information to answer this."

SYSTEM = """You answer questions about NVIDIA, AMD and Intel using ONLY the evidence given.
The evidence has graph facts (short lines extracted from the filings) and passages (the filing text).
- Answer as a list of short claims. Each claim lists the passage ids that support it, e.g. NVDA-10K-2026-02-25-6637efa5bd47.
- Only use ids that appear in the evidence. A graph fact is backed by the passages listed after it.
- If the evidence doesn't answer the question, set answerable=false. Never use outside knowledge."""

_graph = None  # one Neo4j connection + alias index, reused across questions


def graph() -> Graph:
    global _graph
    _graph = _graph or Graph()
    return _graph


class Claim(BaseModel):
    text: str
    chunk_ids: list[str]


class Reply(BaseModel):
    claims: list[Claim]  # every claim carries its own sources, so "every claim cited" can be checked by code
    answerable: bool


def fetch_chunks(conn, chunk_ids: list[str]) -> list[tuple[str, str]]:
    rows = dict(conn.execute("SELECT chunk_id, text FROM chunks WHERE chunk_id = ANY(%s)", (chunk_ids,)).fetchall())
    return [(c, rows[c]) for c in chunk_ids if c in rows]


def gather_evidence(question: str, chosen_route: str) -> dict:
    """Facts from the graph and/or passages from text search. Every item keeps its chunk id."""
    facts, passages, info = [], [], {}
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        if chosen_route in ("graph", "both"):
            g = graph().query(question)
            facts = g["facts"]
            info = {"template": g["template"], "graph_error": g.get("error"), "graph_tokens": g["tokens"]}
            # round-robin: each fact's 1st chunk, then each fact's 2nd... so one busy fact can't take every slot
            ranked = (f["chunk_ids"][i] for i in range(max((len(f["chunk_ids"]) for f in facts), default=0))
                      for f in facts if i < len(f["chunk_ids"]))
            chunk_ids = list(dict.fromkeys(ranked))[:MAX_GRAPH_CHUNKS]
            passages = fetch_chunks(conn, chunk_ids)
        evidence = {"facts": facts, "passages": passages, **info}
        if chosen_route in ("vector", "both") or not facts:  # graph found nothing -> text search as a fallback
            add_text_search(conn, question, evidence)
    return evidence


def add_text_search(conn, question: str, evidence: dict) -> None:
    seen = {c for c, _ in evidence["passages"]}
    evidence["passages"] += [p for p in retrieve(conn, question, 5) if p[0] not in seen]
    evidence["used_vector"] = True


def format_evidence(facts: list[dict], passages: list[tuple[str, str]]) -> str:
    shown = {chunk_id for chunk_id, _ in passages}  # only list ids whose text is included, so every listed id is citable
    lines = ["Graph facts:"] if facts else []
    for f in facts[:40]:
        ids = [c for c in f["chunk_ids"] if c in shown]
        lines.append(f"- {f['sentence']}  (passages: {', '.join(ids) if ids else 'not included'})")
    lines += ["", "Passages:"] + [f"[{chunk_id}]\n{text}\n" for chunk_id, text in passages]
    return "\n".join(lines)


def write(question: str, evidence: dict, tokens: list) -> tuple:
    """LLM answer from the evidence, with the citation check and one retry. Returns (reply, ok, attempts, cited ids)."""
    allowed = {chunk_id for chunk_id, _ in evidence["passages"]}
    prompt = f"Evidence:\n{format_evidence(evidence['facts'], evidence['passages'])}\n\nQuestion: {question}"
    for attempt in (1, 2):  # one retry: tell the model which ids were not in the evidence
        reply = ask_structured(SYSTEM, prompt, Reply)
        tokens.append((reply["input_tokens"], reply["output_tokens"]))
        r = reply["parsed"]
        cited = {c for claim in r.claims for c in claim.chunk_ids}
        bad = cited - allowed
        uncited = [claim.text for claim in r.claims if not claim.chunk_ids]
        ok = not bad and not uncited and (bool(r.claims) or not r.answerable)
        if ok:
            break
        prompt += f"\n\nYour previous answer cited ids that are not in the evidence or cited nothing: {sorted(bad)}, or left claims uncited: {uncited}. Answer again."
    return r, ok, attempt, cited


def answer(question: str) -> dict:
    start = time.perf_counter()
    decision = route(question)
    evidence = gather_evidence(question, decision["route"])
    tokens = [decision["tokens"], evidence.get("graph_tokens", (0, 0))]
    r, ok, attempt, cited = write(question, evidence, tokens)

    # Graph facts were found but didn't answer it (wrong template, chain broke): add text search and try once more.
    if not (ok and r.answerable) and not evidence.get("used_vector"):
        with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
            register_vector(conn)
            add_text_search(conn, question, evidence)
        r, ok, attempt, cited = write(question, evidence, tokens)
        evidence["text_retry"] = True

    cost = sum(i * PRICE_PER_M["input"] + o * PRICE_PER_M["output"] for i, o in tokens) / 1e6
    return {
        "question": question,
        "answer": " ".join(f"{c.text} [{', '.join(c.chunk_ids)}]" for c in r.claims) if ok and r.answerable else NOT_FOUND,
        "citations": sorted(cited) if ok and r.answerable else [],
        "route": decision["route"], "template": evidence.get("template"),
        "facts": len(evidence["facts"]), "passages": len(evidence["passages"]),
        "attempts": attempt, "citations_ok": ok, "text_retry": evidence.get("text_retry", False),
        "latency_s": round(time.perf_counter() - start, 2), "cost_usd": round(cost, 5),
    }


if __name__ == "__main__":
    questions = sys.argv[1:] or [
        "Which supplier do NVIDIA and AMD share?",
        "What does Intel say about its foundry plans?",
        "Who competes with both AMD and Intel?",
        "What was Apple's revenue last year?",
    ]
    for q in questions:
        r = answer(q)
        print(f"\nQ: {q}\nA: {r['answer']}")
        print(f"   route={r['route']} template={r['template']} facts={r['facts']} passages={r['passages']} "
              f"cited={len(r['citations'])} ok={r['citations_ok']} attempts={r['attempts']} {r['latency_s']}s ${r['cost_usd']}")
