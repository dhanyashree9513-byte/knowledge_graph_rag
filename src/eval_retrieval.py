"""How good is text search? Measured on 25 hand-labelled questions (eval/retrieval_gold.json).

Run: uv run python src/eval_retrieval.py
- hit@k:    did ANY right chunk make the top k? (enough for a 1-hop answer)
- recall@k: what share of ALL right chunks made the top k? (a 2-hop answer needs all of them)
- ef_search: how many candidates HNSW looks at. Too low = fast but misses chunks exact search would find.
"""

import json
import os
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from llm import embed

load_dotenv()
GOLD_PATH = Path("eval/retrieval_gold.json")
KS = (1, 3, 5, 10)


def top_ids(conn, vector, k: int) -> list[str]:
    return [r[0] for r in conn.execute("SELECT chunk_id FROM chunks ORDER BY embedding <=> %s LIMIT %s", (vector, k))]


def main() -> None:
    gold = json.loads(GOLD_PATH.read_text())
    vectors = [np.array(v) for v in embed([g["question"] for g in gold])]  # one API call for all 25

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)

        # 1) Retrieval quality with EXACT search (index off): the best this embedding model can do.
        conn.execute("SET enable_indexscan = off")
        exact = [top_ids(conn, v, max(KS)) for v in vectors]
        print(f"{'k':>3}  hit@k  recall@k   (exact search, {len(gold)} questions)")
        for k in KS:
            hits = [bool(set(g["gold"]) & set(ids[:k])) for g, ids in zip(gold, exact)]
            recall = [len(set(g["gold"]) & set(ids[:k])) / len(g["gold"]) for g, ids in zip(gold, exact)]
            print(f"{k:>3}  {np.mean(hits):.2f}   {np.mean(recall):.2f}")

        print("\nmissed at k=5:")
        for g, ids in zip(gold, exact):
            if not set(g["gold"]) & set(ids[:5]):
                print(f"  ✗ {g['question']}")
        two_hop = [(g, ids) for g, ids in zip(gold, exact) if g["type"] == "2-hop"]
        for g, ids in two_hop:
            print(f"  2-hop: {g['question']} → found {len(set(g['gold']) & set(ids[:5]))} of {len(g['gold'])} chunks in top 5")

        # 2) HNSW vs exact: does the index return the same top 10? (index forced on)
        conn.execute("SET enable_indexscan = on")
        conn.execute("SET enable_seqscan = off")
        print("\nef_search  overlap with exact top-10")
        for ef in (10, 20, 40, 100):
            conn.execute(f"SET hnsw.ef_search = {ef}")
            overlap = [len(set(top_ids(conn, v, 10)) & set(ids)) / 10 for v, ids in zip(vectors, exact)]
            print(f"{ef:>9}  {np.mean(overlap):.3f}")


if __name__ == "__main__":
    main()
