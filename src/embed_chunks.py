"""Store every chunk with its embedding in Postgres (pgvector), plus an HNSW index for fast search.

Run: uv run python src/embed_chunks.py   (needs: docker compose up -d)
Only chunks not yet in the table are embedded, so a re-run costs nothing.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from llm import embed

load_dotenv()
CHUNKS_PATH = Path("data/chunks.jsonl")
GRAPH_PATH = Path("data/graph.json")
DIMENSIONS = 1536  # text-embedding-3-small

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   text PRIMARY KEY,         -- the same id the graph edges cite
    doc_id     text NOT NULL,
    ticker     text NOT NULL,
    section    text NOT NULL,            -- item_1 (Business) or item_1a (Risk Factors)
    filed      date NOT NULL,
    entity_ids text[] NOT NULL,          -- graph nodes mentioned in this chunk, e.g. 'Company:TSMC'
    text       text NOT NULL,
    embedding  vector({DIMENSIONS}) NOT NULL
);
-- HNSW: an approximate nearest-neighbour index; cosine ops to match how the embeddings are compared.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
"""


def entity_ids_by_chunk() -> dict[str, list[str]]:
    """Which graph nodes each chunk talks about: lets a text search result jump into the graph."""
    ids = defaultdict(set)
    for e in json.loads(GRAPH_PATH.read_text())["edges"]:
        for chunk_id in e["chunk_ids"]:
            ids[chunk_id] |= {f"{e['source_type']}:{e['source']}", f"{e['target_type']}:{e['target']}"}
    return {chunk_id: sorted(v) for chunk_id, v in ids.items()}


def search(conn, question: str, k: int = 3) -> list[tuple]:
    vector = np.array(embed([question])[0])
    return conn.execute(
        "SELECT chunk_id, round((1 - (embedding <=> %s))::numeric, 3), left(text, 90) FROM chunks "
        "ORDER BY embedding <=> %s LIMIT %s", (vector, vector, k)).fetchall()


def main() -> None:
    chunks = [json.loads(line) for line in CHUNKS_PATH.open()]
    entities = entity_ids_by_chunk()
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)  # lets psycopg send numpy arrays as pgvector values
        conn.execute(SCHEMA)

        stored = {row[0] for row in conn.execute("SELECT chunk_id FROM chunks")}
        new = [c for c in chunks if c["chunk_id"] not in stored]
        tokens = sum(len(c["text"]) for c in new) / 4  # ~4 characters per token
        print(f"{len(chunks)} chunks, {len(new)} new; ~{tokens:,.0f} tokens ≈ ${tokens * 0.02 / 1e6:.4f}")

        if new:
            vectors = embed([c["text"] for c in new])
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (chunk_id) DO NOTHING",
                    [(c["chunk_id"], c["doc_id"], c["ticker"], c["section"], c["doc_id"][-10:],  # date is in the id
                      entities.get(c["chunk_id"], []), c["text"], np.array(v)) for c, v in zip(new, vectors)])

        print("rows in table:", conn.execute("SELECT count(*) FROM chunks").fetchone()[0])
        question = sys.argv[1] if len(sys.argv) > 1 else "Who manufactures NVIDIA's wafers?"
        print(f"\nsearch: {question}")
        for row in search(conn, question):
            print(" ", *row)


if __name__ == "__main__":
    main()
