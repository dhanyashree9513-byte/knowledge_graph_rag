"""Readiness check: are both databases reachable?

Run: uv run python src/check_dbs.py
"""

import os

import psycopg
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


def check_neo4j() -> str:
    uri = os.getenv("NEO4J_URI")
    auth = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    try:
        with GraphDatabase.driver(uri, auth=auth) as driver:
            driver.verify_connectivity()
            driver.execute_query("RETURN 1 AS ok")
        return "✅ Neo4j: connected"
    except Exception as e:
        return f"❌ Neo4j: {type(e).__name__}: {str(e).splitlines()[0]}"


def check_postgres() -> str:
    try:
        with psycopg.connect(os.getenv("DATABASE_URL"), connect_timeout=5) as conn:
            row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
        if row is None:
            return "❌ Postgres: connected, but the vector extension is missing"
        return f"✅ Postgres: pgvector {row[0]}"
    except Exception as e:
        return f"❌ Postgres: {type(e).__name__}: {str(e).splitlines()[0]}"


if __name__ == "__main__":
    print(check_neo4j())
    print(check_postgres())
