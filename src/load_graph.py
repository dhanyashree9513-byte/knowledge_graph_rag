"""Load data/graph.json into Neo4j. Safe to run again: MERGE finds existing nodes/edges instead of duplicating them.

Run: uv run python src/load_graph.py   (needs: docker compose up -d)
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import get_args

from dotenv import load_dotenv
from neo4j import GraphDatabase

from extract import EntityType, RelationType

load_dotenv()
GRAPH_PATH = Path("data/graph.json")
LABELS, RELATIONS = set(get_args(EntityType)), set(get_args(RelationType))


def checked(value: str, allowed: set[str]) -> str:
    """Labels and relation types can't be $parameters in Cypher, so they go into the query text.
    Only names from the ontology are allowed in, so nothing from the data can change the query."""
    if value not in allowed:
        raise ValueError(f"{value!r} is not in the ontology")
    return value


def load(driver, graph: dict) -> None:
    # One unique name per label: makes MERGE fast (index lookup) and makes a duplicate node impossible.
    for label in LABELS:
        driver.execute_query(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.name IS UNIQUE")

    nodes_by_label = defaultdict(list)
    for n in graph["nodes"]:
        nodes_by_label[checked(n["type"], LABELS)].append({"name": n["name"], "aliases": n["aliases"]})
    for label, rows in nodes_by_label.items():  # UNWIND = one round trip per label, not per node
        driver.execute_query(
            f"UNWIND $rows AS row MERGE (n:{label} {{name: row.name}}) SET n.aliases = row.aliases", rows=rows)

    edges_by_shape = defaultdict(list)
    for e in graph["edges"]:
        shape = (checked(e["source_type"], LABELS), checked(e["relation"], RELATIONS), checked(e["target_type"], LABELS))
        edges_by_shape[shape].append(e)
    for (src, rel, dst), rows in edges_by_shape.items():
        # SET (not append) chunk_ids: a second run writes the same list, so nothing grows.
        driver.execute_query(
            f"UNWIND $rows AS row "
            f"MATCH (a:{src} {{name: row.source}}), (b:{dst} {{name: row.target}}) "
            f"MERGE (a)-[r:{rel}]->(b) "
            f"SET r.chunk_ids = row.chunk_ids, r.confidence = row.confidence, r.evidence = row.evidence",
            rows=rows)


def counts(driver) -> tuple[int, int]:
    nodes = driver.execute_query("MATCH (n) RETURN count(n) AS c").records[0]["c"]
    edges = driver.execute_query("MATCH ()-[r]->() RETURN count(r) AS c").records[0]["c"]
    return nodes, edges


if __name__ == "__main__":
    graph = json.loads(GRAPH_PATH.read_text())
    auth = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    with GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=auth) as driver:
        print(f"graph.json: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
        for run in (1, 2):  # load twice: the counts must not change
            load(driver, graph)
            print("run {}: Neo4j has {} nodes, {} edges".format(run, *counts(driver)))
