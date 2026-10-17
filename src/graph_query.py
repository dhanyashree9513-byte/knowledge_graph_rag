"""Graph path: question -> (template, names) -> node ids -> fixed Cypher -> facts with chunk ids.

The LLM only PICKS a template and names the things in the question. It never writes Cypher:
every query below is written and tested by us, and values travel as $parameters.

Run: uv run python src/graph_query.py "Which supplier do NVIDIA and AMD share?"
"""

import os
import sys
from typing import Literal

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pydantic import BaseModel

from llm import ask_structured, embed
from resolve import key

load_dotenv()
MIN_SIMILARITY = 0.80  # embedding fallback for names with no exact alias match; below this, "not in the graph"
TYPE_PREFERENCE = ["Company", "Person", "Product", "Risk", "Location", "Technology", "Market"]  # for names under 2 types

# Every path is returned edge by edge as (source, relation, target, chunk_ids), plus x = the node that answers the step,
# so all templates share one output shape and a later step can start from x.
EDGES = ("UNWIND relationships(p) AS r RETURN DISTINCT x.name AS x, elementId(x) AS x_id, startNode(r).name AS src, "
         "type(r) AS rel, endNode(r).name AS dst, r.chunk_ids AS chunk_ids LIMIT 60")
ONE = "WHERE elementId(a) = $ids[0] "
# "shared" = connected to EVERY node in $ids (2 or more): count the matches per x, keep x only if it matched all.
SHARED = "WITH x, count(DISTINCT c) AS n WHERE n = size($ids) "
TEMPLATES = {
    # one entity (A)
    "suppliers_of":     ("Who supplies A", "MATCH p = (x)-[:SUPPLIES]->(a) " + ONE),
    "customers_of":     ("Who A supplies / sells to", "MATCH p = (a)-[:SUPPLIES]->(x) " + ONE),
    "competitors_of":   ("Who competes with A", "MATCH p = (a)-[:COMPETES_WITH]-(x) " + ONE),
    "products_of":      ("What A makes", "MATCH p = (a)-[:MAKES]->(x) " + ONE),
    "risks_of":         ("Risks A faces", "MATCH p = (a)-[:FACES]->(x) " + ONE),
    "locations_of":     ("Where A operates or is headquartered", "MATCH p = (a)-[:OPERATES_IN]->(x) " + ONE),
    "executives_of":    ("Executives of A", "MATCH p = (x)-[:EXECUTIVE_OF]->(a) " + ONE),
    "employer_of":      ("Company where person A is an executive", "MATCH p = (a)-[:EXECUTIVE_OF]->(x) " + ONE),
    "maker_of":         ("Company that makes product A", "MATCH p = (x)-[:MAKES]->(a) " + ONE),
    "neighbors":        ("Anything directly connected to A", "MATCH p = (a)-[]-(x) " + ONE),
    # two or more entities: facts from different filings joined in one query
    "shared_suppliers":   ("Who supplies ALL of the listed companies", "MATCH (x)-[:SUPPLIES]->(c) WHERE elementId(c) IN $ids " + SHARED +
                           "MATCH p = (x)-[:SUPPLIES]->(c) WHERE elementId(c) IN $ids "),
    "shared_competitors": ("Who competes with ALL of the listed companies", "MATCH (x)-[:COMPETES_WITH]-(c) WHERE elementId(c) IN $ids AND NOT elementId(x) IN $ids " + SHARED +
                           "MATCH p = (x)-[:COMPETES_WITH]-(c) WHERE elementId(c) IN $ids "),
    "shared_risks":       ("Risks ALL of the listed companies face", "MATCH (c)-[:FACES]->(x) WHERE elementId(c) IN $ids " + SHARED +
                           "MATCH p = (c)-[:FACES]->(x) WHERE elementId(c) IN $ids "),
    "connection":         ("How A and B are connected (up to 2 steps)", "MATCH p = allShortestPaths((a)-[*..2]-(x)) WHERE elementId(a) = $ids[0] AND elementId(x) = $ids[1] "),
}
MAX_STEPS, MAX_CARRY = 3, 3  # at most 3 chained steps; at most 3 answers carried from one step into the next

SYSTEM = """Plan graph queries that answer the question. Use 1 step when one template answers it.
Use 2-3 steps when the question describes a thing instead of naming it ("the company whose CFO is X", "the company that acquired Y"):
the first step finds that thing, and a later step uses it by writing "#1" (the answers of step 1) as an entity.
Entities are names as written in the question, or "#n". Shared templates take 2 or more entities.
Templates:
""" + "\n".join(f"- {name}: {desc}" for name, (desc, _) in TEMPLATES.items()) + """
Example: "What risks does the company that makes Radeon GPUs face?" ->
  1. maker_of ["Radeon"]   2. risks_of ["#1"]"""


class Step(BaseModel):
    template: Literal[tuple(TEMPLATES)]
    entities: list[str]


class Plan(BaseModel):
    steps: list[Step]


class Graph:
    def __init__(self):
        self.driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")))
        rows = self.driver.execute_query("MATCH (n) RETURN elementId(n) AS id, labels(n)[0] AS type, n.name AS name, n.aliases AS aliases").records
        self.nodes = [dict(r) for r in rows]
        self.by_key = {}  # normalized spelling -> node; Company wins when a name exists under two types
        for n in sorted(self.nodes, key=lambda n: -TYPE_PREFERENCE.index(n["type"])):
            for alias in n["aliases"] + [n["name"]]:
                self.by_key[key(alias)] = n
        self.vectors = None  # node-name embeddings, built only if an exact match ever fails

    def link(self, name: str) -> dict | None:
        """Question name -> graph node: exact alias match first, then nearest name by meaning."""
        if node := self.by_key.get(key(name)):
            return node | {"match": "alias"}
        if self.vectors is None:
            v = np.array(embed([n["name"] for n in self.nodes]))
            self.vectors = v / np.linalg.norm(v, axis=1, keepdims=True)
        q = np.array(embed([name])[0])
        scores = self.vectors @ (q / np.linalg.norm(q))
        best = int(scores.argmax())
        return self.nodes[best] | {"match": f"embedding {scores[best]:.2f}"} if scores[best] >= MIN_SIMILARITY else None

    def run_step(self, step: Step, answers: list[list[dict]]) -> tuple[list[dict], str | None]:
        """Run one template. Entities are linked names, or "#n" = the answer nodes of step n."""
        nodes = []
        for name in step.entities:
            if name.startswith("#") and name[1:].isdigit() and 0 < int(name[1:]) <= len(answers):
                nodes += answers[int(name[1:]) - 1][:MAX_CARRY]
            elif node := self.link(name):
                nodes.append(node)
            else:
                return [], f"{name!r} is not in the graph"
        if not nodes:
            return [], "no entities"
        cypher = TEMPLATES[step.template][1] + EDGES
        shared = "$ids[0]" not in cypher  # shared templates take all nodes at once; others run once per node
        groups = [[n["id"] for n in nodes]] if shared else [[n["id"]] for n in nodes]
        if step.template == "connection":
            groups = [[nodes[0]["id"], nodes[1]["id"]]] if len(nodes) >= 2 else []
        return [dict(r) for ids in groups for r in self.driver.execute_query(cypher, ids=ids).records], None

    def query(self, question: str) -> dict:
        reply = ask_structured(SYSTEM, question, Plan)
        steps = reply["parsed"].steps[:MAX_STEPS]
        result = {"template": " > ".join(s.template for s in steps), "steps": [s.model_dump() for s in steps],
                  "facts": [], "tokens": (reply["input_tokens"], reply["output_tokens"])}
        answers, facts_by_step = [], []
        for step in steps:
            rows, error = self.run_step(step, answers)
            if error:
                return result | {"error": error}
            answers.append(list({r["x_id"]: {"id": r["x_id"], "name": r["x"]} for r in rows}.values()))
            facts_by_step.append([{"sentence": f"{r['src']} {r['rel']} {r['dst']}", "chunk_ids": r["chunk_ids"]} for r in rows])
        # last step first: those facts answer the question; earlier steps explain how we got there
        result["facts"] = [f for facts in reversed(facts_by_step) for f in facts]
        return result

    def close(self):
        self.driver.close()


if __name__ == "__main__":
    questions = sys.argv[1:] or [
        "Which supplier do NVIDIA and AMD share?",
        "Who does Taiwan Semiconductor supply?",
        "Which risks do NVIDIA and Intel both face?",
        "Who are NVIDIA's executives?",
        "Who competes with both AMD and Intel?",
        "Which risk do NVIDIA, AMD and Intel all face?",
        "What does the company that makes Ryzen processors say about its supply of wafers?",
        "Who supplies Apple?",  # not in the graph
    ]
    graph = Graph()
    for q in questions:
        r = graph.query(q)
        plan = "  ".join(f"{i}. {s['template']} {s['entities']}" for i, s in enumerate(r["steps"], 1))
        print(f"\nQ: {q}\n   {plan}  → {len(r['facts'])} facts {r.get('error', '')}")
        for f in r["facts"][:6]:
            print(f"   {f['sentence']:<45} {', '.join(c[:4] + '…' + c[-6:] for c in f['chunk_ids'][:3])}")
    graph.close()
