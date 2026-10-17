"""Router: decide where to look for evidence. graph = connections between things, vector = what a report says, both = unsure.

Run: uv run python src/router.py   (routes a test set and prints accuracy)
Every decision is appended to data/logs/router.jsonl, so wrong routes can be found and turned into new examples.
"""

import json
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from llm import ask_structured

LOG_PATH = Path("data/logs/router.jsonl")
MIN_CONFIDENCE = 0.7  # below this, search both: a wasted lookup is cheaper than a missed answer

SYSTEM = """You route questions about NVIDIA, AMD and Intel 10-K filings to an evidence source.

graph: the answer is a connection between named things, stored as facts like
  Company -SUPPLIES-> Company, Company -COMPETES_WITH-> Company, Company -MAKES-> Product,
  Company -FACES-> Risk, Company -OPERATES_IN-> Location, Person -EXECUTIVE_OF-> Company.
  Especially questions that join facts across companies ("shared", "both", "which ... also").
vector: the answer is an explanation, plan, number, date or description written in a report.
both: the question needs a connection AND an explanation, or you are not sure.

Examples:
Q: Which supplier do NVIDIA and AMD share? -> graph
Q: Who competes with both AMD and Intel? -> graph
Q: Who is NVIDIA's chief financial officer? -> graph
Q: Which risks do all three companies face? -> graph
Q: What does Intel say about its foundry plans? -> vector
Q: When did NVIDIA acquire Mellanox? -> vector
Q: How do export controls affect NVIDIA's China revenue? -> vector
Q: Which foundries does AMD rely on, and what happens if they fail to deliver? -> both

Give a confidence from 0 to 1 that your route is right."""


class Route(BaseModel):
    route: Literal["graph", "vector", "both"]
    confidence: float


def route(question: str) -> dict:
    start = time.perf_counter()
    reply = ask_structured(SYSTEM, question, Route)
    parsed = reply["parsed"]
    final = parsed.route if parsed.confidence >= MIN_CONFIDENCE else "both"  # low confidence -> fall back
    decision = {"question": question, "model_route": parsed.route, "confidence": parsed.confidence,
                "route": final, "ms": round((time.perf_counter() - start) * 1000), "ts": time.time(),
                "tokens": (reply["input_tokens"], reply["output_tokens"])}
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(decision) + "\n")
    return decision


# Held-out test set: none of these are in the prompt examples. A route is OK if it reaches the needed source.
TEST = [
    ("Which companies supply Intel?", "graph"),
    ("Which companies does NVIDIA compete with in networking?", "graph"),
    ("Which executives work at NVIDIA?", "graph"),
    ("What products does AMD make for data centers?", "graph"),
    ("Which risks do NVIDIA and Intel both face?", "graph"),
    ("Where does Intel operate fabs?", "graph"),
    ("What agreement did AMD sign with OpenAI?", "vector"),
    ("What is the production status of Intel 18A?", "vector"),
    ("Why did Intel discontinue its planned expansion in Germany?", "vector"),
    ("What instability issues affected Intel Core desktop processors?", "vector"),
    ("How does NVIDIA describe the demand for its data center products?", "vector"),
    ("What will Intel do if it can't win customers for Intel 14A?", "vector"),
    ("Who supplies TSMC's customers, and what risks does that create for AMD?", "both"),
    ("Which companies compete with NVIDIA, and how does NVIDIA say it differentiates?", "both"),
]

if __name__ == "__main__":
    correct = 0
    for question, expected in TEST:
        d = route(question)
        ok = d["route"] == expected or d["route"] == "both"  # "both" always reaches the evidence (just costs more)
        exact = d["route"] == expected
        correct += exact
        print(f"{'✓' if exact else ('~' if ok else '✗')} {d['route']:<6} (model {d['model_route']}, {d['confidence']:.2f}, {d['ms']} ms)  "
              f"want {expected:<6} | {question}")
    print(f"\nexact: {correct}/{len(TEST)}   (~ = fell back to both: right evidence, extra cost)   log: {LOG_PATH}")
