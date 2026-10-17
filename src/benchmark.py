"""Run both systems on eval/benchmark.json and grade them the same way.

Run: uv run python src/benchmark.py                               (main set, 58 questions)
     uv run python src/benchmark.py eval/benchmark_holdout.json    (19 questions never used while fixing the system)
Grading is by keywords, not by an LLM: an answer is right if it contains one spelling from every `must` group
(chunk ids are removed first, so digits inside ids can't count). Out-of-scope questions are right only if refused.
Writes eval/results/<system>.jsonl and prints accuracy by question type, p95 latency and cost per question.
"""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import answer as graph_rag
import baseline

BENCH_PATH = Path("eval/benchmark.json")
RESULTS_DIR = Path("eval/results")
CHUNK_ID = re.compile(r"\[[^\]]*\]|\b(?:NVDA|AMD|INTC)-10K-[\w-]+")


def grade(item: dict, text: str, refused: bool) -> bool:
    if item["type"] == "out_of_scope":
        return refused
    text = CHUNK_ID.sub(" ", text).lower()
    return not refused and all(any(spelling in text for spelling in group) for group in item["must"])


SYSTEMS = {
    "baseline": lambda q: (r := baseline.answer(q)) | {"refused": not r["answerable"]},
    "graph_rag": lambda q: (r := graph_rag.answer(q)) | {"refused": r["answer"] == graph_rag.NOT_FOUND},
}


def run(name: str, items: list[dict], tag: str) -> list[dict]:
    def one(item):
        r = SYSTEMS[name](item["question"])
        return item | {"answer": r["answer"], "refused": r["refused"], "correct": grade(item, r["answer"], r["refused"]),
                       "route": r.get("route"), "latency_s": r["latency_s"], "cost_usd": r["cost_usd"]}
    with ThreadPoolExecutor(4) as pool:  # 4 at a time: faster run, latency per question still measured per call
        results = list(pool.map(one, items))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{tag}_{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    return results


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else BENCH_PATH
    items = json.loads(path.read_text())
    graph_rag.graph()  # open the Neo4j connection once, before the threads start
    results = {name: run(name, items, path.stem) for name in SYSTEMS}

    types = list(dict.fromkeys(i["type"] for i in items))
    print(f"\n{'type':<13} {'n':>3}  {'baseline':>9}  {'graph_rag':>9}")
    for t in types + ["ALL"]:
        rows = {s: [r for r in rs if t in (r["type"], "ALL")] for s, rs in results.items()}
        acc = {s: np.mean([r["correct"] for r in rs]) for s, rs in rows.items()}
        print(f"{t:<13} {len(rows['baseline']):>3}  {acc['baseline']:>9.0%}  {acc['graph_rag']:>9.0%}")
    for s, rs in results.items():
        lat = [r["latency_s"] for r in rs]
        print(f"{s:<10} p50 {np.percentile(lat, 50):.1f}s  p95 {np.percentile(lat, 95):.1f}s  "
              f"cost/question ${np.mean([r['cost_usd'] for r in rs]):.4f}  total ${sum(r['cost_usd'] for r in rs):.3f}")

    print("\nwhere the two systems disagree:")
    for b, g in zip(results["baseline"], results["graph_rag"]):
        if b["correct"] != g["correct"]:
            print(f"  {'graph ✓' if g['correct'] else 'base  ✓'}  [{b['type']}] {b['question']}")


if __name__ == "__main__":
    main()
