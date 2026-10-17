"""Entity resolution: merge different spellings of the same entity into one node.

Run: uv run python src/resolve.py
Input: data/extractions.jsonl   Output: data/graph.json (resolved nodes + deduplicated edges)
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from llm import embed

IN_PATH = Path("data/extractions.jsonl")
OUT_PATH = Path("data/graph.json")
THRESHOLD = 0.90  # cosine similarity above which two names (same type) are merged; tuned by reading the printed pairs

# Legal suffixes that don't change which company it is: "Micron Technology, Inc." == "Micron Technology".
SUFFIXES = r"(inc|incorporated|corp|corporation|co|company|ltd|limited|llc|plc|holdings|ag|sa|nv)"


def normalize(name: str) -> str:
    """Lowercase, drop punctuation, legal suffixes and a leading 'the'."""
    key = re.sub(r"[^\w\s&-]", " ", name.lower())
    key = re.sub(rf"(\s+{SUFFIXES})+\s*$", "", key.strip())
    key = re.sub(r"^the\s+", "", key)
    return re.sub(r"\s+", " ", key).strip()


# Hand-checked pairs whose embeddings score below THRESHOLD ("samsung" vs "samsung electronics" = 0.85).
# Also used when matching names in a question to nodes, so full legal names typed by users find the node.
ALIASES = {
    "samsung electronics": "samsung",
    "taiwan semiconductor manufacturing": "tsmc", "taiwan semiconductor": "tsmc",
    "advanced micro devices": "amd",
}

# Words that don't change which product it is: "AMD EPYC" == "AMD EPYC processors".
GENERIC = {"processor", "series", "software", "gpu", "cpu", "product", "platform"}


def key(name: str) -> str:
    """The name used for matching: normalized, then mapped through the alias table."""
    k = normalize(name)
    return ALIASES.get(k, k)


def numbers(name: str) -> list[str]:
    return re.findall(r"\d+", name)


def only_generic_differences(a: str, b: str) -> bool:
    """True if the names differ only by plurals or GENERIC words. Blocks "ryzen" vs "ryzen pro", "xbox series s" vs "x"."""
    words = lambda name: {w[:-1] if len(w) > 3 and w.endswith("s") and w != "series" else w for w in name.split()}
    return (words(a) ^ words(b)) <= GENERIC


class UnionFind:
    """Tracks which names belong to the same group; merging two names merges their groups."""

    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        while self.parent[item] != item:
            item = self.parent[item]
        return item

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def resolve(names: list[str], counts: Counter, entity_type: str) -> dict[str, str]:
    """Map every original name to a canonical name."""
    keys = sorted({key(n) for n in names})
    groups = UnionFind(keys)  # 1) same normalized key = same entity (free, exact)

    # 2) embeddings catch what normalization can't ("Samsung" vs "Samsung Electronics").
    # Risks come from a fixed list, so they never need merging.
    if entity_type != "Risk" and len(keys) > 1:
        vectors = np.array(embed(keys))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        similarity = vectors @ vectors.T  # cosine similarity of every pair
        for i, j in zip(*np.triu_indices(len(keys), k=1)):
            # Names differing only by a number embed almost identically but are different things ("Zen 4" vs "Zen 5").
            # Product names that look alike are often different models ("Threadripper" vs "Threadripper PRO").
            product_ok = entity_type != "Product" or only_generic_differences(keys[i], keys[j])
            if similarity[i, j] >= THRESHOLD and numbers(keys[i]) == numbers(keys[j]) and product_ok:
                groups.union(keys[i], keys[j])
                print(f"  merge {entity_type:<10} {similarity[i, j]:.2f}  {keys[i]!r} + {keys[j]!r}")

    # Canonical name = the most frequently used original spelling in the group (ties: the shortest).
    members = defaultdict(list)
    for name in names:
        members[groups.find(key(name))].append(name)
    mapping = {}
    for group in members.values():
        canonical = max(group, key=lambda n: (counts[n], -len(n)))
        for name in group:
            mapping[name] = canonical
    return mapping


def main() -> None:
    records = [json.loads(line) for line in IN_PATH.open()]
    relations = [r | {"chunk_id": rec["chunk_id"]} for rec in records for r in rec["relations"]]

    counts = Counter()
    names_by_type = defaultdict(set)
    for r in relations:
        for side in ("source", "target"):
            counts[r[side]] += 1
            names_by_type[r[f"{side}_type"]].add(r[side])

    canonical = {}  # (type, original name) -> canonical name
    for entity_type, names in sorted(names_by_type.items()):
        for name, canon in resolve(sorted(names), counts, entity_type).items():
            canonical[(entity_type, name)] = canon

    # Nodes: one per (type, canonical name), with every spelling kept as an alias.
    nodes = defaultdict(set)
    for (entity_type, name), canon in canonical.items():
        nodes[(entity_type, canon)].add(name)

    # Edges: the same fact from several chunks becomes ONE edge listing all its source chunks.
    edges = {}
    for r in relations:
        src = (r["source_type"], canonical[(r["source_type"], r["source"])])
        dst = (r["target_type"], canonical[(r["target_type"], r["target"])])
        if src == dst:  # merging can turn an edge into a self-loop
            continue
        key = (src, r["relation"], dst)
        edge = edges.setdefault(key, {"chunk_ids": set(), "confidence": 0.0, "evidence": r["evidence"]})
        edge["chunk_ids"].add(r["chunk_id"])
        edge["confidence"] = max(edge["confidence"], r["confidence"])

    graph = {
        "nodes": [{"type": t, "name": n, "aliases": sorted(a)} for (t, n), a in sorted(nodes.items())],
        "edges": [
            {"source_type": s[0], "source": s[1], "relation": rel, "target_type": d[0], "target": d[1],
             "chunk_ids": sorted(e["chunk_ids"]), "confidence": e["confidence"], "evidence": e["evidence"]}
            for (s, rel, d), e in sorted(edges.items())
        ],
    }
    OUT_PATH.write_text(json.dumps(graph, indent=1))
    names_before = sum(len(v) for v in names_by_type.values())
    print(f"names: {names_before} -> nodes: {len(graph['nodes'])}; relations: {len(relations)} -> edges: {len(graph['edges'])}")


if __name__ == "__main__":
    main()
