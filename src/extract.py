"""Extract entities and relationships from every chunk, limited to the ontology.

Run:  uv run python src/extract.py --sample 9   (cost estimate on 9 chunks)
      uv run python src/extract.py              (all chunks; cached chunks are free)
Output: data/extractions.jsonl (one line per chunk)
"""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from llm import ask_structured

CHUNKS_PATH = Path("data/chunks.jsonl")
OUT_PATH = Path("data/extractions.jsonl")
CACHE_DIR = Path("data/cache/extract")
PROMPT_VERSION = "v4"  # bump when the prompt or schema changes, so old cache entries are ignored
PRICE_PER_M = {"input": 0.40, "output": 1.60}  # gpt-4.1-mini, USD per million tokens
COMPANY = {"NVDA": "NVIDIA", "AMD": "AMD", "INTC": "Intel"}

# --- The schema: the model can only answer with these types (from ontology.md) ---
EntityType = Literal["Company", "Product", "Market", "Technology", "Risk", "Location", "Person"]
RelationType = Literal[
    "SUPPLIES", "COMPETES_WITH", "MAKES", "TARGETS",
    "USES", "FACES", "INVOLVES", "OPERATES_IN", "EXECUTIVE_OF",
]

# Risks use a fixed list, so the same risk at NVIDIA and AMD becomes ONE node they both connect to.
RISKS = [
    "Supplier concentration", "Supply chain disruption", "Export controls", "Geopolitical tension",
    "Competition", "Customer concentration", "Demand volatility", "Cybersecurity",
    "Natural disasters", "Regulation and legal", "Intellectual property", "Talent retention",
    "Product defects", "Debt and financing", "Acquisitions and integration", "Macroeconomic conditions",
    "Climate and sustainability", "Manufacturing execution",
]

# Which (source type, target type) each relationship may connect.
ALLOWED = {
    "SUPPLIES": {("Company", "Company")},
    "COMPETES_WITH": {("Company", "Company")},
    "MAKES": {("Company", "Product")},
    "TARGETS": {("Product", "Market")},
    "USES": {("Product", "Technology")},
    "FACES": {("Company", "Risk")},
    "INVOLVES": {("Risk", "Company"), ("Risk", "Location")},
    "OPERATES_IN": {("Company", "Location")},
    "EXECUTIVE_OF": {("Person", "Company")},
}


class Entity(BaseModel):
    name: str
    type: EntityType


class Relation(BaseModel):
    source: str
    source_type: EntityType
    relation: RelationType
    target: str
    target_type: EntityType
    confidence: float  # 0-1: how directly the text states it
    evidence: str  # short quote from the chunk


class Extraction(BaseModel):
    entities: list[Entity]
    relations: list[Relation]


SYSTEM = """You build a knowledge graph from SEC 10-K text.
The filing company is {company}: "we", "our" and "us" mean {company}.

Entity types: Company, Product (a named chip, platform or product line), Market (an end market such as Data Center or Gaming),
Technology (an architecture, process node or standard such as CUDA, x86, 3nm), Risk, Location (country or region), Person (a named executive).

Relationships (direction matters):
- (Company)-[SUPPLIES]->(Company): supplier -> buyer. "We use TSMC to make our wafers" or "we buy memory from Micron"
  in {company}'s filing means (TSMC)-[SUPPLIES]->({company}) and (Micron)-[SUPPLIES]->({company}).
- (Company)-[COMPETES_WITH]->(Company): the text names it as a competitor
- (Company)-[MAKES]->(Product)
- (Product)-[TARGETS]->(Market)
- (Product)-[USES]->(Technology)
- (Company)-[FACES]->(Risk)
- (Risk)-[INVOLVES]->(Company or Location): the risk is about that company or place
- (Company)-[OPERATES_IN]->(Location): manufacturing or significant operations there
- (Person)-[EXECUTIVE_OF]->(Company)

Rules:
- Only facts stated in this text. No outside knowledge.
- Use short common names: "TSMC" not "Taiwan Semiconductor Manufacturing Company Limited", "NVIDIA" not "NVIDIA Corporation".
- Risk names must be one of: {risks}. Map each risk in the text to the closest category; one FACES relation per category, not one per list item.
- Products are named chips, platforms, systems or software. Debt notes, business segments and divisions are NOT products.
- Skip generic phrases ("our products", "customers", "competitors") that are not named.
- confidence: 1.0 if stated explicitly, lower if implied.
- evidence: a quote of at most 25 words from the text.
- Return empty lists if the text has no relevant facts."""


NOT_A_PRODUCT = re.compile(r"%|\bnotes?\b|\bdebentures?\b", re.I)  # debt such as "3.924% Senior Notes due 2032"


def validate(extraction: Extraction, company: str) -> tuple[list[dict], int]:
    """Keep relations that fit the ontology and pass the code checks; return them plus how many were dropped."""
    kept = []
    for r in extraction.relations:
        if r.target_type == "Product" and NOT_A_PRODUCT.search(r.target):
            continue
        if r.relation == "INVOLVES" and r.target == company:  # "a risk involves the filer" adds nothing
            continue
        fits = (r.source_type, r.target_type) in ALLOWED[r.relation]
        risk_ok = (r.target in RISKS) if r.relation == "FACES" else (r.source in RISKS) if r.relation == "INVOLVES" else True
        if fits and risk_ok and r.source.strip() and r.target.strip() and r.source != r.target and 0 <= r.confidence <= 1:
            kept.append(r.model_dump())
    return kept, len(extraction.relations) - len(kept)


def extract_chunk(chunk: dict) -> dict:
    cache_path = CACHE_DIR / f"{chunk['chunk_id']}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached["prompt_version"] == PROMPT_VERSION:
            return cached | {"from_cache": True}

    company = COMPANY[chunk["ticker"]]
    result = ask_structured(SYSTEM.format(company=company, risks=", ".join(RISKS)), chunk["text"], Extraction)
    relations, dropped = validate(result["parsed"], company)
    record = {
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "prompt_version": PROMPT_VERSION,
        "model": result["model"],
        "entities": [e.model_dump() for e in result["parsed"].entities],
        "relations": relations,
        "dropped": dropped,
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
    }
    cache_path.write_text(json.dumps(record, indent=2))
    return record | {"from_cache": False}


def cost(records: list[dict]) -> float:
    tokens_in = sum(r["input_tokens"] for r in records)
    tokens_out = sum(r["output_tokens"] for r in records)
    return (tokens_in * PRICE_PER_M["input"] + tokens_out * PRICE_PER_M["output"]) / 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, help="only run N evenly spaced chunks (for a cost estimate)")
    args = parser.parse_args()

    chunks = [json.loads(line) for line in CHUNKS_PATH.open()]
    total = len(chunks)
    if args.sample:
        chunks = chunks[:: total // args.sample][: args.sample]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as pool:  # 8 calls in parallel
        records = list(pool.map(extract_chunk, chunks))

    new = [r for r in records if not r["from_cache"]]
    print(f"{len(records)} chunks ({len(new)} new, {len(records) - len(new)} from cache)")
    print(f"relations kept: {sum(len(r['relations']) for r in records)}, dropped by validation: {sum(r['dropped'] for r in records)}")
    print(f"cost of this run: ${cost(new):.4f}")
    print(f"estimated cost for all {total} chunks: ${cost(records) / len(records) * total:.2f}")
    for doc_id in sorted({r["doc_id"] for r in records}):
        doc = [r for r in records if r["doc_id"] == doc_id]
        print(f"  {doc_id}: {len(doc)} chunks, ${cost(doc):.4f}")

    if not args.sample:
        with OUT_PATH.open("w") as f:
            for r in records:
                f.write(json.dumps({k: v for k, v in r.items() if k != "from_cache"}) + "\n")
        print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
