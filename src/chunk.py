"""Split the 10-K sections into chunks with stable ids.

Run: uv run python src/chunk.py
Input: data/raw/<ticker>.json   Output: data/chunks.jsonl (one chunk per line)
"""

import hashlib
import json
import re
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/chunks.jsonl")
MAX_CHARS = 2400  # about 600 tokens (1 token is roughly 4 characters of English)
SECTION_NAMES = {"item_1": "Item 1 Business", "item_1a": "Item 1A Risk Factors"}

# Page furniture left over from the HTML: footers, page numbers, broken "Table of Contents" links.
NOISE = re.compile(r"^(table of conten(ts)?|[ts]|\d+|our business|risk factors|item 1a?\.?|\.)$", re.I)
BULLETS = {"•", "▪"}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def clean_paragraphs(text: str) -> list[str]:
    """One paragraph per line, with page noise removed and broken lines rejoined."""
    paragraphs: list[str] = []
    bullet = False
    text = re.sub(r"\s*\n\s*([®™])\s*\n\s*", r"\1 ", text)  # "Intel\n®\nXeon" -> "Intel® Xeon"
    for line in text.replace("\u202f", " ").split("\n"):
        line = line.strip()
        if not line or NOISE.match(line):
            continue
        if line in BULLETS:  # the bullet sits on its own line; attach it to the next one
            bullet = True
            continue
        if paragraphs and line[0].islower():  # a sentence split across lines by the HTML
            paragraphs[-1] += " " + line
        else:
            paragraphs.append(("• " if bullet else "") + line)
        bullet = False
    return paragraphs


def is_heading(paragraph: str) -> bool:
    """Short, starts with a capital, no closing punctuation: e.g. 'Risks Related to Our Industry and Markets'."""
    return len(paragraph) < 100 and paragraph[0].isupper() and paragraph[-1] not in ".:;,)"


def split_long(paragraph: str) -> list[str]:
    """Break a paragraph bigger than MAX_CHARS at sentence ends."""
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        if current and len(current) + len(sentence) > MAX_CHARS:
            pieces.append(current)
            current = ""
        current = f"{current} {sentence}".strip()
    return pieces + [current]


def chunk_section(paragraphs: list[str]) -> list[tuple[str, str]]:
    """Pack paragraphs into (heading, text) chunks; a new heading always starts a new chunk."""
    chunks, current, heading = [], "", ""
    for paragraph in paragraphs:
        if is_heading(paragraph):
            if current:
                chunks.append((heading, current))
            current, heading = "", paragraph
            continue
        for piece in split_long(paragraph):
            if current and len(current) + len(piece) > MAX_CHARS:
                chunks.append((heading, current))
                current = ""
            current = f"{current}\n{piece}".strip()
    if current:
        chunks.append((heading, current))
    return chunks


def main() -> None:
    rows = []
    for path in sorted(RAW_DIR.glob("*.json")):
        filing = json.loads(path.read_text())
        doc_id = f"{filing['ticker']}-10K-{filing['filing_date']}"
        doc_hash = sha(json.dumps(filing["sections"], sort_keys=True))  # changes only if the text changes

        for section, text in filing["sections"].items():
            for position, (heading, body) in enumerate(chunk_section(clean_paragraphs(text))):
                rows.append({
                    "chunk_id": f"{doc_id}-{sha(section + body)[:12]}",  # from content, not position
                    "doc_id": doc_id,
                    "doc_hash": doc_hash,
                    "ticker": filing["ticker"],
                    "section": section,
                    "section_path": [doc_id, SECTION_NAMES[section], heading],
                    "position": position,
                    "text": body,
                })

    ids = [row["chunk_id"] for row in rows]
    assert len(ids) == len(set(ids)), "duplicate chunk_id: two chunks have identical text"

    with OUT_PATH.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    sizes = [len(row["text"]) for row in rows]
    print(f"{len(rows)} chunks from {len({r['doc_id'] for r in rows})} docs -> {OUT_PATH}")
    print(f"chars per chunk: min {min(sizes)}, avg {sum(sizes) // len(sizes)}, max {max(sizes)}")


if __name__ == "__main__":
    main()
