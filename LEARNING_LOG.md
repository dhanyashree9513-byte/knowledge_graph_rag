# Learning Log

One entry per day: what I built, what broke, what I'd say in an interview.

---
## Day 1–2 · Setup, LLM wrapper, databases (2026-10-15)

### The big picture
```
.env (secrets) ──► llm.py ─────────► OpenAI      (Day 4: extraction, Day 11: answers)
               └─► check_dbs.py ──► Neo4j + Postgres  (running via docker-compose)
ontology.md ─── tells the extractor what it may pull out
```

### What I built: what / why / when / how

**1. Project environment: `pyproject.toml` + `uv`**
- **What:** the project's Python packages (`openai`, `neo4j`, `psycopg`, ...) with exact versions locked in `uv.lock`.
- **Why:** anyone, including me on a new laptop, gets the same setup with `uv sync`. No "works on my machine" bugs.
- **When:** `uv add <pkg>` when a package is needed; `uv run python ...` on every run.
- **How:** `uv` creates `.venv/`, a private Python just for this project.

**2. Secrets: `.env`, `.env.example`, `.gitignore`**
- **What:** `.env` holds the real API key and passwords. `.env.example` lists the same names with no values. `.gitignore` keeps `.env` off GitHub.
- **Why:** a key pushed to GitHub gets scraped by bots and abused within minutes.
- **When:** every script calls `load_dotenv()` at startup.
- **How:** `os.getenv("OPENAI_API_KEY")` reads a value; the OpenAI client reads its key automatically.

**3. `src/llm.py`: the single doorway to the LLM**
- **What:** one function, `ask(prompt)`, returning `{text, input_tokens, output_tokens, model}`.
- **Why:** token counts on every call make cost tracking possible. Switching model or provider means editing one file. Retries, caching and JSON output get added here later; P3 (the LLM gateway) is the same idea at a larger scale.
- **When:** Day 4 (extraction), Day 7 (baseline answers), Day 11 (final answers).
- **How:** `OpenAI()` client → `chat.completions.create(model, messages)` → answer and token usage copied into a dict. Model comes from `OPENAI_MODEL` in `.env` (`gpt-4.1-mini`).
- Started with mock and Ollama modes; removed them to use real OpenAI calls only.

**4. `src/hello_llm.py`: smoke test**
- **What:** asks one question and prints model, tokens and answer.
- **Why:** proves key, credits and network work before building on top. It caught my empty credit balance.
- **When:** after changing the key or model, or when anything LLM-related breaks.
- First real call: 24 input + 99 output tokens ≈ $0.0002.

**5. `docker-compose.yml`: the two databases**
- **What:** Neo4j (graph) and Postgres + pgvector (embeddings).
- **Why both:** the project's claim is "graph + vector beats vector alone", so I need both the system and the baseline.
- **When:** `docker compose up -d` at the start of each session.
- **How:** `ports` connect the Mac to each container, `volumes` keep data across restarts, `healthcheck` says when a database is actually ready.

**6. `ontology.md`: the graph's schema**
- **What:** 7 entity types and 10 relationship types for the NVIDIA, AMD and Intel 10-Ks, plus 5 multi-hop questions.
- **Why:** "extract everything" gives a messy graph nobody can query. Fixed types keep it clean.
- **When:** Day 4. The extractor may only output these types.

**7. `src/check_dbs.py`: readiness check**
- **What:** prints ✅ or ❌ for Neo4j and Postgres.
- **Why:** a stopped database shows up as one clear line, not a crash halfway through ingestion. Real services do the same thing (Kubernetes calls it a readiness probe).
- **When:** before any pipeline run.
- **How:** connect, run a tiny query (`RETURN 1` / check the `vector` extension), and turn any error into one line. `connect_timeout=5` stops it hanging on a dead database.

### Postgres vs Neo4j
Both are databases; they store data in different shapes.

**Postgres** stores **rows and columns**, like a spreadsheet:

| chunk_id | company | text |
|---|---|---|
| nvda-001 | NVIDIA | "We depend on TSMC to manufacture..." |
| amd-014 | AMD | "TSMC produces our chips..." |

**pgvector** adds an **embedding** per row (a list of numbers capturing the text's meaning), so I can ask for the 5 chunks closest in meaning to a question. That search is **vector RAG**, my baseline.

**Neo4j** stores **nodes** (things) and **edges** (relationships):
```
(NVIDIA) ◄──SUPPLIES── (TSMC) ──SUPPLIES──► (AMD)
   │                                          │
   └──────────── COMPETES_WITH ───────────────┘
```
Its query language, **Cypher**, follows connections:
```cypher
MATCH (n:Company {name:"NVIDIA"})<-[:SUPPLIES]-(s)-[:SUPPLIES]->(a:Company {name:"AMD"})
RETURN s.name          // → TSMC
```

**Why both:** for "Which suppliers do NVIDIA and AMD share?", vector search finds chunks that *sound similar* and may miss the connection because the two facts are in different documents. The graph walks NVIDIA → TSMC → AMD and returns the exact answer with a source chunk per edge.
Rule of thumb: vector search = "what does the text say about X?"; graph = "how are X and Y connected?"

### How Neo4j was set up
1. **Downloaded the image:** `docker pull neo4j:5-community`. An image is a packaged, ready-to-run Neo4j; a container is a running copy of it. Docker keeps it isolated from my Mac, deletable and reproducible.
2. **Wrote the compose section:**
   ```yaml
   neo4j:
     image: neo4j:5-community          # which image to run
     container_name: kgrag-neo4j       # fixed name, easy to type
     ports:
       - "7474:7474"                   # browser UI → http://localhost:7474
       - "7687:7687"                   # bolt port, used by Python
     environment:
       NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}   # read from .env
     volumes:
       - neo4j_data:/data              # data survives container restarts
     healthcheck: ...                  # "healthy" only once the UI answers
   ```
   Ports are `Mac port : container port`. Two ports: one for humans (browser), one for code (bolt).
3. **Password and driver:** `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` (8+ chars) in `.env`; `uv add neo4j` installed the Python driver.
4. **Started and checked:** `docker compose up -d` → `docker compose ps` until `(healthy)` → `uv run python src/check_dbs.py` → ✅.

Postgres followed the same steps (image `pgvector/pgvector:pg16`, port 5432), plus `CREATE EXTENSION vector;` to turn on pgvector.

**Daily commands**

| Command | Does |
|---|---|
| `docker compose up -d` | Start both databases |
| `docker compose ps` | Check they're healthy |
| `docker compose down` | Stop and remove containers; data stays in volumes |
| `docker compose logs neo4j` | See why Neo4j isn't starting |

### What broke
- OpenAI returned `429 insufficient_quota`: the key was valid but the account had no credits. A 429 can mean a rate limit (wait and retry) or an empty quota (retrying never helps); check the error `code`, not just the status. This matters for P3's retry logic.
- VS Code showed "Import could not be resolved": it was using the wrong Python. Fix: **Python: Select Interpreter** → `.venv`.

### Interview line
"Every LLM call goes through one wrapper that returns token usage, so I can estimate cost before batch runs. Before any pipeline run, a readiness check confirms Neo4j and Postgres are up, so failures are one clear line instead of a mid-run crash."

---

## Concepts: Docker, YAML, docker-compose (explained simply)

### Docker: why
Installing a database directly on my Mac is messy: versions clash, settings scatter, and uninstalling is hard. Docker runs each program in a sealed box called a **container**.

| Word | Simple meaning | Example |
|---|---|---|
| **Image** | A packaged, ready-to-run program (like an installer) | `neo4j:5-community` |
| **Container** | A running copy of an image (like an installed, open app) | `kgrag-neo4j` |
| **Volume** | A folder on my disk that the container saves its data into | `neo4j_data` |
| **Port** | A door from my Mac into the container | `7474:7474` |

Delete a container and its data is gone, unless the data lives in a volume.

### YAML: what it is
YAML is a plain-text format for settings. Only a few rules:
```yaml
name: value          # key, colon, SPACE, value
parent:
  child: value       # indentation (2 spaces) = "belongs to"
list:
  - first item       # dash = list item
  - second item
# a comment
```
- **Spaces only, never tabs.** Wrong indentation is the most common YAML error.
- Always a space after the colon: `image: neo4j`, not `image:neo4j`.

### docker-compose.yml: why we need it
Without it, each database needs a long command typed every time:
```
docker run -d --name kgrag-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/secret -v neo4j_data:/data neo4j:5-community
```
With it, I write those settings **once** in a file, commit it to git, and start everything with **one** command. Anyone who clones the repo gets the same setup.

### How to write it: each line maps to part of that long command
```yaml
services:                         # the programs to run
  neo4j:                          # my name for this service
    image: neo4j:5-community      # which image            (docker run ... neo4j:5-community)
    container_name: kgrag-neo4j   # fixed container name   (--name kgrag-neo4j)
    ports:
      - "7474:7474"               # Mac:container, browser (-p 7474:7474)
      - "7687:7687"               # Mac:container, Python  (-p 7687:7687)
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}   # setting; ${...} is read from .env  (-e ...)
    volumes:
      - neo4j_data:/data          # save /data into the volume  (-v neo4j_data:/data)
    healthcheck:                  # how Docker knows it's READY, not just started
      test: ["CMD-SHELL", "wget -q --spider http://localhost:7474 || exit 1"]
      interval: 5s
      retries: 20

volumes:                          # declare the volumes used above
  neo4j_data:
```
Where the values come from: the **image name** from Docker Hub (hub.docker.com), and the **ports, environment variables and data folder** from that image's documentation.

### How to use it

| When | Command |
|---|---|
| Start of session | `docker compose up -d` (`-d` = run in background) |
| Is it ready? | `docker compose ps` → wait for `(healthy)` |
| Something's wrong | `docker compose logs neo4j` |
| End of session | `docker compose down` (containers removed, data kept) |
| Wipe all data | `docker compose down -v` (careful: deletes volumes) |

Run these inside the project folder, where `docker-compose.yml` lives.

### Adding a new service later (e.g. Redis in Project 2)
Same pattern: add a block under `services:`.
```yaml
  redis:
    image: redis:7
    ports:
      - "6379:6379"
```
Then `docker compose up -d` starts only what's new.

---

## Day 2 (part 2) · Downloading the 10-Ks (2026-10-15)

**What:** `src/download_10k.py` gets the latest 10-K of NVIDIA, AMD and Intel from SEC EDGAR and keeps only Item 1 (Business) and Item 1A (Risk Factors). Output: `data/raw/<ticker>.json`.

**How, in 5 steps:**
1. Ask SEC's free API (`data.sec.gov/submissions/CIK….json`) for the newest 10-K. SEC uses a number called the **CIK**, not the ticker.
2. Download the HTML once; re-runs use the cached file.
3. Strip HTML to plain text (BeautifulSoup); drop hidden XBRL data (`ix:header`).
4. Cut out the two sections with regexes that only match **at the start of a line**, so a mid-sentence "see Item 1A" doesn't count.
5. Save the sections with `accession`, `filing_date` and `url` (provenance: every answer can be traced to the exact filing).

**What broke: Intel.** The size check printed ⚠ (Item 1 = 160 chars). Intel files its 10-K as an annual report: no "Item 1." headings in the body, only a cross-reference index at the end. I looked at the text first, then:
- gave Intel its own heading patterns ("Overview" → "Management's Discussion and Analysis"; "Risk Factors" → "Other Key Information"),
- changed the rule from "longest span" to "**first span over 5,000 chars**", because MD&A has its own "Overview" that won the longest-span contest,
- re-ran NVIDIA and AMD to confirm their numbers didn't change.

**Lesson:** parsers fail silently. Add a sanity check (here: section size) and look at the real data before changing code.

**Result:** ~533k characters (~130k tokens). NVDA 48k + 115k, AMD 60k + 132k, INTC 82k + 96k chars.

---

## Day 3 · Chunking with stable ids (2026-10-16)

**What:** `src/chunk.py` turns the 6 sections into **334 chunks** in `data/chunks.jsonl` (one JSON per line).

A chunk:
```json
{
  "chunk_id": "NVDA-10K-2026-02-25-6637efa5bd47",
  "doc_id": "NVDA-10K-2026-02-25",
  "doc_hash": "d61ada10…",
  "ticker": "NVDA",
  "section": "item_1",
  "section_path": ["NVDA-10K-2026-02-25", "Item 1 Business", "Manufacturing"],
  "position": 19,
  "text": "… We utilize foundries, such as … TSMC, and … Samsung …"
}
```

**How, in 4 steps:**
1. **Clean:** drop page junk ("Table of Contents", page numbers), attach bullets to their text, rejoin sentences the HTML split.
2. **Cut:** pack whole paragraphs up to ~600 tokens (2,400 chars). A heading always starts a new chunk; long paragraphs split only at sentence ends.
3. **Label:**
   - `chunk_id` = doc id + hash of the chunk's **text**, not its position, so ids don't shift when the document changes. Graph edges (Day 6) and vectors (Day 7) point at these ids.
   - `doc_hash` = fingerprint of the report; Day 4 skips re-extraction (and its cost) if it hasn't changed.
   - `section_path` = where the text came from, for filters and citations.
4. **Save** to `data/chunks.jsonl` (gitignored: rebuildable from the raw data).

**Bugs found by reading the output:**
- a chunk that was just `"ITEM 1."` (AMD's heading ends with a period) → added to the noise list,
- a chunk that was just `"®"` (Intel puts it on its own line: "Intel / ® / Xeon") → joined back into "Intel® Xeon".

**Checks:** running twice gives a byte-identical file; all ids unique (the script asserts it).

### How to choose chunk size, overlap and metadata
- **Size follows the job.** LLM extraction: 400–800 tokens. Vector search: 200–500 (one topic per chunk = a sharp embedding). Too small loses context and costs more calls; too big blurs topics and the extractor misses facts. I chose ~600 for both.
- **Overlap** (repeat 10–20% of the previous chunk) protects facts cut at a boundary, but costs tokens and creates duplicates. I use **none**, because I cut at headings and paragraphs, never blindly every N characters.
- **Where to cut, best first:** headings → paragraphs → sentences → never every N characters.
- **Metadata:** each field needs a future user. `chunk_id` (graph + vectors), `doc_id` (citations), `doc_hash` (cache), `ticker` and `section` (filters), `position` (reading order).
- **Measure, don't guess:** Day 5 measures recall@k on labeled questions; if it's low, change the size and re-measure.

**Interview line:** "I chunk on document structure, not fixed character counts, and ids come from content hashes so graph edges and vectors stay valid when documents change."
