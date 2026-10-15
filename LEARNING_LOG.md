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
