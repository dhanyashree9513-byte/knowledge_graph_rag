"""One web endpoint for the whole system: POST /ask. The demo, curl and the benchmark all use the same path.

Run:  uv run uvicorn api:app --app-dir src --port 8000
Try:  curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question": "Which supplier do NVIDIA and AMD share?"}'
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from answer import answer

app = FastAPI(title="Knowledge Graph RAG")


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Answer(BaseModel):
    answer: str
    citations: list[str]
    route: str
    template: str | None
    latency_s: float
    cost_usd: float


@app.post("/ask", response_model=Answer)
def ask(body: Question) -> dict:  # sync def: FastAPI runs it in a thread, so blocking DB/LLM calls don't stall the server
    return answer(body.question)
