"""One place for every LLM call in this project.

Every call goes to OpenAI and costs money. The model comes from
OPENAI_MODEL in .env, so switching models never needs a code change.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # cheap model while learning

client = OpenAI(max_retries=3)  # reads OPENAI_API_KEY; retries rate limits and 5xx with backoff


def ask(prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = 300) -> dict:
    """Send one prompt and return the answer plus token usage."""
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "text": response.choices[0].message.content,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "model": response.model,
    }


def ask_structured(system: str, prompt: str, schema: type, model: str = DEFAULT_MODEL) -> dict:
    """Like ask(), but the answer must match a Pydantic class (OpenAI structured outputs)."""
    response = client.chat.completions.parse(
        model=model,
        temperature=0,  # same input -> (nearly) same output; we want repeatable extraction
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        response_format=schema,
    )
    message = response.choices[0].message
    if message.parsed is None:  # the model refused or was cut off
        raise ValueError(f"no parsed output: {message.refusal or response.choices[0].finish_reason}")
    return {
        "parsed": message.parsed,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "model": response.model,
    }


EMBED_MODEL = "text-embedding-3-small"  # $0.02 per million tokens


def embed(texts: list[str]) -> list[list[float]]:
    """One embedding vector per text, sent in batches."""
    vectors = []
    for i in range(0, len(texts), 1000):
        response = client.embeddings.create(model=EMBED_MODEL, input=texts[i : i + 1000])
        vectors += [item.embedding for item in response.data]
    return vectors
