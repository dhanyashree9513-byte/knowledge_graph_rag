"""One place for every LLM call in this project.

Every call goes to OpenAI and costs money. The model comes from
OPENAI_MODEL in .env, so switching models never needs a code change.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # cheap model while learning

client = OpenAI()  # reads OPENAI_API_KEY from the environment


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
