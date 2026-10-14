"""One place for every LLM call in this project.

LLM_MODE in .env decides what happens:
    mock -> no network call, fixed placeholder text, $0 (the default)
    live -> real Claude call; errors are raised, never hidden

Mock on purpose, fail loudly by accident: we never fall back to fake
text when a live call fails, because that would hide a broken key and
quietly corrupt benchmark results.
"""

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "claude-haiku-5-5"  # cheap model while learning
MOCK_TEXT = "[MOCK] This is a placeholder answer."


def ask(prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = 300) -> dict:
    """Send one prompt and return the answer plus token usage."""
    mode = os.getenv("LLM_MODE", "mock")

    if mode == "mock":
        return {"text": MOCK_TEXT, "input_tokens": 0, "output_tokens": 0, "mode": "mock"}

    if mode == "live":
        # Imported here so mock mode never needs the SDK or an API key.
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return {
            "text": response.content[0].text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "mode": "live",
        }

    raise ValueError(f"LLM_MODE must be 'mock' or 'live', got {mode!r}. Check your .env file.")
