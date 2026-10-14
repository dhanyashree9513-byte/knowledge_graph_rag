"""Day 1 smoke test: one call through the llm.ask() wrapper.

Run: uv run python src/hello_claude.py
"""

from llm import ask

QUESTION = "In 3 sentences, why does multi-hop question answering break plain vector RAG?"

result = ask(QUESTION)

print(f"mode:          {result['mode']}")
print(f"input tokens:  {result['input_tokens']}")
print(f"output tokens: {result['output_tokens']}")
print()
print(result["text"])
