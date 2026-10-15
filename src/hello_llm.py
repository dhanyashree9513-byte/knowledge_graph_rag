"""Smoke test: one call through the llm.ask() wrapper.

Run: uv run python src/hello_llm.py
"""

from llm import ask

QUESTION = "In 3 sentences, why does multi-hop question answering break plain vector RAG?"

result = ask(QUESTION)

print(f"model:         {result['model']}")
print(f"input tokens:  {result['input_tokens']}")
print(f"output tokens: {result['output_tokens']}")
print()
print(result["text"])
