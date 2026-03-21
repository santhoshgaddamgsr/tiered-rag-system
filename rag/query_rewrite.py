from langchain_ollama import OllamaLLM

# load small LLM instance for rewriting
llm = OllamaLLM(
    model="qwen2.5-coder:7b",
    temperature=0
)


def rewrite_query(query: str) -> str:

    prompt = f"""
Rewrite the user question to improve document retrieval.

Rules:
- keep the same meaning
- make it clear and searchable
- keep it short
- do not add explanations

Question:
{query}

Rewritten query:
"""

    rewritten = llm.invoke(prompt)

    return rewritten.strip()