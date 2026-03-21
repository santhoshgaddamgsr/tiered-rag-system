from langchain_ollama import OllamaLLM


llm = OllamaLLM(
    model="qwen2.5-coder:7b",
    temperature=0
)


def decompose_query(query: str):

    prompt = f"""
Break the question into smaller search queries if needed.

Rules:
- If the question is simple return the same query
- If complex split into 2-3 short search queries
- Return each query on a new line
- Do NOT explain

Question:
{query}

Search queries:
"""

    response = llm.invoke(prompt)

    lines = response.split("\n")

    queries = [q.strip() for q in lines if q.strip()]

    if len(queries) == 0:
        queries = [query]

    return queries