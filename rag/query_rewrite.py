from langchain_ollama import OllamaLLM

# -------------------------
# LOAD LLM (for rewriting)
# -------------------------
llm = OllamaLLM(
    model="llama3.1",
    temperature=0
)


# -------------------------
# REWRITE FUNCTION
# -------------------------
def rewrite_query(query: str) -> str:

    prompt = f"""Improve this search query for document retrieval.
RULES: Keep the same meaning. Expand abbreviations only. Do not change the topic. Output one line.

Query: {query}
Improved query:"""

    try:
        rewritten = llm.invoke(prompt).strip()
        rewritten = rewritten.split("\n")[0].strip()

        print(f"\n--- REWRITE DEBUG ---")
        print(f"Original: {query}")
        print(f"Rewritten: {rewritten}")

        words = len(rewritten.split())

        # -------------------------
        # GUARD 1: Length
        # -------------------------
        if words < 4 or words > 20:
            print(f"FALLBACK: length {words} out of range")
            return query

        # -------------------------
        # GUARD 2: Topic drift check
        # -------------------------
        original_words = set(query.lower().split())
        rewritten_words = set(rewritten.lower().split())

        stopwords = {
            "what", "is", "are", "the", "a", "an",
            "and", "of", "to", "in", "on", "for", "with",
            "how", "does", "do", "given", "specific",
            "associated", "phrases", "keywords", "text", "corpus"
        }

        orig_keywords = {w for w in original_words if w not in stopwords}

        if orig_keywords:
            overlap = orig_keywords.intersection(rewritten_words)
            ratio = len(overlap) / len(orig_keywords)
            print(f"Overlap ratio: {ratio} | Keywords: {orig_keywords} | Overlap: {overlap}")

            if ratio < 0.3:
                print("FALLBACK: topic changed")
                return query

        return rewritten

    except Exception as e:
        print(f"Rewrite error: {e}")
        return query