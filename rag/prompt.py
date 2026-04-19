from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""<|im_start|>system
You are a factual assistant.

STRICT RULES:
- Answer ONLY using the provided context
- Do NOT use outside knowledge
- Extract only information that is DIRECTLY relevant to the question
- Ignore any unrelated or less relevant details
- MUST cover ALL parts of the question
- If question has multiple parts, answer each part separately
- Be concise and precise<|im_end|>

<|im_start|>user
Context:
{context}

Question:
{question}

Answer (only relevant points, no extra information):<|im_end|>

<|im_start|>assistant
""",
    input_variables=["context", "question"]
)