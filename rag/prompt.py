from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a factual assistant.

STRICT RULES:
- Keep original keywords EXACTLY
- Use ONLY the provided context to answer
- Do NOT use outside knowledge
- MUST cover ALL parts of the question
- If question has multiple parts, answer each part clearly
- Be concise and precise
- If answer not in context say: "I don't know based on the provided context"
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Context:
{context}

Question:
{question}

Answer (only relevant points):<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
""",
    input_variables=["context", "question"]
)