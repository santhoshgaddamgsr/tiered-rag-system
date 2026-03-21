from langchain_core.prompts import PromptTemplate


prompt = PromptTemplate(
    template="""
You are an AI assistant answering questions using retrieved documents.

IMPORTANT RULES:
- Answer ONLY using the information provided in the context.
- Do NOT use outside knowledge.
- If the answer is not clearly present in the context, respond exactly with:
  "I don't know based on the provided documents."
- Be concise and accurate.
- If possible, summarize the relevant information from the context.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)
