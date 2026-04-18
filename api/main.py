from fastapi import FastAPI
from pydantic import BaseModel

from rag.graph import graph


app = FastAPI(title="Tiered RAG LangGraph")


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
def query(request: QueryRequest):

    result = graph.invoke({
        "question": request.question
    })

    return {
        "question": request.question,
        "rewritten_query": result.get("rewritten_query"),
        "answer": result.get("answer"),
        "sources": result.get("sources"),
        "reranker_scores": result.get("reranker_scores"),
        "retrieved_chunks": result.get("retrieved_chunks"),
        "route": result.get("route"),
        "avg_distance": result.get("avg_distance"),
        "top_k_distances": result.get("distances"),
        "confidence": result.get("confidence"),
        "llm_used": result.get("llm_used")
    }