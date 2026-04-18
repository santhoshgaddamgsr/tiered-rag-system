from typing import TypedDict
import time

from langgraph.graph import StateGraph

from rag.retriever import get_retriever
from rag.router import similarity_topk
from rag.cache import SimpleCache
from rag.prompt import prompt
from rag.query_rewrite import rewrite_query
from rag.query_decompose import decompose_query

from langchain_ollama import OllamaLLM
from sentence_transformers import CrossEncoder
from rag.logger import log_event
from rag.config import config


class GraphState(TypedDict, total=False):

    question: str
    rewritten_query: str
    context: str
    sources: list
    answer: str
    route: str
    avg_distance: float
    distances: list
    reranker_scores: list
    retrieved_chunks: list
    llm_used: bool


# -------------------------
# LOAD COMPONENTS
# -------------------------

vector_retriever, bm25 = get_retriever()

cache = SimpleCache(max_size=50)

llm = OllamaLLM(
    model="qwen2.5-coder:7b",
    temperature=config["llm"]["temperature"]
)

reranker = CrossEncoder("BAAI/bge-reranker-base")

THRESHOLD = 0.35


# -------------------------
# CACHE CHECK NODE
# -------------------------

def cache_check(state):

    query = state["question"]

    cached = cache.get(query)

    if cached:

        return {
            "answer": cached["answer"],
            "sources": cached["sources"],
            "route": "cache",
            "llm_used": False
        }

    return {
        "question": query,
        "route": "rewrite"
    }


# -------------------------
# QUERY REWRITE NODE
# -------------------------

def rewrite(state):

    query = state["question"]

    rewritten = rewrite_query(query)

    return {
        "question": query,
        "rewritten_query": rewritten,
        "route": "router"
    }


# -------------------------
# ROUTER NODE (FIXED)
# -------------------------

def router(state):

    query = state["rewritten_query"]

    distances, avg_distance = similarity_topk(query)

    if distances:
        top_k = min(3, len(distances))
        top_scores = sorted(distances)[:top_k]
        score = sum(top_scores) / len(top_scores)
    else:
        score = avg_distance

    if score > THRESHOLD:

        return {
            "question": state["question"],
            "rewritten_query": query,
            "route": "llm",
            "avg_distance": score,
            "distances": distances
        }

    return {
        "question": state["question"],
        "rewritten_query": query,
        "route": "rag",
        "avg_distance": score,
        "distances": distances
    }


# -------------------------
# RETRIEVE NODE
# -------------------------

def retrieve(state):

    start_time = time.time()

    rewritten_query = state["rewritten_query"]

    queries = decompose_query(rewritten_query)

    vector_docs = []
    bm25_docs = []

    num_queries = len(queries)

    if num_queries == 1:
        vector_docs = vector_retriever.invoke(queries[0])[:30]
        bm25_docs = bm25.invoke(queries[0])[:10]

    else:
        vector_k = max(1, 30 // num_queries)
        bm25_k = max(1, 10 // num_queries)

        for q in queries:
            vector_docs.extend(vector_retriever.invoke(q)[:vector_k])
            bm25_docs.extend(bm25.invoke(q)[:bm25_k])

    # RRF
    rrf_scores = {}
    k = 60

    for rank, doc in enumerate(vector_docs):

        key = doc.page_content.strip()

        if key not in rrf_scores:
            rrf_scores[key] = {"doc": doc, "score": 0}

        rrf_scores[key]["score"] += 1 / (k + rank + 1)

    for rank, doc in enumerate(bm25_docs):

        key = doc.page_content.strip()

        if key not in rrf_scores:
            rrf_scores[key] = {"doc": doc, "score": 0}

        rrf_scores[key]["score"] += 1 / (k + rank + 1)

    fused_docs = sorted(
        rrf_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    unique_docs = [item["doc"] for item in fused_docs]

    # RERANKER
    pairs = [
        (rewritten_query, doc.page_content)
        for doc in unique_docs[:80]
    ]

    scores = reranker.predict(pairs)

    scored_docs = list(zip(unique_docs[:80], scores))

    scored_docs.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # DYNAMIC FILTERING
    if scored_docs:
        max_score = scored_docs[0][1]
    else:
        max_score = 0

    threshold = max_score * config["reranker"]["threshold_multiplier"]

    filtered = [
        (doc, score) for doc, score in scored_docs
        if score >= threshold and score > config["reranker"]["min_score"]
    ]

    if not filtered:
        top_docs = []
        top_scores = []
    else:
        max_docs = config["reranker"]["max_docs"]
        top_docs = [doc for doc, _ in filtered][:max_docs]
        top_scores = [float(score) for _, score in filtered][:max_docs]

    print("\n--- RERANK DEBUG ---")
    print("Top scores:", [round(s, 4) for _, s in scored_docs[:10]])
    print("Threshold:", round(threshold, 4))
    print("Filtered count:", len(filtered))

    if top_docs:
        context = "\n\n".join(
            [doc.page_content for doc in top_docs]
        )
    else:
        context = ""

    sources = []
    seen_sources = set()

    for doc in top_docs:

        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")

        key = (source, page)

        if key not in seen_sources:

            sources.append({
                "source": source,
                "page": page
            })

            seen_sources.add(key)

    retrieved_chunks = []

    for doc, score in zip(top_docs, top_scores):

        retrieved_chunks.append({
            "score": float(score),
            "text": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "unknown")
        })

    log_event({
        "query": state["question"],
        "rewritten_query": rewritten_query,
        "route": "rag",
        "num_queries": num_queries,
        "num_retrieved": len(scored_docs),
        "num_filtered": len(filtered),
        "top_scores": [round(float(s), 4) for _, s in scored_docs[:5]],
        "avg_score": round(
            sum(float(s) for _, s in scored_docs[:5]) / min(5, len(scored_docs)), 4
        ) if scored_docs else 0,
        "final_docs": len(top_docs),
        "latency": round(time.time() - start_time, 3)
    })

    return {
        "question": state["question"],
        "rewritten_query": rewritten_query,
        "context": context,
        "sources": sources,
        "reranker_scores": top_scores,
        "retrieved_chunks": retrieved_chunks
    }


# -------------------------
# CONFIDENCE FUNCTION (NEW)
# -------------------------

def compute_confidence(rerank_scores):

    if not rerank_scores:
        return 0.0

    top_score = rerank_scores[0]

    top_k = min(3, len(rerank_scores))
    avg_top3 = sum(rerank_scores[:top_k]) / top_k

    num_good = len([s for s in rerank_scores if s > 0.7])
    max_docs = max(1, len(rerank_scores))

    confidence = (
        0.5 * top_score +
        0.3 * avg_top3 +
        0.2 * (num_good / max_docs)
    )

    return round(confidence, 4)
#--------------------
# Query type detector
#--------------------
def is_reasoning_query(query: str) -> bool:
    q = query.lower().strip()

    # strong signals
    if q.startswith(("why", "how")):
        return True

    # reasoning patterns
    keywords = [
        "compare", "difference", "explain",
        "reason", "impact", "advantages", "disadvantages"
    ]

    return any(k in q for k in keywords)
# -------------------------
# GENERATE NODE (UPDATED)
# -------------------------

def generate(state):

    query = state["question"]
    rewritten_query = state.get("rewritten_query", query)
    context = state.get("context", "")

    rerank_scores = state.get("reranker_scores", [])
    retrieved_chunks = state.get("retrieved_chunks", [])

    confidence = compute_confidence(rerank_scores)

    is_reasoning = is_reasoning_query(query)

    if context and confidence >= 0.75 and not is_reasoning:

        if retrieved_chunks:
            top_chunk = retrieved_chunks[0]["text"].strip()
            answer = " ".join(top_chunk.split())[:500]

            if "." in answer:
                answer = answer[:answer.rfind(".")+1]
        else:
            answer = ""

        mode = "SKIPPED_LLM"
        llm_used = False

    elif context and confidence >= 0.4:

        formatted_prompt = prompt.format(
            context=context,
            question=rewritten_query
        )

        answer = llm.invoke(formatted_prompt)

        mode = "RAG"
        llm_used = True

    else:

        answer = llm.invoke(query)

        mode = "LLM_FALLBACK"
        llm_used = True

    sources = state.get("sources", [])

    cache.set(rewritten_query, {
        "answer": answer,
        "sources": sources
    })

    log_event({
        "query": query,
        "mode": mode,
        "confidence": confidence,
        "llm_used": llm_used,
        "used_context": bool(context),
        "num_chunks": len(rerank_scores),
        "context_length": len(context),
        "answer_length": len(answer)
    })
    print("\n--- DECISION DEBUG ---")
    print("Confidence:", confidence)
    print("Mode:", mode)
    print("LLM Used:", llm_used)
    print("Top Scores:", rerank_scores[:3])

    return {
        "answer": answer,
        "sources": sources,
        "llm_used": llm_used,
        "confidence": confidence
    }


# -------------------------
# BUILD GRAPH
# -------------------------

builder = StateGraph(GraphState)

builder.add_node("cache_check", cache_check)
builder.add_node("rewrite", rewrite)
builder.add_node("router", router)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)

builder.set_entry_point("cache_check")

builder.add_conditional_edges(
    "cache_check",
    lambda x: x["route"],
    {
        "cache": "__end__",
        "rewrite": "rewrite"
    }
)

builder.add_conditional_edges(
    "rewrite",
    lambda x: x["route"],
    {
        "router": "router"
    }
)

builder.add_conditional_edges(
    "router",
    lambda x: x["route"],
    {
        "rag": "retrieve",
        "llm": "generate"
    }
)

builder.add_edge("retrieve", "generate")

graph = builder.compile()