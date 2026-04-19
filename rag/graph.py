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
    confidence: float 


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
    print(f"\n--- CACHE DEBUG --- hit: {cache.get(query) is not None}")
    
    cached = cache.get(query)

    if cached:
        return {
            "answer": cached["answer"],
            "sources": cached["sources"],
            "route": "cache",      # ← already present
            "llm_used": False
        }

    return {
        "question": query,
        "route": "rewrite"         # ← already present
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


def run_retrieval_pipeline(query: str):

    # --- HYBRID RETRIEVAL ---
    queries = decompose_query(query)

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

    # --- RRF ---
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

    fused_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    unique_docs = [item["doc"] for item in fused_docs]

    # --- RERANK ---
    pairs = [(query, doc.page_content) for doc in unique_docs[:80]]
    scores = reranker.predict(pairs)

    scored_docs = list(zip(unique_docs[:80], scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # --- FILTER ---
    if scored_docs:
        max_score = scored_docs[0][1]
    else:
        max_score = 0

    threshold = max_score * config["reranker"]["threshold_multiplier"]

    filtered = [
        (doc, score) for doc, score in scored_docs
        if score >= threshold and score > config["reranker"]["min_score"]
    ]

    max_docs = config["reranker"]["max_docs"]
    top_docs = [doc for doc, _ in filtered][:max_docs]

    context = "\n\n".join([doc.page_content for doc in top_docs])

    return context

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

    if q.startswith(("why", "how", "what")):
        return True

    keywords = [
        "compare", "difference", "explain",
        "reason", "impact", "advantages", "disadvantages",
        "challenges", "improvements", "benefits"
    ]

    return any(k in q for k in keywords)
#---------------
# Multihop
#---------------
def is_multihop_query(query: str) -> bool:
    q = query.lower()

    # strong signals
    if any(k in q for k in ["compare", "difference", "vs"]):
        return True

    # weaker signal (avoid false positives)
    if " and " in q and len(q.split()) > 10:
        return True

    return False

# -------------------------
# HELPER: Sentence Filter
# -------------------------
def filter_relevant_sentences(chunks, query):
    keywords = query.lower().split()
    selected = []

    for doc in chunks:
        text = doc["text"] if isinstance(doc, dict) else str(doc)

        sentences = text.split(".")
        for sent in sentences:
            if any(k in sent.lower() for k in keywords):
                selected.append(sent.strip())

    return ". ".join(selected[:5])
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

    is_multihop = is_multihop_query(query)

    # -------------------------
    # MULTI-HOP LOGIC (UPDATED)
    # -------------------------
    if is_multihop:

        sub_queries = decompose_query(rewritten_query)
        
        print("\n--- DECOMPOSE DEBUG ---")
        print("Raw sub_queries:", sub_queries)

        sub_queries = [q for q in sub_queries if len(q.split()) >= 4]

        print("Filtered sub_queries:", sub_queries)

        sub_queries = sub_queries[:2]

        # ✅ filter weak queries FIRST
        sub_queries = [q for q in sub_queries if len(q.split()) >= 4]

        # ✅ then limit
        sub_queries = sub_queries[:2]

        # ✅ only proceed if valid multi-hop
        if len(sub_queries) > 1:

            sub_answers = []

            for sub_q in sub_queries:

                raw_context = run_retrieval_pipeline(sub_q)
                raw_context = str(raw_context)

                sub_context = "\n\n".join(raw_context.split("\n\n")[:4]).strip()

                if not sub_context:
                    continue  # skip weak sub-query

                prompt_text = prompt.format(
                    context=sub_context,
                    question=sub_q
                )

                ans = llm.invoke(prompt_text)

                # cleaner format for aggregation
                sub_answers.append(ans.strip())

            # ✅ fallback if multi-hop failed
            if not sub_answers:

                fallback_context = run_retrieval_pipeline(rewritten_query)
                if fallback_context:
                    fallback_context = "\n\n".join(fallback_context.split("\n\n")[:3])

                formatted_prompt = prompt.format(
                    context=fallback_context,
                    question=rewritten_query
                )

                answer = llm.invoke(formatted_prompt)

                return {
                    **state,
                    "answer": answer,
                    "llm_used": True,
                    "confidence": confidence,
                    "route": "rag"
                }

            # ✅ aggregation prompt
            sub_answers_text = "\n\n".join(
                [f"Sub-question {i+1}: {ans}" for i, ans in enumerate(sub_answers)]
            )

            final_prompt = f"""You are a helpful assistant. Below are answers to sub-questions about the same topic.
            Your task is to write ONE concise, combined answer using ONLY the information from these sub-answers.
            Do NOT add any external knowledge. Do NOT repeat the sub-questions.

            {sub_answers_text}

            Combined Answer:"""

            final_answer = llm.invoke(final_prompt)

            print("\n--- MULTI-HOP DEBUG ---")
            print("Sub-queries:", sub_queries)

            return {
                **state,
                "answer": final_answer,
                "llm_used": True,
                "confidence": 1.0
            }

        # -------------------------
    # NORMAL PIPELINE
    # -------------------------

    if context and confidence >= 0.75 and not is_reasoning and not is_multihop:

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

        # ✅ Use raw chunks directly (NO filtering)
        if retrieved_chunks:
            context = "\n\n".join(
                chunk["text"] for chunk in retrieved_chunks[:3]
            )

        formatted_prompt = prompt.format(
            context=context,
            question=rewritten_query
        )

        answer = llm.invoke(formatted_prompt)

        mode = "RAG"
        llm_used = True

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
    return {
        **state,
        "answer": answer,
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
    lambda x: x.get("route", "rewrite"),
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