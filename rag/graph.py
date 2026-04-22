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
from rag.query_rewrite import rewrite_query


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
    model="llama3.1",
    temperature=config["llm"]["temperature"]
)

reranker = CrossEncoder("BAAI/bge-reranker-base")

THRESHOLD = 0.35

# -------------------------
# REWRITE NODE
# -------------------------
def rewrite(state):
    print("➡️ ENTER rewrite")
    original_query = state["question"]

    rewritten_query = rewrite_query(original_query)

    return {
        "question": original_query,
        "rewritten_query": rewritten_query,
        "route": "cache_check"
    }


# -------------------------
# CACHE CHECK NODE
# -------------------------

def cache_check(state):
    print("➡️ ENTER cache_check")
    query = state.get("rewritten_query", state["question"])
    print("CACHE CHECK KEY:", query)
    print(f"\n--- CACHE DEBUG --- hit: {cache.get(query) is not None}")
    
    cached = cache.get(query)

    if not cached:
        cached = cache.semantic_get(query)

    print(f"\n--- CACHE DEBUG --- hit: {cached is not None}")

    if cached:
        return {
            "answer": cached["answer"],
            "sources": cached["sources"],
            "route": "cache",      # ← already present
            "llm_used": False
        }

    return {
        "question": query,
        "route": "router"         # ← already present
    }


# -------------------------
# ROUTER NODE (FIXED)
# -------------------------

def router(state):
    print("➡️ ENTER router")

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
    print("➡️ ENTER retrieve")

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

    print("\n--- RETRIEVE DEBUG ---")
    print("Top 3 docs retrieved:")
    for doc, score in scored_docs[:3]:
        print(f"  Score: {round(float(score),4)} | Source: {doc.metadata.get('source','?')[:60]}")

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
    num_good = len([s for s in rerank_scores if s > 0.85])
    max_docs = max(1, len(rerank_scores))

    confidence = (
        0.5 * top_score +
        0.3 * avg_top3 +
        0.2 * (num_good / max_docs)
    )

    print(f"\n--- CONFIDENCE DEBUG ---")
    print(f"top_score: {round(top_score, 4)}")
    print(f"avg_top3: {round(avg_top3, 4)}")
    print(f"num_good: {num_good} / max_docs: {max_docs}")
    print(f"raw confidence: {round(confidence, 4)}")

    return round(min(confidence, 1.0), 4)
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
# Replace with:
def is_multihop_query(query: str) -> bool:
    q = query.lower()

    # strong signals only
    if any(k in q for k in ["compare", "difference", "vs"]):
        return True

    # ✅ must have action words on BOTH sides of "and"
    # "machine learning and NLP" is a topic, not two separate questions
    if " and " in q and len(q.split()) > 10:
        parts = q.split(" and ")
        # both parts must have a verb/action to be true multihop
        action_words = ["how", "why", "what", "impact", "improve", "affect", "enhance", "work"]
        both_have_action = all(
            any(w in part for w in action_words)
            for part in parts
        )
        return both_have_action

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

def estimate_tokens(text):
    return len(text.split())


def generate(state):
    print("➡️ ENTER generate")

    start_time = time.time()

    # ✅ token counters
    prompt_tokens = 0
    response_tokens = 0

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

        raw_sub_queries = decompose_query(rewritten_query)

        print("\n--- DECOMPOSE DEBUG ---")
        print("Raw sub_queries:", raw_sub_queries)

        # ✅ clean (remove empty / spaces)
        raw_sub_queries = [q.strip() for q in raw_sub_queries if q.strip()]

        # ✅ moderate filtering
        filtered_sub_queries = [
            q for q in raw_sub_queries if len(q.split()) >= 3
        ]

        # ✅ fallback if filtering too strict
        sub_queries = (
            filtered_sub_queries
            if len(filtered_sub_queries) >= 2
            else raw_sub_queries
        )

        # ✅ limit to 3
        sub_queries = sub_queries[:3]

        print("Final sub_queries:", sub_queries)

        sub_query_results = []

        if len(sub_queries) > 1:

            sub_answers = []

            for sub_q in sub_queries:

                raw_context = run_retrieval_pipeline(sub_q)
                raw_context = str(raw_context)

                sub_context = "\n\n".join(raw_context.split("\n\n")[:4]).strip()

                sub_query_results.append({
                    "sub_query": sub_q,
                    "context": sub_context
                })

                if not sub_context:
                    continue

                prompt_text = prompt.format(
                    context=sub_context,
                    question=sub_q
                )

                # ✅ token tracking
                prompt_tokens += estimate_tokens(prompt_text)

                ans = llm.invoke(prompt_text)

                response_tokens += estimate_tokens(ans)

                sub_answers.append(ans.strip())

            # -------------------------
            # MULTI-HOP FALLBACK
            # -------------------------
            if not sub_answers:

                fallback_context = run_retrieval_pipeline(rewritten_query)
                if fallback_context:
                    fallback_context = "\n\n".join(fallback_context.split("\n\n")[:3])

                formatted_prompt = prompt.format(
                    context=fallback_context,
                    question=rewritten_query
                )

                prompt_tokens += estimate_tokens(formatted_prompt)

                answer = llm.invoke(formatted_prompt)

                response_tokens += estimate_tokens(answer)

                latency = round(time.time() - start_time, 3)
                print("DEBUG TOKENS:", prompt_tokens, response_tokens)
                # =========================
                # ✅ ADD CACHE HERE
                # =========================
                rewritten_key = state.get("rewritten_query", state["question"]).strip().lower()
                original_key = state.get("question", rewritten_key).strip().lower()

                print("CACHE SET KEY (rewritten):", rewritten_key)
                print("CACHE SET KEY (original):", original_key)

                cache.set(rewritten_key, {
                    "answer": answer,
                    "sources": state.get("sources", [])
                })

                if original_key != rewritten_key:
                    cache.set(original_key, {
                        "answer": answer,
                        "sources": state.get("sources", [])
                    })
                # =========================

                return {
                    **state,
                    "answer": answer,
                    "llm_used": True,
                    "confidence": confidence,
                    "route": "rag",
                    "sub_queries": sub_queries,
                    "sub_query_results": sub_query_results,
                    "latency": latency,
                    "tokens": {
                        "prompt": prompt_tokens,
                        "response": response_tokens,
                        "total": prompt_tokens + response_tokens
                    }
                }

            # -------------------------
            # AGGREGATION
            # -------------------------
            sub_answers_text = "\n\n".join(
                [f"Sub-question {i+1}: {ans}" for i, ans in enumerate(sub_answers)]
            )

            final_prompt = f"""You are a helpful assistant. Below are answers to sub-questions about the same topic.
Your task is to write ONE concise, combined answer using ONLY the information from these sub-answers.
Do NOT add any external knowledge. Do NOT repeat the sub-questions.

{sub_answers_text}

Combined Answer:"""

            prompt_tokens += estimate_tokens(final_prompt)

            final_answer = llm.invoke(final_prompt)

            response_tokens += estimate_tokens(final_answer)

            print("\n--- MULTI-HOP DEBUG ---")
            print("Sub-queries:", sub_queries)

            latency = round(time.time() - start_time, 3)
            print("DEBUG TOKENS:", prompt_tokens, response_tokens)

            # =========================
            # ✅ ADD THIS BLOCK
            # =========================
            rewritten_key = state.get("rewritten_query", state["question"]).strip().lower()
            original_key = state.get("question", rewritten_key).strip().lower()

            print("CACHE SET KEY (rewritten):", rewritten_key)
            print("CACHE SET KEY (original):", original_key)

            cache.set(rewritten_key, {
                "answer": final_answer,
                "sources": state.get("sources", [])
            })

            if original_key != rewritten_key:
                cache.set(original_key, {
                    "answer": final_answer,
                    "sources": state.get("sources", [])
                })
            # =========================

            return {
                **state,
                "answer": final_answer,
                "llm_used": True,
                "confidence": 1.0,
                "sub_queries": sub_queries,
                "sub_query_results": sub_query_results,
                "latency": latency,
                "tokens": {
                    "prompt": prompt_tokens,
                    "response": response_tokens,
                    "total": prompt_tokens + response_tokens
                }
            }

    # -------------------------
    # NORMAL PIPELINE
    # -------------------------

    if context and confidence >= 0.85 and not is_reasoning and not is_multihop:

        if retrieved_chunks:
            top_chunks = [c["text"].strip() for c in retrieved_chunks[:2]]
            answer = "\n\n".join([c[:200] for c in top_chunks])
        else:
            answer = ""

        mode = "SKIPPED_LLM"
        llm_used = False

    elif context and confidence >= 0.4:

        if retrieved_chunks:
            context = "\n\n".join(
                chunk["text"] for chunk in retrieved_chunks[:3]
            )

        if not context.strip():
            answer = "I don't know based on the provided documents."
            mode = "NO_CONTEXT"
            llm_used = False

        else:
            formatted_prompt = prompt.format(
                context=context,
                question=rewritten_query
            )

            prompt_tokens += estimate_tokens(formatted_prompt)

            answer = llm.invoke(formatted_prompt)

            response_tokens += estimate_tokens(answer)

            mode = "RAG"
            llm_used = True

    else:

        prompt_tokens += estimate_tokens(query) 
        answer = llm.invoke(f"Answer briefly in 3-4 lines: {query}")
        response_tokens += estimate_tokens(answer)
        mode = "LLM_FALLBACK"
        llm_used = True
        route = "llm"

    latency = round(time.time() - start_time, 3)
    llm_used = (prompt_tokens + response_tokens) > 0
    print("DEBUG TOKENS:", prompt_tokens, response_tokens)

    rewritten_key = state.get("rewritten_query", state["question"])
    original_key = state.get("question", rewritten_key)
    sources = state.get("retrieved_chunks", [])

    print("CACHE SET KEY (rewritten):", rewritten_key)
    print("CACHE SET KEY (original):", original_key)

    cache.set(rewritten_key, {
        "answer": answer,
        "sources": sources
    })

    if original_key != rewritten_key:
        cache.set(original_key, {
            "answer": answer,
            "sources": sources
        })

    return {
        **state,
        "route": route if 'route' in locals() else state.get("route", "rag"),
        "answer": answer,
        "llm_used": llm_used,
        "confidence": confidence,
        "sub_queries": [],
        "sub_query_results": [],
        "latency": latency,
        "tokens": {
            "prompt": prompt_tokens,
            "response": response_tokens,
            "total": prompt_tokens + response_tokens
        }
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

# ✅ Start from rewrite
builder.set_entry_point("rewrite")

# ✅ ONLY ONE conditional edge for rewrite
builder.add_conditional_edges(
    "rewrite",
    lambda x: x.get("route", "cache_check"),
    {
        "cache_check": "cache_check",
        "router": "router"
    }
)

# ✅ Cache check → either end or go to router
builder.add_conditional_edges(
    "cache_check",
    lambda x: x.get("route", "router"),
    {
        "cache": "__end__",
        "router": "router"
    }
)

# ✅ Router logic
builder.add_conditional_edges(
    "router",
    lambda x: x["route"],
    {
        "rag": "retrieve",
        "llm": "generate"
    }
)

# ✅ Retrieval → generate
builder.add_edge("retrieve", "generate")

graph = builder.compile()