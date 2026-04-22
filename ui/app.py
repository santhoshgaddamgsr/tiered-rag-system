import streamlit as st
import requests
import time

st.set_page_config(page_title="Intelligent RAG", layout="wide")

st.title("🧠 Intelligent RAG System")
st.caption("Confidence-based + Query-aware + Multi-hop reasoning")

# -------------------------
# INPUT
# -------------------------
query = st.text_input("Enter your question:")

# -------------------------
# MAIN ACTION
# -------------------------
if st.button("Ask") and query:

    with st.spinner("Processing..."):

        try:
            start_time = time.time()

            response = requests.post(
                "http://127.0.0.1:8000/query",
                json={"question": query}
            )

            end_time = time.time()
            latency = round(end_time - start_time, 2)

            data = response.json()

        except:
            st.error("❌ Backend not running. Please start FastAPI server.")
            st.stop()

    # -------------------------
    # REWRITTEN QUERY
    # -------------------------
    st.subheader("🔄 Rewritten Query")
    st.info(data.get("rewritten_query", "N/A"))

    # -------------------------
    # MULTI-HOP SUB-QUERIES
    # -------------------------
    if data.get("sub_queries"):
        st.subheader("🧩 Decomposed Sub-Queries")
        for q in data["sub_queries"]:
            st.write(f"- {q}")

    # -------------------------
    # ANSWER (streaming effect)
    # -------------------------
    st.subheader("📌 Answer")

    placeholder = st.empty()
    streamed_text = ""

    answer = data.get("answer", "")  # ✅ SAFE GUARD

    for word in answer.split():
        streamed_text += word + " "
        placeholder.markdown(streamed_text)
        time.sleep(0.02)

    # -------------------------
    # METADATA
    # -------------------------
    col1, col2, col3, col4 = st.columns(4)

    route = data.get("route", "N/A")

    with col1:
        if route == "rag":
            st.success(f"Route: {route.upper()}")
        elif route == "multi_hop":
            st.warning(f"Route: {route.upper()}")
        elif route == "cache":
            st.info(f"Route: {route.upper()}")   # ✅ cleaner for cache
        else:
            st.error(f"Route: {route.upper()}")

    # ✅ FIXED CONFIDENCE (no crash)
    with col2:
        confidence = data.get("confidence")

        if confidence is not None:
            st.metric("Confidence", round(confidence, 2))
        else:
            st.metric("Confidence", "N/A")

    # ✅ FIXED LATENCY (safe fallback)
    with col3:
        lat = data.get("latency")
        if lat is not None:
            st.metric("Latency (s)", lat)
        else:
            st.metric("Latency (s)", latency)

    # -------------------------
    # TOKENS
    # -------------------------
    with col4:
        llm_used = data.get("llm_used", False)

        if llm_used:
            approx_tokens = int(len(answer.split()) * 1.3)
            st.metric("Tokens (approx)", approx_tokens)
        else:
            st.metric("Tokens", "Skipped")

    # -------------------------
    # TABS
    # -------------------------
    tab1, tab2, tab3 = st.tabs(
        ["📚 Sources", "🔍 Reranker Filtered Chunks", "🧩 Debug"]
    )

    # -------------------------
    # SOURCES
    # -------------------------
    with tab1:
        sources = data.get("sources", [])
        if sources:
            for src in sources:
                st.markdown(f"📄 **{src['source']}** (page {src['page']})")
        else:
            st.info("No sources available")

    # -------------------------
    # CHUNKS (POST-RERANK)
    # -------------------------
    with tab2:
        st.caption(
            "These chunks passed reranker filtering (final context used for answer)"
        )

        chunks = data.get("retrieved_chunks", [])
        if chunks:
            for chunk in chunks:
                st.markdown(f"**Score:** {round(chunk['score'], 3)}")
                st.info(chunk["text"][:300] + "...")
        else:
            st.warning("No chunks passed reranker threshold (low relevance)")

    # -------------------------
    # DEBUG PANEL
    # -------------------------
    with tab3:
        st.subheader("🔍 System Debug")

        if data.get("sub_query_results"):
            st.subheader("🧩 Multi-hop Retrieval Details")

            for item in data["sub_query_results"]:
                st.markdown(f"### Sub-query: {item['sub_query']}")

                if item["context"]:
                    st.info(item["context"][:400] + "...")
                else:
                    st.warning("No relevant context found")

        st.write("Rewritten Query:", data.get("rewritten_query"))
        st.write("Confidence:", data.get("confidence"))
        st.write("LLM Used:", data.get("llm_used"))

        st.subheader("🔍 Retrieval vs Filtering")
        st.write(
            "Raw retrieved candidates:",
            len(data.get("top_k_distances") or [])
        )
        st.write(
            "Final chunks after reranker:",
            len(data.get("retrieved_chunks") or [])
        )

        if data.get("sub_queries"):
            st.subheader("🧩 Sub-Queries (Debug)")
            for q in data["sub_queries"]:
                st.write(f"- {q}")

        with st.expander("Advanced Debug"):
            st.write("Top Distances:", data.get("top_k_distances"))
            st.write("Reranker Scores:", data.get("reranker_scores"))
            st.write("Tokens Debug:", data.get("tokens"))