import streamlit as st
import pandas as pd

st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")

st.title("📊 RAG Evaluation Dashboard")

# -------------------------
# LOAD DATA
# -------------------------

@st.cache_data
def load_data():
    return pd.read_csv("evaluation/results.csv")

try:
    df = load_data()
except:
    st.error("No results.csv found. Run evaluation first.")
    st.stop()

# -------------------------
# METRICS SUMMARY
# -------------------------

st.subheader("🔹 Overall Metrics")
import matplotlib.pyplot as plt

st.subheader("📈 Metrics Distribution")

fig = plt.figure()

metrics = [
    df["faithfulness"].mean(),
    df["answer_relevancy"].mean(),
    df["context_precision"].mean(),
    df["context_recall"].mean()
]

labels = ["Faithfulness", "Answer Rel.", "Precision", "Recall"]

plt.bar(labels, metrics)
plt.ylabel("Score")

st.pyplot(fig)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Faithfulness", round(df["faithfulness"].mean(), 3))
col2.metric("Answer Relevancy", round(df["answer_relevancy"].mean(), 3))
col3.metric("Context Precision", round(df["context_precision"].mean(), 3))
col4.metric("Context Recall", round(df["context_recall"].mean(), 3))

# -------------------------
# FAILURES
# -------------------------

st.subheader("❌ Failure Cases")
st.subheader("❌ Failure Cases (Precision < 0.2)")

failures = df[df["context_precision"] < 0.2]

st.write(f"Total failures: {len(failures)}")

st.dataframe(failures[["question", "context_precision"]])

failures = df[df["context_precision"] < 0.2]

st.dataframe(failures)

st.subheader("🚨 Lowest Precision Queries")

worst = df.sort_values("context_precision").head(5)

st.dataframe(worst[["question", "context_precision"]])

# -------------------------
# ALL RESULTS
# -------------------------
st.subheader("🔍 Inspect Query")

selected_q = st.selectbox(
    "Select a question",
    df["question"]
)

row = df[df["question"] == selected_q].iloc[0]

st.write("### Question")
st.write(row["question"])

st.write("### Answer")
st.write(row["answer"])

st.write("### Ground Truth")
st.write(row["ground_truth"])

st.write("### Context")
st.write(row["contexts"])

import ast

st.write("### Retrieved Chunks (with scores)")

try:
    chunks = ast.literal_eval(row["retrieved_chunks"])
    
    for i, chunk in enumerate(chunks):
        st.write(f"**Chunk {i+1}**")
        st.write(f"Score: {round(chunk['score'], 4)}")
        st.write(f"Source: {chunk['source']} | Page: {chunk['page']}")
        st.write(chunk["text"])
        st.write("---")

except:
    st.write("No chunk data available")

st.subheader("📋 Full Results")

st.dataframe(df)