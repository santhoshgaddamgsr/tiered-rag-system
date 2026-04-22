import pandas as pd
import requests

from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)

from ragas.run_config import RunConfig

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings


API_URL = "http://127.0.0.1:8000/query"


def query_rag(question):
    """
    Send query to RAG API and retrieve answer + contexts
    """

    response = requests.post(
        API_URL,
        json={"question": question}
    )

    result = response.json()

    contexts = []

    # ✅ FIX: use reranker-filtered chunks (actual context used)
    if result.get("reranker_filtered_chunks"):
        contexts = result["reranker_filtered_chunks"]

    # 🔁 fallback
    elif result.get("retrieved_chunks"):
        contexts = [c["text"] for c in result["retrieved_chunks"]]

    return result["answer"], contexts, result.get("retrieved_chunks", [])


def main():

    print("\nLoading evaluation dataset...\n")

    df = pd.read_csv("evaluation/dataset.csv")

    answers = []
    contexts = []
    all_chunks = []

    print("Running queries through RAG system...\n")

    for q in df["question"]:

        answer, ctx, chunks = query_rag(q)

        answers.append(answer)
        contexts.append(ctx)
        all_chunks.append(chunks)

    print("Preparing evaluation dataset...\n")

    eval_data = Dataset.from_dict({
        "question": df["question"].tolist(),
        "answer": answers,
        "contexts": contexts,
        "ground_truth": df["ground_truth"].tolist()
    })

    print("Loading evaluator LLM...\n")

    evaluator_llm = OllamaLLM(
        model="qwen2.5-coder:7b",
        temperature=0
    )

    print("Loading embedding model...\n")

    evaluator_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en"
    )

    print("Running RAGAS evaluation...\n")

    # Prevent timeouts when using local LLM
    run_config = RunConfig(
        max_workers=1,
        timeout=180
    )

    results = evaluate(
        eval_data,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall()
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config
    )

    print("\n==============================")
    print("   Evaluation Results")
    print("==============================\n")

    print(results)

    # -------------------------
    # SAVE RESULTS FOR DASHBOARD
    # -------------------------

    df_results = pd.DataFrame({
        "question": df["question"],
        "answer": answers,
        "contexts": [" | ".join(ctx) for ctx in contexts],
        "ground_truth": df["ground_truth"],
        "retrieved_chunks": all_chunks,
        
        "faithfulness": results["faithfulness"],
        "answer_relevancy": results["answer_relevancy"],
        "context_precision": results["context_precision"],
        "context_recall": results["context_recall"]
    })

    df_results.to_csv("evaluation/results.csv", index=False)

    print("\nResults saved to evaluation/results.csv\n")


if __name__ == "__main__":
    main()