import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

VECTOR_STORE = "data/vector_store"

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en"
)

vectorstore = FAISS.load_local(
    VECTOR_STORE,
    embeddings,
    allow_dangerous_deserialization=True
)


def similarity_topk(query, k=5):

    docs_scores = vectorstore.similarity_search_with_score(query, k=k)

    # convert numpy floats → python floats
    distances = [float(score) for _, score in docs_scores]

    avg_distance = float(sum(distances) / len(distances))

    return distances, avg_distance