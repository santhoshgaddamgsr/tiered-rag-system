from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rag.config import config

VECTOR_STORE = "data/vector_store"


def get_retriever():

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en"
    )

    vectorstore = FAISS.load_local(
        VECTOR_STORE,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # vector retrieval candidates
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": config["retrieval"]["vector_k"]}
    )

    # build BM25 corpus
    docs = vectorstore.similarity_search("", k=200)

    bm25 = BM25Retriever.from_documents(docs)

    # BM25 candidates
    bm25.k = config["retrieval"]["bm25_k"]

    return vector_retriever, bm25