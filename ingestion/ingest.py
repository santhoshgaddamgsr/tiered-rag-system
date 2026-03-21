import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


RAW_DOCS = "data/raw_docs"
VECTOR_STORE = "data/vector_store"


def load_documents():

    docs = []

    for file in os.listdir(RAW_DOCS):

        if file.endswith(".pdf"):

            loader = PyPDFLoader(os.path.join(RAW_DOCS, file))
            docs.extend(loader.load())

    return docs


def main():

    print("Loading documents...")
    docs = load_documents()

    print("Chunking documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en"
    )

    print("Creating FAISS index...")

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local(VECTOR_STORE)

    print("Index created successfully!")


if __name__ == "__main__":
    main()