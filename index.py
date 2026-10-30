import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


def get_embeddings_model(api_key: Optional[str] = None) -> GoogleGenerativeAIEmbeddings:
    """
    Instantiate Google Gemini embeddings model using the provided key or GOOGLE_API_KEY from environment.
    """
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "Google API key not found. Please set GOOGLE_API_KEY in your .env file or environment."
        )
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=key,
    )


def build_vectorstore(
    chunks: List[Document],
    embeddings: Optional[Embeddings] = None,
    api_key: Optional[str] = None,
) -> FAISS:
    """
    Create a FAISS vector index from document chunks using Google GenAI embeddings.
    """
    if not chunks:
        raise ValueError("Cannot build vector store from empty document chunks.")

    if embeddings is None:
        embeddings = get_embeddings_model(api_key=api_key)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def get_retriever(
    vectorstore: FAISS,
    top_k: int = 4,
):
    """
    Return a similarity retriever configured to return top_k most relevant chunks.
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )


def save_vectorstore(vectorstore: FAISS, folder_path: str) -> None:
    """Save FAISS index and docstore locally."""
    vectorstore.save_local(folder_path)


def load_vectorstore(
    folder_path: str,
    embeddings: Optional[Embeddings] = None,
    api_key: Optional[str] = None,
) -> FAISS:
    """Load FAISS index and docstore from a local folder."""
    if embeddings is None:
        embeddings = get_embeddings_model(api_key=api_key)
    return FAISS.load_local(
        folder_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
