import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

from rag.index import get_retriever

load_dotenv()

REFUSAL_STRING = "I am sorry, but the provided documents do not contain information to answer this question."

SYSTEM_PROMPT = """You are a helpful and strictly factual assistant for document question answering.

Rules:
1. Answer the question using ONLY the provided document context below.
2. Do NOT use outside knowledge, speculate, or make up facts.
3. If the answer cannot be determined from the provided context, or if the context does not contain sufficient information, you MUST state exactly:
"{refusal_string}"
4. When answering, provide clear, concise, and direct information citing what is stated in the context.

Context:
{context}
"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


def format_docs_with_sources(docs: List[Document]) -> str:
    """Format retrieved documents with source headers for context injection."""
    formatted_chunks = []
    for i, doc in enumerate(docs):
        src = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", 1)
        formatted_chunks.append(
            f"[Chunk {i+1} | Source: {src} | Page: {page}]\n{doc.page_content}"
        )
    return "\n\n".join(formatted_chunks)


def extract_source_metadata(docs: List[Document]) -> List[Dict[str, Any]]:
    """Extract structured source details (filename, page, content snippet)."""
    sources = []
    for i, doc in enumerate(docs):
        sources.append({
            "chunk_number": i + 1,
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", 1),
            "chunk_id": doc.metadata.get("chunk_id", i),
            "content": doc.page_content,
        })
    return sources


def get_llm(model_name: str = "gemini-2.5-flash", temperature: float = 0.0, api_key: Optional[str] = None) -> ChatGoogleGenerativeAI:
    """Instantiate Google Gemini chat model."""
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "Google API key not found. Please set GOOGLE_API_KEY in your .env file or environment."
        )
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=key,
    )


def answer_question(
    question: str,
    vectorstore: FAISS,
    top_k: int = 4,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.0,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the Simple RAG pipeline:
    1. Retrieve top-k chunks from FAISS vectorstore.
    2. Format prompt with strict grounding instructions.
    3. Generate answer strictly from context.
    4. Return answer and detailed source chunks with page numbers.
    """
    retriever = get_retriever(vectorstore, top_k=top_k)
    retrieved_docs: List[Document] = retriever.invoke(question)

    if not retrieved_docs:
        return {
            "answer": REFUSAL_STRING,
            "sources": [],
            "retrieved_docs": [],
        }

    context_str = format_docs_with_sources(retrieved_docs)
    llm = get_llm(model_name=model_name, temperature=temperature, api_key=api_key)

    chain = QA_PROMPT | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context_str,
        "question": question,
        "refusal_string": REFUSAL_STRING,
    })

    return {
        "answer": answer.strip(),
        "sources": extract_source_metadata(retrieved_docs),
        "retrieved_docs": retrieved_docs,
    }
