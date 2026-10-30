import io
import os
from typing import List, Union, Dict
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf_from_stream(stream: io.BytesIO, filename: str) -> List[Document]:
    """Extract pages from a PDF stream into LangChain Document objects with page metadata."""
    reader = PdfReader(stream)
    docs = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "page": idx + 1,  # 1-indexed page number
                    },
                )
            )
    return docs


def load_txt_from_stream(stream: io.BytesIO, filename: str) -> List[Document]:
    """Extract content from a text file stream into a LangChain Document."""
    content = stream.read().decode("utf-8", errors="replace").strip()
    if not content:
        return []
    return [
        Document(
            page_content=content,
            metadata={
                "source": filename,
                "page": 1,
            },
        )
    ]


def load_file_from_path(file_path: str) -> List[Document]:
    """Load a single PDF or TXT file from a filesystem path."""
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

    with open(file_path, "rb") as f:
        data = io.BytesIO(f.read())

    if ext == ".pdf":
        return load_pdf_from_stream(data, filename)
    elif ext in [".txt", ".md"]:
        return load_txt_from_stream(data, filename)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Only PDF and TXT are supported.")


def load_documents_from_streams(uploaded_files: List) -> List[Document]:
    """
    Load a list of files (e.g. Streamlit UploadedFile objects).
    Each object is expected to have .name and .getvalue() or .read().
    """
    documents = []
    for f in uploaded_files:
        filename = getattr(f, "name", "document")
        ext = os.path.splitext(filename)[1].lower()

        if hasattr(f, "getvalue"):
            data = io.BytesIO(f.getvalue())
        elif hasattr(f, "read"):
            data = io.BytesIO(f.read())
        else:
            continue

        if ext == ".pdf":
            documents.extend(load_pdf_from_stream(data, filename))
        elif ext in [".txt", ".md"]:
            documents.extend(load_txt_from_stream(data, filename))
        else:
            continue

    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    Preserves and propagates page and source metadata to each chunk.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    # Add chunk index to metadata for traceability
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def group_documents_by_filename(documents: List[Document]) -> Dict[str, str]:
    """Combine text of documents/pages by source filename for summaries and cataloging."""
    catalog: Dict[str, List[str]] = {}
    for doc in documents:
        src = doc.metadata.get("source", "unknown")
        catalog.setdefault(src, []).append(doc.page_content)
    return {src: "\n\n".join(texts) for src, texts in catalog.items()}
