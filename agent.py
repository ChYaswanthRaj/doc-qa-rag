import os
import ast
import operator
import math
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langgraph.prebuilt import create_react_agent

from rag.qa import REFUSAL_STRING, extract_source_metadata, get_llm
from rag.index import get_retriever

load_dotenv()

# Safe arithmetic evaluator operators
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_ALLOWED_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
}


def _safe_eval(expr: str) -> Any:
    """Safely evaluate mathematical expressions without eval()."""
    clean_expr = expr.strip().replace("^", "**")
    node = ast.parse(clean_expr, mode="eval").body

    def _evaluate_node(n):
        if isinstance(n, ast.Constant):
            return n.value
        elif isinstance(n, ast.BinOp):
            op_type = type(n.op)
            if op_type not in _ALLOWED_OPS:
                raise ValueError(f"Operator {op_type} not supported.")
            return _ALLOWED_OPS[op_type](_evaluate_node(n.left), _evaluate_node(n.right))
        elif isinstance(n, ast.UnaryOp):
            op_type = type(n.op)
            if op_type not in _ALLOWED_OPS:
                raise ValueError(f"Unary operator {op_type} not supported.")
            return _ALLOWED_OPS[op_type](_evaluate_node(n.operand))
        elif isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name) and n.func.id in _ALLOWED_FUNCS:
                args = [_evaluate_node(arg) for arg in n.args]
                return _ALLOWED_FUNCS[n.func.id](*args)
            raise ValueError(f"Function {getattr(n.func, 'id', 'unknown')} not allowed.")
        elif isinstance(n, ast.List):
            return [_evaluate_node(elt) for elt in n.elts]
        elif isinstance(n, ast.Tuple):
            return tuple(_evaluate_node(elt) for elt in n.elts)
        raise ValueError(f"Unsupported expression construct: {type(n).__name__}")

    return _evaluate_node(node)


def create_agent_system_prompt() -> str:
    return f"""You are an advanced agentic document assistant with access to tools.
You answer user questions strictly based on the uploaded documents.

CORE PRINCIPLES & RULES:
1. STRICT GROUNDING: Answer questions ONLY using facts and figures explicitly stated in the documents retrieved via tools. Do NOT assume, extrapolate, or bring in outside knowledge.
2. SOURCE CITATIONS: Whenever stating a fact or answering, you MUST cite the source document and page number (e.g. "[financial_report.pdf, Page 3]").
3. SEARCH & RE-WRITE STRATEGY (Maximum 3 search attempts):
   - Call `search_documents(query)` with a clear keyword search to find relevant information.
   - If the returned context does NOT contain enough information to answer the question, or is ambiguous:
     * Analyze what was missing.
     * REWRITE the query using different keywords, broader terms, or specific synonyms.
     * Call `search_documents(query)` again.
     * You can make up to 3 search attempts total.
4. REFUSAL RULE:
   - If after searching (up to 3 attempts) the documents do not contain the answer, you MUST refuse and respond with exactly:
     "{REFUSAL_STRING}"
   - Do NOT invent or fabricate an answer.
5. ARITHMETIC / CALCULATOR:
   - Whenever a calculation is required (e.g., percentage change, sum, difference, ratios), retrieve the raw values from documents first, then call `calculator(expression)` to verify the calculation.

TOOLS AVAILABLE:
- search_documents(query): Retrieve relevant document chunks with source filename and page numbers.
- list_documents(): List all currently available documents and their metadata.
- summarize_document(filename): Produce an executive summary of a specific document.
- calculator(expression): Safely evaluate mathematical expressions.
"""


def build_agent_tools(
    vectorstore: FAISS,
    top_k: int = 4,
    doc_catalog: Optional[Dict[str, str]] = None,
    doc_metadata: Optional[List[Dict[str, Any]]] = None,
    llm: Optional[ChatGoogleGenerativeAI] = None,
    collected_sources: Optional[List[Document]] = None,
):
    """
    Build the four required tools for the agent:
    1. search_documents(query)
    2. list_documents()
    3. summarize_document(filename)
    4. calculator(expression)
    """
    if collected_sources is None:
        collected_sources = []

    retriever = get_retriever(vectorstore, top_k=top_k)

    @tool
    def search_documents(query: str) -> str:
        """Search the indexed documents for chunks relevant to the query. Returns chunks with source filenames and page numbers."""
        docs: List[Document] = retriever.invoke(query)
        if not docs:
            return f"No relevant document chunks found for query: '{query}'."

        formatted_chunks = []
        for i, doc in enumerate(docs):
            collected_sources.append(doc)
            src = doc.metadata.get("source", "Unknown Document")
            page = doc.metadata.get("page", 1)
            formatted_chunks.append(
                f"[Result {i+1} | Source: {src} | Page: {page}]\n{doc.page_content}"
            )
        return "\n\n".join(formatted_chunks)

    @tool
    def list_documents() -> str:
        """List all indexed document filenames and summary information."""
        if not doc_catalog:
            return "No documents are currently indexed."

        lines = ["Indexed Documents:"]
        for filename, text in doc_catalog.items():
            char_count = len(text)
            lines.append(f"- {filename} (approx {char_count} characters)")
        return "\n".join(lines)

    @tool
    def summarize_document(filename: str) -> str:
        """Summarize the entire content of a specific document by its filename."""
        if not doc_catalog:
            return "No documents available to summarize."

        matched_key = None
        for name in doc_catalog.keys():
            if filename.lower() in name.lower() or name.lower() in filename.lower():
                matched_key = name
                break

        if not matched_key:
            available = ", ".join(doc_catalog.keys())
            return f"Document '{filename}' not found. Available documents: {available}"

        doc_text = doc_catalog[matched_key]
        # Truncate text if very large to prevent token overflow
        truncated_text = doc_text[:8000]

        if llm:
            summary_prompt = (
                f"Summarize the following document content concisely, highlighting main topics and key facts:\n\n"
                f"Document: {matched_key}\n\nContent:\n{truncated_text}"
            )
            response = llm.invoke([HumanMessage(content=summary_prompt)])
            summary = response.content if isinstance(response.content, str) else str(response.content)
            return f"Summary of {matched_key}:\n{summary.strip()}"

        return f"Document '{matched_key}' Preview:\n{truncated_text[:1000]}..."

    @tool
    def calculator(expression: str) -> str:
        """Safely compute mathematical expressions like '250000 * 0.15' or '(120 - 95) / 95 * 100'."""
        try:
            result = _safe_eval(expression)
            return f"{expression} = {result}"
        except Exception as e:
            return f"Error evaluating expression '{expression}': {str(e)}"

    return [search_documents, list_documents, summarize_document, calculator]


def run_agent_qa(
    question: str,
    vectorstore: FAISS,
    top_k: int = 4,
    doc_catalog: Optional[Dict[str, str]] = None,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.0,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute the Agentic RAG pipeline:
    - Sets up tools (search_documents with re-try tracking, list_documents, summarize_document, calculator).
    - Runs the LangGraph ReAct agent loop.
    - Captures the agent trace (tool calls and results).
    - Deduplicates and formats source chunks.
    """
    llm = get_llm(model_name=model_name, temperature=temperature, api_key=api_key)
    collected_sources: List[Document] = []

    tools = build_agent_tools(
        vectorstore=vectorstore,
        top_k=top_k,
        doc_catalog=doc_catalog,
        llm=llm,
        collected_sources=collected_sources,
    )

    system_prompt = create_agent_system_prompt()
    agent_graph = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SystemMessage(content=system_prompt),
    )

    # Execute graph
    response = agent_graph.invoke({"messages": [HumanMessage(content=question)]})
    messages = response.get("messages", [])

    # Extract trace and final answer
    trace: List[Dict[str, Any]] = []
    final_answer = ""
    step_num = 1

    # Map of tool_call_id to tool output
    tool_outputs: Dict[str, str] = {}
    tool_messages_list: List[str] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
            if getattr(msg, "tool_call_id", None):
                tool_outputs[msg.tool_call_id] = content_str
            tool_messages_list.append(content_str)

    for msg in messages:
        if isinstance(msg, AIMessage):
            # Check for tool calls
            tool_calls = getattr(msg, "tool_calls", []) or []
            for tc in tool_calls:
                call_id = tc.get("id", "")
                tool_name = tc.get("name", "unknown_tool")
                tool_args = tc.get("args", {})
                if call_id and call_id in tool_outputs:
                    tool_output = tool_outputs[call_id]
                elif (step_num - 1) < len(tool_messages_list):
                    tool_output = tool_messages_list[step_num - 1]
                else:
                    tool_output = "Completed"
                trace.append({
                    "step": step_num,
                    "tool": tool_name,
                    "input": tool_args,
                    "output": tool_output,
                })
                step_num += 1

            # Check if this message has final textual content
            if msg.content and not tool_calls:
                if isinstance(msg.content, str):
                    final_answer = msg.content
                elif isinstance(msg.content, list):
                    text_parts = [
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in msg.content
                    ]
                    final_answer = "\n".join(text_parts)

    if not final_answer:
        # Fallback to last message content if present
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                final_answer = str(msg.content)
                break
        if not final_answer:
            final_answer = REFUSAL_STRING

    # Deduplicate sources
    unique_sources = []
    seen = set()
    for doc in collected_sources:
        src_key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content.strip()[:100])
        if src_key not in seen:
            seen.add(src_key)
            unique_sources.append(doc)

    return {
        "answer": final_answer.strip(),
        "trace": trace,
        "sources": extract_source_metadata(unique_sources),
        "tool_calls_count": len(trace),
    }
