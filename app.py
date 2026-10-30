import sys, types
try:
    import loader, index, qa, agent
    rag = types.ModuleType('rag')
    rag.loader, rag.index, rag.qa, rag.agent = loader, index, qa, agent
    sys.modules['rag'] = rag
    sys.modules['rag.loader'] = loader
    sys.modules['rag.index'] = index
    sys.modules['rag.qa'] = qa
    sys.modules['rag.agent'] = agent
except ImportError:
    pass
    import os
import streamlit as st
from dotenv import load_dotenv

from rag.loader import load_documents_from_streams, split_documents, group_documents_by_filename
from rag.index import build_vectorstore
from rag.qa import answer_question
from rag.agent import run_agent_qa

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Agentic RAG Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Modern Glassmorphism & Glow Styles
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Glassmorphism Hero Banner */
    .hero-banner {
        position: relative;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.015) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5), inset 0 1px 0 0 rgba(255, 255, 255, 0.15);
        overflow: hidden;
    }

    /* Ambient glow behind hero */
    .hero-banner::before {
        content: "";
        position: absolute;
        top: -60px;
        right: -40px;
        width: 320px;
        height: 320px;
        background: radial-gradient(circle, rgba(255, 94, 58, 0.3) 0%, rgba(255, 42, 104, 0.12) 50%, transparent 75%);
        border-radius: 50%;
        filter: blur(45px);
        pointer-events: none;
        z-index: 0;
    }

    .hero-content {
        position: relative;
        z-index: 1;
    }

    /* Glowing Vibrant Gradient Title */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.15;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #FF3366 0%, #FF5E3A 30%, #FFAA00 70%, #FF007F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 28px rgba(255, 94, 58, 0.45));
        display: inline-block;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        line-height: 1.6;
        max-width: 820px;
        margin-bottom: 1rem;
    }

    /* Feature Badges in Hero */
    .hero-badges {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        align-items: center;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #e2e8f0;
        backdrop-filter: blur(8px);
        letter-spacing: 0.02em;
    }

    .hero-badge.active {
        background: rgba(255, 94, 58, 0.15);
        border-color: rgba(255, 94, 58, 0.45);
        color: #ff8a65;
        box-shadow: 0 0 16px rgba(255, 94, 58, 0.25);
    }

    /* Live Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.8rem;
    }

    .glass-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.01) 100%);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .glass-card:hover {
        border-color: rgba(255, 94, 58, 0.4);
        box-shadow: 0 12px 30px rgba(255, 94, 58, 0.15);
        transform: translateY(-2px);
    }

    .glass-card::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255, 94, 58, 0.5), transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .glass-card:hover::after {
        opacity: 1;
    }

    .metric-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.4rem;
    }

    .metric-title {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        font-weight: 700;
    }

    .metric-icon {
        font-size: 1.1rem;
    }

    .metric-number {
        font-size: 1.85rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }

    .metric-footer {
        font-size: 0.74rem;
        color: #64748b;
        margin-top: 0.35rem;
        font-weight: 500;
    }

    .metric-footer.ready {
        color: #10b981;
        font-weight: 600;
    }

    .metric-footer.pending {
        color: #f59e0b;
        font-weight: 600;
    }

    /* Glowing Red/Orange Process Button in Sidebar */
    div[data-testid="stSidebar"] button[kind="primary"] {
        background: linear-gradient(135deg, #FF3366 0%, #FF5E3A 50%, #FF8A00 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        color: #ffffff !important;
        font-weight: 750 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.02em !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.25rem !important;
        box-shadow: 0 4px 22px rgba(255, 94, 58, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    div[data-testid="stSidebar"] button[kind="primary"]:hover {
        background: linear-gradient(135deg, #FF1E56 0%, #FF451A 50%, #FF7700 100%) !important;
        box-shadow: 0 8px 32px rgba(255, 94, 58, 0.85), 0 0 18px rgba(255, 138, 0, 0.6) !important;
        transform: translateY(-2px) !important;
    }

    div[data-testid="stSidebar"] button[kind="primary"]:active {
        transform: translateY(1px) !important;
        box-shadow: 0 2px 12px rgba(255, 94, 58, 0.4) !important;
    }

    /* Source Citation & Trace Card Styling */
    .source-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 4px solid #FF5E3A;
        padding: 0.85rem 1.1rem;
        border-radius: 8px;
        margin-top: 0.6rem;
        font-size: 0.88rem;
        color: #e2e8f0;
    }

    .trace-card {
        background: rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
        font-family: monospace;
        font-size: 0.84rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "doc_catalog" not in st.session_state:
    st.session_state.doc_catalog = {}

if "indexed_stats" not in st.session_state:
    st.session_state.indexed_stats = {"docs": 0, "pages": 0, "chunks": 0}

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ RAG Control Center")

    # API Key check
    env_key = os.getenv("GOOGLE_API_KEY", "")
    with st.expander("🔑 Google Gemini API Key", expanded=not bool(env_key)):
        user_key = st.text_input(
            "API Key",
            value=env_key,
            type="password",
            help="Loaded from .env by default. You can override it here.",
            label_visibility="collapsed",
        )
        if user_key:
            api_key = user_key
            st.success("API Key configured", icon="✅")
        else:
            api_key = None
            st.error("Missing GOOGLE_API_KEY in .env or input.", icon="⚠️")

    st.markdown("---")
    st.markdown("### 🤖 Reasoning Pipeline")
    agent_mode = st.toggle("Agent Mode (ReAct + Rewrites)", value=True)
    if agent_mode:
        st.caption("✨ **Active**: Autonomous ReAct agent with tools (`search`, `list`, `summarize`, `calculator`) & query rewrite retries.")
    else:
        st.caption("⚡ **Active**: Direct single-pass similarity retrieval with strict context grounding.")

    st.markdown("---")
    st.markdown("### 📁 Ingestion & Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Upload one or multiple documents to build the FAISS vector index.",
    )

    st.markdown("---")
    st.markdown("### 🎛️ Retrieval Settings")
    chunk_size = st.slider("Chunk Size (characters)", min_value=200, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=500, value=200, step=50)
    top_k = st.slider("Top-K Chunks to Retrieve", min_value=1, max_value=10, value=4, step=1)

    model_name = st.selectbox(
        "Gemini Model",
        options=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
        index=0,
    )

    st.markdown("---")
    # Glowing Process & Index Button
    if st.button("⚡ Process & Index Documents", type="primary", width="stretch"):
        if not api_key:
            st.error("Please configure a Google Gemini API Key before indexing.")
        elif not uploaded_files:
            st.warning("Please upload at least one PDF or TXT file.")
        else:
            with st.spinner("Processing documents, extracting pages, and creating FAISS index..."):
                try:
                    # 1. Load documents
                    raw_docs = load_documents_from_streams(uploaded_files)
                    if not raw_docs:
                        st.error("Could not extract any readable text from the uploaded files.")
                    else:
                        # 2. Split documents
                        chunks = split_documents(
                            raw_docs,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )
                        # 3. Create FAISS Vectorstore
                        vectorstore = build_vectorstore(chunks, api_key=api_key)
                        catalog = group_documents_by_filename(raw_docs)

                        # Save to session
                        st.session_state.vectorstore = vectorstore
                        st.session_state.doc_catalog = catalog
                        st.session_state.indexed_stats = {
                            "docs": len(catalog),
                            "pages": len(raw_docs),
                            "chunks": len(chunks),
                        }
                        st.success(f"Indexed {len(catalog)} file(s), {len(raw_docs)} page(s), {len(chunks)} chunks!", icon="🚀")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error during indexing: {str(e)}")

    if st.button("🧹 Clear Chat History", width="stretch"):
        st.session_state.messages = []
        st.rerun()

# --- Main Interface ---

# 1. Modern Glassmorphism Hero Banner with Glowing Vibrant Gradient Title
active_mode_badge = "✨ Agent Mode (Tools + Rewrite)" if agent_mode else "⚡ Simple RAG (Direct)"
engine_status_label = "Active & Ready" if st.session_state.vectorstore is not None else "Standby (Upload Docs)"
engine_status_class = "ready" if st.session_state.vectorstore is not None else "pending"

st.markdown(f"""
<div class="hero-banner">
    <div class="hero-content">
        <div class="hero-title">⚡ Agentic RAG Document Q&A</div>
        <div class="hero-subtitle">
            Upload your documents, inspect real-time FAISS vector embeddings, and interrogate knowledge with strict grounding, source citations, and autonomous multi-step reasoning.
        </div>
        <div class="hero-badges">
            <span class="hero-badge active">{active_mode_badge}</span>
            <span class="hero-badge">🧠 {model_name}</span>
            <span class="hero-badge">🧬 text-embedding-004</span>
            <span class="hero-badge">🔒 100% Strict Grounding</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 2. Live Metric Cards Grid
stats = st.session_state.indexed_stats
status_text = "🟢 ONLINE" if st.session_state.vectorstore is not None else "⚪ STANDBY"
status_sub = "Index loaded & searchable" if st.session_state.vectorstore is not None else "Waiting for document upload"

st.markdown(f"""
<div class="metric-grid">
    <div class="glass-card">
        <div class="metric-header">
            <span class="metric-title">Indexed Files</span>
            <span class="metric-icon">📁</span>
        </div>
        <div class="metric-number">{stats['docs']}</div>
        <div class="metric-footer">Uploaded & cataloged</div>
    </div>
    <div class="glass-card">
        <div class="metric-header">
            <span class="metric-title">Total Pages</span>
            <span class="metric-icon">📑</span>
        </div>
        <div class="metric-number">{stats['pages']}</div>
        <div class="metric-footer">PDF & text pages parsed</div>
    </div>
    <div class="glass-card">
        <div class="metric-header">
            <span class="metric-title">Semantic Chunks</span>
            <span class="metric-icon">🧩</span>
        </div>
        <div class="metric-number">{stats['chunks']}</div>
        <div class="metric-footer">Size: {chunk_size} | Overlap: {chunk_overlap}</div>
    </div>
    <div class="glass-card">
        <div class="metric-header">
            <span class="metric-title">Engine State</span>
            <span class="metric-icon">⚡</span>
        </div>
        <div class="metric-number" style="font-size: 1.4rem; padding-top: 0.35rem;">{status_text}</div>
        <div class="metric-footer {engine_status_class}">{status_sub}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# 3. Chat History Display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show agent trace if available
        if msg.get("trace"):
            trace_count = len(msg["trace"])
            with st.expander(f"🕵️ Agent Execution Trace ({trace_count} step{'s' if trace_count > 1 else ''})"):
                for step in msg["trace"]:
                    st.markdown(f"**Step {step['step']}: Tool `{step['tool']}`**")
                    st.caption(f"Input: `{step['input']}`")
                    st.code(step['output'], language="markdown")

        # Show source citations
        if msg.get("sources"):
            source_count = len(msg["sources"])
            with st.expander(f"📚 Source Citations & Context ({source_count} chunk{'s' if source_count > 1 else ''})"):
                for s in msg["sources"]:
                    st.markdown(f"**Source:** `{s['source']}` &nbsp;|&nbsp; **Page:** `{s['page']}` &nbsp;|&nbsp; **Chunk ID:** `{s['chunk_number']}`")
                    st.markdown(f'<div class="source-box">{s["content"]}</div>', unsafe_allow_html=True)

# 4. User Question Chat Input
if prompt := st.chat_input("Ask a question about your indexed documents..."):
    # Append user question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        if st.session_state.vectorstore is None:
            err_msg = "⚠️ No documents are indexed yet. Please upload files in the sidebar and click **⚡ Process & Index Documents**."
            st.warning(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})
        elif not api_key:
            err_msg = "⚠️ Google Gemini API key is missing. Please set GOOGLE_API_KEY in your .env file or sidebar."
            st.error(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})
        else:
            spinner_msg = "Agent reasoning with tools & query rewriting..." if agent_mode else "Searching vectorstore and retrieving context..."
            with st.spinner(spinner_msg):
                try:
                    if agent_mode:
                        result = run_agent_qa(
                            question=prompt,
                            vectorstore=st.session_state.vectorstore,
                            top_k=top_k,
                            doc_catalog=st.session_state.doc_catalog,
                            model_name=model_name,
                            api_key=api_key,
                        )
                        answer = result["answer"]
                        trace = result["trace"]
                        sources = result["sources"]

                        st.markdown(answer)

                        if trace:
                            trace_steps_label = f"({len(trace)} step{'s' if len(trace) > 1 else ''})"
                            with st.expander(f"🕵️ Agent Execution Trace {trace_steps_label}"):
                                for step in trace:
                                    st.markdown(f"**Step {step['step']}: Tool `{step['tool']}`**")
                                    st.caption(f"Input: `{step['input']}`")
                                    st.code(step['output'], language="markdown")

                        if sources:
                            with st.expander(f"📚 Source Citations & Context ({len(sources)} chunk{'s' if len(sources) > 1 else ''})"):
                                for s in sources:
                                    st.markdown(f"**Source:** `{s['source']}` &nbsp;|&nbsp; **Page:** `{s['page']}` &nbsp;|&nbsp; **Chunk ID:** `{s['chunk_number']}`")
                                    st.markdown(f'<div class="source-box">{s["content"]}</div>', unsafe_allow_html=True)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "mode": "agent",
                            "trace": trace,
                            "sources": sources,
                        })

                    else:
                        result = answer_question(
                            question=prompt,
                            vectorstore=st.session_state.vectorstore,
                            top_k=top_k,
                            model_name=model_name,
                            api_key=api_key,
                        )
                        answer = result["answer"]
                        sources = result["sources"]

                        st.markdown(answer)

                        if sources:
                            with st.expander(f"📚 Source Citations & Context ({len(sources)} chunk{'s' if len(sources) > 1 else ''})"):
                                for s in sources:
                                    st.markdown(f"**Source:** `{s['source']}` &nbsp;|&nbsp; **Page:** `{s['page']}` &nbsp;|&nbsp; **Chunk ID:** `{s['chunk_number']}`")
                                    st.markdown(f'<div class="source-box">{s["content"]}</div>', unsafe_allow_html=True)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "mode": "simple_rag",
                            "sources": sources,
                        })

                except Exception as e:
                    err = f"Error processing query: {str(e)}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
