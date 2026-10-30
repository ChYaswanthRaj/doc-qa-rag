# 🤖 Agentic Retrieval-Augmented Generation (RAG) Document Q&A with Google Gemini

A production-ready, modular Retrieval-Augmented Generation (RAG) Document Q&A chatbot built with **Python**, **Streamlit**, **LangChain**, **LangGraph**, **Google Gemini**, and **FAISS**.

Supports uploading **PDF** and **TXT** files, semantic chunking, FAISS vector indexing with Google embeddings (`models/text-embedding-004`), strict document-grounded question answering with source citations (filename + page numbers), and an **Agentic RAG mode** powered by LangGraph and Gemini (`gemini-1.5-flash`) with tool-calling, autonomous query rewriting (up to 3 retries), and safe calculation.

---

## 🏗️ Architecture Diagram

```
                             +-------------------------------+
                             |    User (Streamlit Web UI)    |
                             +---------------+---------------+
                                             |
                   +-------------------------+-------------------------+
                   |                                                   |
      [1] Document Ingestion & Indexing                   [2] Question Answering Query
                   |                                                   |
        +----------v-----------+                          +------------v------------+
        |   Upload PDF / TXT   |                          |    Mode Toggle in UI    |
        +----------+-----------+                          +------+-----------+------+
                   |                                             |           |
        +----------v-----------+                       [Simple RAG]      [Agent Mode]
        |    rag/loader.py     |                                 |           |
        | - Page tracking      |                                 |           |
        | - Text splitting     |                                 |           |
        +----------+-----------+                                 |           |
                   | (Chunks)                                    |           |
        +----------v-----------+                                 |           |
        |     rag/index.py     |                                 |           |
        | - Google Embeddings  |                                 |           |
        | (text-embedding-004) |                                 |           |
        | - FAISS Vector Store |                                 |           |
        +----------+-----------+                                 |           |
                   |                                             |           |
                   +---------------------+                       |           |
                                         |                       |           |
                                         v                       v           v
                                 +---------------+       +-----------+  +-----------+
                                 |  FAISS Index  |<------+ rag/qa.py |  |rag/agent.py|
                                 +---------------+       +-----+-----+  +-----+-----+
                                                               |              |
                                                    Strict Context LLM   LangGraph ReAct
                                                    (gemini-1.5-flash)   Tools & Loop
                                                               |              |
                                                               +-------+------+
                                                                       |
                                                                       v
                                                        +---------------------------+
                                                        |  Answer + Source Cards    |
                                                        |  + Expandable Agent Trace |
                                                        +---------------------------+
```

---

## 🔄 The Agent Loop (LangGraph ReAct)

When **Agent Mode** is enabled, the pipeline transitions from a static retrieve-and-generate pattern to an active reasoning loop implemented in [`rag/agent.py`](file:///Users/yaswanthch/Downloads/doc-qa-rag/rag/agent.py) using LangGraph:

```
                  +----------------------------------+
                  |           User Prompt            |
                  +-----------------+----------------+
                                    |
                                    v
                       +------------------------+
              +------->|   LLM Reasoning Node   |<--------+
              |        |   (gemini-1.5-flash)   |         |
              |        +------------+-----------+         |
              |                     |                     |
     [Insufficient context]         | [Decides Tool]      | [Tool Results]
     [Rewrite query (max 3)]        v                     |
              |            +-----------------+            |
              +------------+  Tool Execution +------------+
                           +--------+--------+
                                    |
             +----------------------+----------------------+
             |                      |                      |
             v                      v                      v
    `search_documents`      `summarize_document`      `calculator`
    (FAISS top-k chunks     (Full document summary)   (Safe math eval)
    + source & page)
             |
             +-----> [Or `list_documents`]
```

### Agent Behaviors:
1. **Tool Calling**: Autonomously calls `search_documents`, `list_documents`, `summarize_document`, and `calculator`.
2. **Autonomous Query Rewriting & Retries**: If the retrieved chunks from `search_documents` do not provide sufficient clarity or context, the agent analyzes what is missing, rewrites the search query using synonyms or alternative keywords, and searches again (up to **3 retry attempts**).
3. **Calculation Delegation**: For financial or numerical metrics (e.g. percentage growth, ratios), the agent calls the `calculator` tool to verify calculations without mental arithmetic hallucinations.
4. **Strict Grounding & Refusal Rule**: If the answer cannot be found in the documents after retries, the agent halts and strictly refuses to answer with:
   > *"I am sorry, but the provided documents do not contain information to answer this question."*
5. **Citations**: All claims cite source filename and page numbers (e.g. `[annual_report.pdf, Page 3]`).

---

## 📂 Project Directory Structure

```
doc-qa-rag/
├── app.py                     # Streamlit frontend with Simple RAG vs Agent Mode & Traces
├── requirements.txt           # Python dependencies (includes langchain-google-genai)
├── .env.example               # Template for GOOGLE_API_KEY
├── .gitignore                 # Git ignore rules
├── README.md                  # Documentation and architecture guide
├── rag/
│   ├── __init__.py
│   ├── loader.py              # Ingests PDF/TXT with page tracking & recursive chunking
│   ├── index.py               # Google embeddings (text-embedding-004) & FAISS vectorstore management
│   ├── rag/qa.py              # Simple RAG pipeline with strict Gemini grounding & refusal
│   └── agent.py               # Agentic RAG using LangGraph ReAct agent & tools
├── eval/
│   ├── __init__.py
│   └── evaluate.py            # Benchmark script comparing Simple RAG vs Agent Mode
└── sample_docs/
    ├── acme_corp_financials_2024.txt   # Sample corporate quarterly financial data
    └── employee_travel_policy.txt      # Sample corporate business travel policy
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Google Gemini API Key

### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your Google API Key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 💻 Running the Streamlit Web Application

Launch the application:
```bash
streamlit run app.py
```

### Using the App:
1. Open your browser at `http://localhost:8501`.
2. In the sidebar:
   - Configure your **Google Gemini API Key** (loaded automatically from `.env`).
   - Upload PDF and/or TXT documents (or use the sample files in `sample_docs/`).
   - Select Gemini Model: `gemini-1.5-flash` (default), `gemini-1.5-pro`, or `gemini-2.0-flash`.
   - Adjust **Chunk Size** (e.g. 1000) and **Chunk Overlap** (e.g. 200).
   - Click **🚀 Process & Index Documents**.
3. Toggle **Agent mode** ON (for tool-calling, multi-hop reasoning, and retry) or OFF (for direct single-pass RAG).
4. Ask questions in the chat box!
5. Inspect the expandable **🕵️ Agent Trace** and **📚 Source Chunks & Citations** sections under each answer.

---

## 📊 Evaluation & Benchmarking

Run the automated evaluation benchmark across the test dataset:

```bash
python eval/evaluate.py
```

### Benchmark Metrics Tracked:
- **Answerable Accuracy**: Percentage of in-domain questions answered correctly.
- **Refusal Accuracy**: Percentage of out-of-domain questions correctly refused without hallucination.
- **Overall Accuracy**: Total accuracy across all benchmark questions.
- **Average Tool Calls per Question**: Measures agent activity and tool usage.
- **Average Latency**: Execution time per question.

---

## 🛡️ Strict Grounding & Security Measures
- **No Hardcoded Keys**: API keys are loaded via `python-dotenv` from `.env`.
- **Context Boundaries**: Zero-temperature Gemini prompts with hard refusal guarantees prevent extrapolation beyond retrieved context.
- **Safe Calculator**: The `calculator` tool parses Abstract Syntax Trees (`ast`) without using Python's unsafe `eval()`.
