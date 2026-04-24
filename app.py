# app.py
# Author: Josavina - 10022300071
# CS4241 - Introduction to Artificial Intelligence - 2026
# Final Deliverable: Streamlit RAG Chat Application

import os
import sys
import streamlit as st

sys.path.insert(0, "src")

from pipeline import RAGPipeline
from memory_rag import ConversationMemory, build_memory_prompt
from evaluator import ADVERSARIAL_QUERIES, run_adversarial_tests

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ACity Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --bg: #071312;
        --panel: #0c1c1a;
        --panel-soft: #112725;
        --text: #e7f5f2;
        --muted: #9dc5be;
        --accent: #1ec8a5;
        --accent-2: #42e1bf;
        --border: rgba(66, 225, 191, 0.28);
    }

    .stApp {
        background:
            radial-gradient(1200px 500px at 5% -10%, rgba(30, 200, 165, 0.18), transparent 55%),
            radial-gradient(1000px 500px at 95% -20%, rgba(66, 225, 191, 0.14), transparent 50%),
            var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #081615 0%, #0c1c1a 100%);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }
    [data-testid="stSidebar"] .stButton button {
        border-radius: 10px;
        border: 1px solid var(--border);
        background: #12302b;
    }

    .hero {
        padding: 1.1rem 1.25rem;
        border: 1px solid var(--border);
        background: linear-gradient(130deg, rgba(26, 74, 66, 0.55), rgba(12, 28, 26, 0.78));
        border-radius: 14px;
        margin-bottom: 0.9rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 0.2px;
        background: linear-gradient(90deg, #9cf8e2, #1ec8a5, #42e1bf, #9cf8e2);
        background-size: 220% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: titleFlow 6s ease infinite;
    }
    .hero-sub {
        color: var(--muted);
        margin-top: 0.3rem;
        font-size: 0.95rem;
    }
    @keyframes titleFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .side-title { font-size: 1rem; font-weight: 700; margin: 0.2rem 0 0.5rem 0; color: var(--accent); }
    .side-footnote { color: var(--muted); font-size: 0.8rem; margin-top: 0.6rem; }

    .chunk-card {
        background: rgba(20, 44, 40, 0.62);
        border: 1px solid var(--border);
        padding: 0.55rem 0.75rem;
        margin: 0.38rem 0;
        border-radius: 10px;
        font-size: 0.82rem;
        color: var(--text);
    }
    .score-badge {
        background: rgba(30, 200, 165, 0.2);
        color: #bff9ec;
        border: 1px solid rgba(66, 225, 191, 0.45);
        border-radius: 999px;
        padding: 2px 8px;
        font-size: 0.73rem;
        font-weight: 700;
    }
    .source-tag {
        background: rgba(30, 200, 165, 0.15);
        color: #9cf8e2;
        border: 1px solid rgba(66, 225, 191, 0.35);
        border-radius: 6px;
        padding: 2px 6px;
        font-size: 0.73rem;
        margin-right: 4px;
    }
    .fade-in {
        animation: fadeInUp 0.35s ease;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .typing-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        color: var(--muted);
        font-size: 0.9rem;
    }
    .typing-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--accent-2);
        display: inline-block;
        animation: blink 1.2s infinite ease-in-out;
    }
    .typing-dot:nth-child(2) { animation-delay: 0.15s; }
    .typing-dot:nth-child(3) { animation-delay: 0.3s; }
    @keyframes blink {
        0%, 80%, 100% { opacity: 0.2; transform: scale(0.9); }
        40% { opacity: 1; transform: scale(1.05); }
    }

    /* Main content hierarchy */
    [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 1120px;
        padding-top: 1rem;
        border: 1px solid rgba(66, 225, 191, 0.16);
        border-radius: 16px;
        background: linear-gradient(180deg, rgba(12, 28, 26, 0.92), rgba(10, 22, 20, 0.92));
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.28);
    }

    /* Sidebar secondary emphasis */
    [data-testid="stSidebar"] {
        filter: saturate(0.9);
    }
    [data-testid="stSidebar"] .block-container {
        opacity: 0.92;
    }

    /* Inputs and controls polish */
    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 10px !important;
        border: 1px solid rgba(66, 225, 191, 0.24) !important;
        background: rgba(11, 26, 24, 0.9) !important;
        transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.12s ease;
    }
    .stTextInput input:focus,
    .stTextArea textarea:focus,
    .stSelectbox div[data-baseweb="select"] > div:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(30, 200, 165, 0.18) !important;
    }

    .stButton button {
        border-radius: 10px !important;
        border: 1px solid rgba(66, 225, 191, 0.3) !important;
        background: linear-gradient(180deg, rgba(20, 50, 45, 0.95), rgba(15, 40, 36, 0.95)) !important;
        color: #d5f7ef !important;
        transition: transform 0.12s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }
    .stButton button:hover {
        transform: translateY(-1px) !important;
        border-color: var(--accent) !important;
        box-shadow:
            0 10px 18px rgba(30, 200, 165, 0.3),
            0 0 0 1px rgba(30, 200, 165, 0.55) inset !important;
    }
    .stButton button:focus {
        border-color: var(--accent) !important;
        box-shadow:
            0 0 0 3px rgba(30, 200, 165, 0.26),
            0 8px 18px rgba(30, 200, 165, 0.26) !important;
    }

    /* Make sidebar button hover visible too */
    [data-testid="stSidebar"] .stButton button:hover {
        box-shadow:
            0 10px 18px rgba(30, 200, 165, 0.32),
            0 0 0 1px rgba(30, 200, 165, 0.6) inset !important;
    }

    /* Chat input send button glow */
    [data-testid="stChatInput"] button:hover {
        border-color: var(--accent) !important;
        box-shadow:
            0 8px 16px rgba(30, 200, 165, 0.32),
            0 0 0 1px rgba(30, 200, 165, 0.5) inset !important;
    }

    /* Typography and readability */
    h1, h2, h3 {
        letter-spacing: 0.2px;
        margin-bottom: 0.35rem !important;
    }
    p, li, label {
        line-height: 1.45;
    }

    /* Empty state helper */
    .empty-hint {
        margin: 0.3rem 0 0.85rem 0;
        padding: 0.65rem 0.8rem;
        border-radius: 10px;
        border: 1px dashed rgba(66, 225, 191, 0.34);
        background: rgba(17, 39, 36, 0.52);
        color: #a9d7ce;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
def init_session():
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory(max_history=5)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pipeline_ready" not in st.session_state:
        st.session_state.pipeline_ready = False
    if "eval_results" not in st.session_state:
        st.session_state.eval_results = None

init_session()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://acity.edu.gh/wp-content/uploads/2025/01/Academic-City-red-white-logo-448x134.png",
             width=150)
    st.markdown('<div class="side-title">⚙️ Configuration</div>', unsafe_allow_html=True)

    openrouter_key = st.text_input(
    "API Key (Groq or OpenRouter)",
    type="password",
    placeholder="gsk_... or sk-or-...",
    help="Paste a Groq key (gsk_) or OpenRouter key (sk-or-)"
)

    st.markdown("---")
    st.markdown('<div class="side-title">🔧 Retrieval Settings</div>', unsafe_allow_html=True)

    top_k = st.slider("Top-K chunks to retrieve", min_value=2, max_value=10, value=5)
    template = st.selectbox("Prompt Template", ["strict", "analytical", "basic"],
                            help="strict=hallucination control, analytical=step-by-step")
    use_expansion = st.toggle("Query Expansion", value=True,
                               help="Expands query with synonyms for better recall")
    use_memory = st.toggle("Conversation Memory (Part G)", value=True,
                            help="Enables multi-turn context awareness")

    st.markdown("---")
    if st.button("🔄 Clear Conversation"):
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.rerun()

    if st.button("🗑️ Rebuild Index", type="secondary"):
        if st.session_state.pipeline:
            with st.spinner("Rebuilding index..."):
                st.session_state.pipeline.setup(force_rebuild=True)
            st.success("Index rebuilt!")

    st.markdown("---")
    st.markdown('<div class="side-title">📚 Data Sources</div>', unsafe_allow_html=True)
    st.caption("🗳️ Ghana Election Results CSV")
    st.caption("💰 2025 Budget Statement PDF")
    st.markdown('<div class="side-footnote">CS4241 · Academic City University</div>', unsafe_allow_html=True)


# ── Main header / hero ────────────────────────────────────────────────────────
st.image(
    "https://acity.edu.gh/wp-content/uploads/2025/01/Academic-City-red-white-logo-448x134.png",
    width=190
)
st.markdown(
    """
    <div class="hero">
      <h1 class="hero-title">ACity Knowledge Assistant</h1>
      <div class="hero-sub">Ask grounded questions about Ghana's election results and 2025 budget with retrieval-backed answers.</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_retrieval, tab_eval, tab_arch = st.tabs(
    ["💬 Chat", "🔍 Retrieval Inspector", "⚗️ Evaluation", "🏗️ Architecture"]
)


# ── Setup pipeline ────────────────────────────────────────────────────────────
def get_pipeline(api_key: str) -> RAGPipeline:
    if not st.session_state.pipeline_ready:
        with st.spinner("🚀 Loading data and building index (first time may take 2-3 min)..."):
            pipeline = RAGPipeline(api_key=api_key, top_k=top_k, use_query_expansion=use_expansion)
            pipeline.setup()
            st.session_state.pipeline = pipeline
            st.session_state.pipeline_ready = True
        st.success("✅ Pipeline ready!")
    return st.session_state.pipeline

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1: CHAT
# ════════════════════════════════════════════════════════════════════════════════
with tab_chat:
    if not openrouter_key:
        st.info("👆 Enter your Groq or OpenRouter API key in the sidebar to get started.")
        st.stop()

    pipeline = get_pipeline(openrouter_key)

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(f'<div class="fade-in">{msg["content"]}</div>', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown(
            """
            <div class="empty-hint">
                Try asking: "Who won in Greater Accra?", "Summarize the 2025 budget priorities", or "Compare Ashanti and Volta results."
            </div>
            """,
            unsafe_allow_html=True
        )

    # Query input
    query = st.chat_input("Ask about Ghana's elections or 2025 budget...")

    if query:
        # Show user message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Memory: rewrite query if enabled
        effective_query = st.session_state.memory.rewrite_query(query) if use_memory else query
        if effective_query != query:
            st.caption(f"🧠 Query rewritten to: *{effective_query}*")

        # Run pipeline
        with st.chat_message("assistant"):
            typing_placeholder = st.empty()
            typing_placeholder.markdown(
                """
                <div class="typing-indicator fade-in">
                    <span>Thinking</span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
                """,
                unsafe_allow_html=True
            )
            with st.spinner("Thinking..."):
                result = pipeline.query(
                    effective_query,
                    template=template,
                    top_k=top_k,
                    use_expansion=use_expansion
                )
                response = result["response"]

            typing_placeholder.empty()
            st.markdown(f'<div class="fade-in">{response}</div>', unsafe_allow_html=True)
            chunks = result["retrieved_chunks"]

        # Update memory and history
        if use_memory:
            st.session_state.memory.add_turn(
                query=effective_query,
                response=response,
                sources=result["metadata"].get("sources_used", []),
                chunks=chunks
            )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2: RETRIEVAL INSPECTOR
# ════════════════════════════════════════════════════════════════════════════════
with tab_retrieval:
    st.markdown("### 🔍 Inspect Retrieval Quality")
    st.markdown("Test queries directly and see similarity scores, chunk sources, and query expansion effects.")

    if not openrouter_key:
        st.info("Enter API key in sidebar first.")
    else:
        pipeline = get_pipeline(openrouter_key)
        inspect_query = st.text_input("Test Query", placeholder="e.g. NDC votes in Ashanti")
        col1, col2 = st.columns(2)

        if inspect_query:
            with col1:
                st.markdown("**Standard Retrieval**")
                standard = pipeline.embedding_pipeline.search(inspect_query, top_k=5)
                for r in standard:
                    score = r["similarity_score"]
                    color = "green" if score > 0.7 else "orange" if score > 0.5 else "red"
                    st.markdown(f"**:{color}[{score:.4f}]** `{r['source']}` — {r['text'][:150]}...")

            with col2:
                st.markdown("**With Query Expansion**")
                expanded = pipeline.embedding_pipeline.search_with_query_expansion(inspect_query, top_k=5)
                for r in expanded:
                    score = r["similarity_score"]
                    color = "green" if score > 0.7 else "orange" if score > 0.5 else "red"
                    matched = r.get("matched_query", inspect_query)
                    st.markdown(f"**:{color}[{score:.4f}]** `{r['source']}` *(via: {matched[:40]})* — {r['text'][:120]}...")

            st.markdown("---")
            st.markdown("### 🧪 Full Pipeline Preview")
            if st.button("Run Full Pipeline for Inspector Query", key="run_inspector_full"):
                with st.spinner("Running full pipeline..."):
                    inspect_result = pipeline.query(
                        inspect_query,
                        template=template,
                        top_k=top_k,
                        use_expansion=use_expansion
                    )

                st.markdown("**Assistant Response**")
                st.info(inspect_result["response"])

                chunks = inspect_result["retrieved_chunks"]
                with st.expander(f"📄 Retrieved {len(chunks)} chunks", expanded=False):
                    for chunk in chunks:
                        score = chunk.get("similarity_score", 0)
                        source = chunk.get("source", "unknown")
                        badge_color = "🟢" if score > 0.7 else "🟡" if score > 0.5 else "🔴"
                        st.markdown(
                            f'<div class="chunk-card">'
                            f'{badge_color} <span class="score-badge">{score:.3f}</span> '
                            f'<span class="source-tag">{source}</span><br>'
                            f'{chunk["text"][:250]}...</div>',
                            unsafe_allow_html=True
                        )

                with st.expander("📝 Final Prompt Sent to LLM", expanded=False):
                    st.code(inspect_result["final_prompt"], language="markdown")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3: EVALUATION
# ════════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("### ⚗️ Adversarial Testing & Evaluation (Part E)")
    st.markdown("Compare RAG responses vs pure LLM on challenging queries.")

    if not openrouter_key:
        st.info("Enter API key in sidebar first.")
    else:
        pipeline = get_pipeline(openrouter_key)
        if st.button("▶️ Run All Tests", key="run_all_tests"):
            all_results = []
            with st.spinner("Running all adversarial tests..."):
                for test in ADVERSARIAL_QUERIES:
                    rag_result = pipeline.query(test["query"])
                    llm_result = pipeline.query_pure_llm(test["query"])
                    all_results.append({
                        "test": test,
                        "rag_result": rag_result,
                        "llm_result": llm_result
                    })
            st.session_state.eval_results = all_results

        if st.session_state.eval_results:
            for item in st.session_state.eval_results:
                test = item["test"]
                rag_result = item["rag_result"]
                llm_result = item["llm_result"]

                with st.expander(f"[{test['id']}] {test['type'].upper()}: *{test['query']}*"):
                    st.markdown(f"**Description:** {test['description']}")
                    st.markdown(f"**Expected behavior:** {test['expected_behavior']}")
                    st.markdown(f"**Expected failure mode:** _{test['failure_mode']}_")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**🔍 RAG Response**")
                        st.info(rag_result["response"])
                        st.caption(f"Sources: {rag_result['metadata'].get('sources_used', [])}")
                    with col_b:
                        st.markdown("**🤖 Pure LLM (no retrieval)**")
                        st.warning(llm_result)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4: ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════════
with tab_arch:
    st.markdown("### 🏗️ System Architecture (Part F)")

    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                    RAG PIPELINE FLOW                        │
    └─────────────────────────────────────────────────────────────┘

    DATA SOURCES                INGESTION                 VECTOR STORE
    ┌──────────────┐   clean    ┌──────────────┐  embed   ┌──────────────┐
    │ Election CSV │──────────▶│   Chunker    │─────────▶│    FAISS     │
    │ Budget PDF   │           │ (fixed/sent) │          │  IndexFlatIP │
    └──────────────┘           └──────────────┘          └──────┬───────┘
                                                                 │
    USER QUERY                 MEMORY                   RETRIEVAL│
    ┌──────────────┐  rewrite  ┌──────────────┐  top-k  ┌──────▼───────┐
    │  Streamlit   │─────────▶│  Conversation│────────▶│  Embedder    │
    │     UI       │          │   Memory     │         │ (MiniLM-L6)  │
    └──────────────┘          └──────────────┘         └──────┬───────┘
                                                              │
    RESPONSE                  GENERATION              PROMPT  │
    ┌──────────────┐          ┌──────────────┐  build  ┌──────▼───────┐
    │  Display +   │◀────────│ Groq API   │◀───────    │  Prompt     │
    │  Chunks +    │          │              │        │   Engine     │
    │  Scores      │          └──────────────┘        └──────────────┘
    └──────────────┘
    ```
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Component Justifications:**
        - **FAISS IndexFlatIP** — Exact cosine similarity search. Suitable for our dataset size (~5000 chunks). No approximation needed.
        - **all-MiniLM-L6-v2** — Runs locally, 384-dim embeddings, best balance of speed vs quality for free deployment.
        - **Fixed-window chunking (400w/80 overlap)** — Uniform chunks ensure consistent embedding quality. 20% overlap captures cross-boundary context.
        - **Query expansion** — Bridges vocabulary gap between user terms and document vocabulary (e.g., "winner" → "elected candidate").
        """)
    with col2:
        st.markdown("""
        **Design Suitability for Domain:**
        - Ghana election CSV has structured row data → row-level chunking is optimal.
        - Budget PDF has policy prose → sliding window preserves paragraph context.
        - Memory RAG handles follow-up questions natural in budget/election analysis.
        - Strict prompt template prevents hallucination on specific numerical claims (votes, GHS amounts).
        - OpenRouter's strong few-shot capabilities allow us to use a single prompt template effectively, reducing engineering overhead.
        """)