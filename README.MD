# AI Knowledge Assistant — Academic City University
**Student Name:** Josavina  Abbey
**Index Number:** 10022300071
**Course:** CS4241 — Introduction to Artificial Intelligence  
**Lecturer:** Godwin N. Danso  
**Year:** 2026  

---


📌 Project Overview
A fully custom Retrieval-Augmented Generation (RAG) chatbot built for Academic City University. It allows users to query Ghana's 2024 Election Results and the 2025 Budget Statement using natural language.
No LangChain. No LlamaIndex. All RAG components built from scratch.

🏗️ Architecture
User Query → Memory Rewriter → Embedder (MiniLM-L6) → FAISS Retrieval
           → Query Expansion → Prompt Engine → OpenRouter LLM (mistralai/mistral-7b-instruct:free) → Response

📦 Tech Stack
ComponentTechnologyEmbeddingssentence-transformers/all-MiniLM-L6-v2 (local)Vector StoreFAISS IndexFlatIPLLMOpenRouter API (mistralai/mistral-7b-instruct:free) — Free tierUIStreamlitPDF parsingpdfplumberDatapandas

🚀 Quick Start
1. Clone the repo
bashgit clone https://github.com/[your-username]/ai_[index-number]
cd ai_[10022300071]
2. Install dependencies
bashpip install -r requirements.txt
3. Get a free OpenRouter or Groq API key
4. Run the app
bashstreamlit run app.py
5. Enter your OpenRouter or Groq API key in the sidebar

📁 Project Structure
ai_[index]/
├── src/
│   ├── data_loader.py      # Part A: Data engineering & cleaning
│   ├── chunker.py          # Part A: Chunking strategies
│   ├── embedder.py         # Part B: Embedding pipeline + FAISS
│   ├── prompt_engine.py    # Part C: Prompt templates
│   ├── pipeline.py         # Part D: Full RAG pipeline
│   ├── evaluator.py        # Part E: Adversarial testing
│   └── memory_rag.py       # Part G: Innovation — Memory RAG
├── logs/                   # Experiment logs (auto-generated)
├── vectorstore/            # FAISS index (auto-generated)
├── data/                   # Downloaded datasets (auto-generated)
├── app.py                  # Streamlit UI
├── requirements.txt
└── README.md

📋 Parts Addressed
PartDescriptionMarksAData Engineering & Chunking4BCustom Retrieval + Query Expansion6CPrompt Engineering (3 templates)4DFull Pipeline with logging10EAdversarial Testing (4 queries)6FArchitecture diagram + justification8GMemory-Based RAG (innovation)6FinalUI + video + logs + docs16

🧪 Experiment Logs
See logs/ directory:

chunking_experiment.txt — Fixed vs sentence-aware chunking comparison
prompt_experiments.txt — Same query across 3 prompt templates
adversarial_tests.txt — RAG vs pure LLM on adversarial queries
pipeline_runs.jsonl — Every query run with full stage logging
memory_rag_sessions.jsonl — Multi-turn conversation sessions


⚠️ Important Constraints Met

✅ No LangChain
✅ No LlamaIndex
✅ No pre-built RAG pipelines
✅ Manual chunking implementation
✅ Manual embedding pipeline
✅ Manual vector storage (FAISS)
✅ Manual prompt construction


🌐 Deployed URL

