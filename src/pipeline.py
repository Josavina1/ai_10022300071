# pipeline.py
# Author: Josavina - 10022300071
# CS4241 - Introduction to Artificial Intelligence - 2026
# Part D: Full RAG Pipeline Implementation

import os
import json
import logging
import datetime
import requests
from typing import List, Dict, Optional
 
from data_loader import load_all_data
from chunker import chunk_all_documents
from embedder import EmbeddingPipeline
from prompt_engine import build_prompt
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
 
# ════════════════════════════════════════════════════════════════════════════════
# PIPELINE FLOW:
# User Query → [STAGE 1] Retrieval → [STAGE 2] Context Selection →
# [STAGE 3] Prompt Construction → [STAGE 4] LLM Generation → Response
# ════════════════════════════════════════════════════════════════════════════════
 
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct:free"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"
LOG_FILE = "logs/pipeline_runs.jsonl"
os.makedirs("logs", exist_ok=True)
 
 
class RAGPipeline:
    def __init__(self, api_key: str, top_k: int = 5, use_query_expansion: bool = True):
        self.top_k = top_k
        self.use_query_expansion = use_query_expansion
        self.embedding_pipeline = EmbeddingPipeline()
        self.api_key = api_key
        self.provider = self._detect_provider(api_key)
        self._ready = False

    def _detect_provider(self, api_key: str) -> str:
        key = (api_key or "").strip()
        if key.startswith("gsk_"):
            return "groq"
        if key.startswith("sk-or-"):
            return "openrouter"
        # Default to Groq so users with Groq keys can paste and run quickly.
        return "groq"
 
    # ── Setup ─────────────────────────────────────────────────────────────────
 
    def setup(self, force_rebuild: bool = False) -> None:
        if not force_rebuild and self.embedding_pipeline.load():
            logger.info("Loaded existing FAISS index from disk.")
            self._ready = True
            return
 
        logger.info("=== Building RAG index from scratch ===")
        data = load_all_data()
        all_chunks = chunk_all_documents(
            election_docs=data["election_docs"],
            budget_text=data["budget_text"],
            pdf_strategy="fixed_window"
        )
        self.embedding_pipeline.build_index(all_chunks)
        self.embedding_pipeline.save()
        self._ready = True
        logger.info("=== RAG pipeline ready ===")
 
    # ── Call OpenRouter LLM ───────────────────────────────────────────────────
 
    def _call_llm(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        endpoint = GROQ_URL
        model = GROQ_MODEL

        if self.provider == "openrouter":
            endpoint = OPENROUTER_URL
            model = OPENROUTER_MODEL
            headers["HTTP-Referer"] = "https://acity.edu.gh"
            headers["X-Title"] = "ACity Knowledge Assistant"

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 300
        }
        response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        if not response.ok:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        return response.json()["choices"][0]["message"]["content"]
    
       
 
    # ── Main query ────────────────────────────────────────────────────────────
 
    def query(
        self,
        user_query: str,
        template: str = "strict",
        top_k: Optional[int] = None,
        use_expansion: Optional[bool] = None
    ) -> Dict:
        if not self._ready:
            raise RuntimeError("Pipeline not ready. Call setup() first.")
 
        k = top_k or self.top_k
        expand = use_expansion if use_expansion is not None else self.use_query_expansion
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
 
        log_entry = {
            "run_id": run_id,
            "query": user_query,
            "template": template,
            "top_k": k,
            "use_expansion": expand,
            "timestamp": datetime.datetime.now().isoformat(),
            "stages": {}
        }
 
        # ── STAGE 1: Retrieval ────────────────────────────────────────────────
        logger.info(f"[STAGE 1] Retrieving top-{k} chunks for: '{user_query}'")
        if expand:
            retrieved_chunks = self.embedding_pipeline.search_with_query_expansion(user_query, top_k=k)
        else:
            retrieved_chunks = self.embedding_pipeline.search(user_query, top_k=k)
 
        log_entry["stages"]["retrieval"] = {
            "method": "query_expansion" if expand else "standard",
            "num_retrieved": len(retrieved_chunks),
            "chunks": [
                {
                    "chunk_id": c["chunk_id"],
                    "source": c["source"],
                    "similarity_score": c["similarity_score"],
                    "text_preview": c["text"][:150]
                }
                for c in retrieved_chunks
            ]
        }
 
        # ── STAGE 2: Context selection ────────────────────────────────────────
        logger.info(f"[STAGE 2] Building prompt with template='{template}'")
        prompt_result = build_prompt(user_query, retrieved_chunks, template_name=template)
 
        log_entry["stages"]["context_selection"] = {
            "num_chunks_used": prompt_result["metadata"]["num_chunks_used"],
            "context_word_count": prompt_result["metadata"]["context_word_count"],
            "sources_used": prompt_result["metadata"]["sources_used"],
        }
 
        # ── STAGE 3: Prompt construction ──────────────────────────────────────
        final_prompt = prompt_result["prompt"]
        log_entry["stages"]["prompt_construction"] = {
            "prompt_word_count": prompt_result["metadata"]["prompt_word_count"],
            "final_prompt": final_prompt
        }
 
        # ── STAGE 4: LLM Generation ───────────────────────────────────────────
        model_name = GROQ_MODEL if self.provider == "groq" else OPENROUTER_MODEL
        logger.info(f"[STAGE 4] Sending prompt to {self.provider.upper()} ({model_name})...")
        try:
            response_text = self._call_llm(final_prompt)
            log_entry["stages"]["generation"] = {
                "provider": self.provider,
                "model": model_name,
                "response": response_text,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            response_text = f"Error generating response: {str(e)}"
            log_entry["stages"]["generation"] = {"status": "error", "error": str(e)}
 
        self._log_run(log_entry)
 
        return {
            "run_id": run_id,
            "query": user_query,
            "response": response_text,
            "retrieved_chunks": retrieved_chunks,
            "context": prompt_result["context"],
            "final_prompt": final_prompt,
            "metadata": prompt_result["metadata"],
        }
 
    # ── Pure LLM (no retrieval) for Part E comparison ────────────────────────
 
    def query_pure_llm(self, user_query: str) -> str:
        logger.info(f"[PURE LLM] Querying without retrieval: '{user_query}'")
        try:
            prompt = f"You are a helpful assistant. Answer this question about Ghana's elections and budget:\n\n{user_query}"
            return self._call_llm(prompt)
        except Exception as e:
            return f"Error: {str(e)}"
 
    # ── Logging ───────────────────────────────────────────────────────────────
 
    def _log_run(self, log_entry: dict) -> None:
        os.makedirs("logs", exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        logger.info(f"Run logged: {log_entry['run_id']}")
 
    # ── Evaluation ────────────────────────────────────────────────────────────
 
    def evaluate_query(self, query: str) -> Dict:
        rag_result = self.query(query)
        llm_result = self.query_pure_llm(query)
        comparison = {
            "query": query,
            "rag_response": rag_result["response"],
            "pure_llm_response": llm_result,
            "rag_sources": rag_result["metadata"].get("sources_used", []),
            "rag_chunks_used": rag_result["metadata"].get("num_chunks_used", 0),
        }
        with open("logs/rag_vs_llm_comparison.txt", "a") as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"QUERY: {query}\n\n")
            f.write(f"RAG RESPONSE:\n{comparison['rag_response']}\n\n")
            f.write(f"PURE LLM RESPONSE:\n{comparison['pure_llm_response']}\n")
            f.write(f"Sources used by RAG: {comparison['rag_sources']}\n")
        return comparison