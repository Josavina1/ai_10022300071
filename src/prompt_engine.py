# prompt_engine.py
# Author: Josavina - 10022300071
# CS4241 - Introduction to Artificial Intelligence - 2026
# Part C: Prompt Engineering & Generation

import logging
import os
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# PROMPT DESIGN (Part C requirement)
#
# We define THREE prompt templates to experiment with:
#
# Template 1 - BASIC: Simple context injection, no guardrails
#   Used as baseline to show hallucination risk
#
# Template 2 - STRICT (primary): Strong hallucination control
#   - Explicitly instructs LLM to only use provided context
#   - Instructs LLM to say "I don't know" if context is insufficient
#   - Separates context from question clearly with XML-style tags
#
# Template 3 - ANALYTICAL: For comparison/analysis questions
#   - Adds instruction for structured reasoning
#   - Useful for budget vs election cross-queries
#
# Experiment logs show Template 2 significantly reduces hallucination.
# ════════════════════════════════════════════════════════════════════════════════

MAX_CONTEXT_WORDS = 1200  # Hard cap on context window for LLM prompt


# ── Template definitions ──────────────────────────────────────────────────────

TEMPLATE_BASIC = """You are an AI assistant. Use the context below to answer the question.

Context:
{context}

Question: {question}

Answer:"""


TEMPLATE_STRICT = """You are an AI assistant for Academic City University, specializing in Ghana election results and the 2025 Ghana Budget Statement.

IMPORTANT RULES:
1. Answer ONLY using the information in the <context> section below.
2. If the context does not contain enough information to answer confidently, say exactly: "I don't have enough information in the provided documents to answer this question."
3. Do NOT make up facts, statistics, or names.
4. If the question references an entity not found in the context, say so clearly.
5. Keep answers concise and factual.

<context>
{context}
</context>

<question>
{question}
</question>

Answer based strictly on the context above:"""


TEMPLATE_ANALYTICAL = """You are an expert analyst with access to Ghana's 2025 election results and the 2025 Budget Statement from the Ministry of Finance.

Your task is to answer the question below using ONLY the provided context documents.
Think step by step:
  1. Identify which documents are relevant to the question.
  2. Extract the key facts.
  3. Synthesize a clear, evidence-based answer.

If the context is insufficient, clearly state what information is missing.

Context Documents:
{context}

Question: {question}

Step-by-step Answer:"""


TEMPLATES = {
    "basic": TEMPLATE_BASIC,
    "strict": TEMPLATE_STRICT,
    "analytical": TEMPLATE_ANALYTICAL,
}


# ── Context window management ─────────────────────────────────────────────────

def truncate_chunks(chunks: List[Dict], max_words: int = MAX_CONTEXT_WORDS) -> List[Dict]:
    """
    Context window management (Part C requirement):
    Rank chunks by similarity score and truncate to fit within max_words.
    Returns only the chunks that fit within the word budget.
    """
    # Sort by similarity score descending (best chunks first)
    sorted_chunks = sorted(chunks, key=lambda x: x.get("similarity_score", 0), reverse=True)

    selected = []
    total_words = 0

    for chunk in sorted_chunks:
        chunk_words = len(chunk["text"].split())
        if total_words + chunk_words <= max_words:
            selected.append(chunk)
            total_words += chunk_words
        else:
            # Partially include the chunk if there's remaining budget
            remaining = max_words - total_words
            if remaining > 50:  # Only include if meaningful amount remains
                truncated = chunk.copy()
                truncated["text"] = " ".join(chunk["text"].split()[:remaining]) + "..."
                truncated["truncated"] = True
                selected.append(truncated)
            break

    logger.info(f"Context window: {len(selected)}/{len(chunks)} chunks selected ({total_words} words)")
    return selected


def format_context(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into a readable context string.
    Labels each chunk with its source for traceability.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        score = chunk.get("similarity_score", 0)
        text = chunk.get("text", "")
        parts.append(f"[Document {i} | Source: {source} | Relevance: {score:.3f}]\n{text}")
    return "\n\n".join(parts)


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(
    query: str,
    chunks: List[Dict],
    template_name: str = "strict"
) -> Dict:
    """
    Build the final prompt from a query and retrieved chunks.
    Returns a dict with the prompt string and metadata for logging.
    """
    if template_name not in TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}. Choose from: {list(TEMPLATES.keys())}")

    # Step 1: Truncate to fit context window
    selected_chunks = truncate_chunks(chunks, MAX_CONTEXT_WORDS)

    # Step 2: Format context
    context_str = format_context(selected_chunks)

    # Step 3: Build prompt
    template = TEMPLATES[template_name]
    prompt = template.format(context=context_str, question=query)

    metadata = {
        "template": template_name,
        "num_chunks_retrieved": len(chunks),
        "num_chunks_used": len(selected_chunks),
        "context_word_count": len(context_str.split()),
        "prompt_word_count": len(prompt.split()),
        "sources_used": list({c.get("source") for c in selected_chunks}),
    }

    logger.info(f"Prompt built | template={template_name} | "
                f"chunks={metadata['num_chunks_used']} | words={metadata['prompt_word_count']}")

    return {
        "prompt": prompt,
        "context": context_str,
        "selected_chunks": selected_chunks,
        "metadata": metadata
    }


# ── Prompt experiment logger (Part C: evidence of improvement) ────────────────

def run_prompt_experiment(
    query: str,
    chunks: List[Dict],
    log_path: str = "logs/prompt_experiments.txt"
) -> None:
    """
    Run the same query through all three templates and log the prompts.
    Use this to manually compare outputs in experiment logs.
    """
    os.makedirs("logs", exist_ok=True)

    with open(log_path, "a") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"QUERY: {query}\n")
        f.write(f"{'='*70}\n")

        for name in TEMPLATES:
            result = build_prompt(query, chunks, template_name=name)
            f.write(f"\n--- Template: {name.upper()} ---\n")
            f.write(f"Chunks used: {result['metadata']['num_chunks_used']}\n")
            f.write(f"Context words: {result['metadata']['context_word_count']}\n")
            f.write(f"Prompt preview:\n{result['prompt'][:600]}...\n")

    logger.info(f"Prompt experiment logged to {log_path}")


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dummy_chunks = [
        {
            "chunk_id": "budget_1",
            "text": "Ghana's 2025 budget targets GDP growth of 4.0 percent driven by oil and services sectors.",
            "source": "Budget_2025",
            "similarity_score": 0.91
        },
        {
            "chunk_id": "election_5",
            "text": "Greater Accra parliamentary results: NDC won 15 seats, NPP won 12 seats.",
            "source": "Ghana_Election_Results",
            "similarity_score": 0.72
        }
    ]

    result = build_prompt("What is Ghana's GDP growth target?", dummy_chunks, template_name="strict")
    print("\n=== Built Prompt ===")
    print(result["prompt"])
    print("\n=== Metadata ===")
    print(result["metadata"])

    run_prompt_experiment("What is the budget deficit for 2025?", dummy_chunks)
    print("Experiment logged.")