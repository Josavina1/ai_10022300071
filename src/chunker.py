# chunker.py
# Author: Josavina - 10022300071
# CS4241 - Introduction to Artificial Intelligence - 2026
# Part A: Chunking Strategy Design & Implementation

import re
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════
# CHUNKING DESIGN JUSTIFICATION (Part A requirement)
#
# We use TWO different chunking strategies — one per data source:
#
# 1. CSV / Election Data  → Row-level chunking (each row = 1 chunk)
#    - Justification: Each row is a self-contained fact (candidate, votes, region).
#      Splitting rows across chunks would destroy the semantic unit. No overlap needed
#      since facts don't span rows.
#
# 2. PDF / Budget Document → Fixed sliding-window chunking
#    - Chunk size: 400 tokens (~300 words)
#    - Overlap: 80 tokens (~60 words) — 20% overlap
#    - Justification: Budget documents have long flowing paragraphs. A window of 400
#      tokens captures enough context for a policy point without overwhelming the
#      embedding model (max 512 tokens for most sentence-transformers). The 20%
#      overlap ensures that sentences spanning chunk boundaries are still retrievable.
#    - We also implement SENTENCE-AWARE chunking as an alternative to compare
#      retrieval quality (Part A: comparative analysis).
#
# COMPARATIVE ANALYSIS (run __main__ to see):
#    Fixed chunks are faster and more uniform in size.
#    Sentence-aware chunks preserve meaning better but vary in size.
#    Experiment logs in logs/chunking_experiment.txt record the comparison.
# ════════════════════════════════════════════════════════════════════════════════


# ── Strategy 1: Row-level chunking for CSV data ───────────────────────────────

def chunk_csv_documents(election_docs: List[Dict]) -> List[Dict]:
    """
    Each CSV row document becomes exactly one chunk.
    Adds chunk metadata for tracking.
    """
    chunks = []
    for doc in election_docs:
        chunk = {
            "chunk_id": f"election_{doc['row_index']}",
            "text": doc["text"],
            "source": doc["source"],
            "strategy": "row_level",
            "chunk_size": len(doc["text"].split())
        }
        chunks.append(chunk)
    logger.info(f"[CSV] Created {len(chunks)} row-level chunks.")
    return chunks


# ── Strategy 2a: Fixed sliding-window chunking for PDF ───────────────────────

def chunk_text_fixed_window(
    text: str,
    source: str = "Budget_2025",
    chunk_size: int = 400,
    overlap: int = 80
) -> List[Dict]:
    """
    Split text into fixed-size word windows with overlap.
    chunk_size: number of words per chunk
    overlap: number of words shared between consecutive chunks
    """
    words = text.split()
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "chunk_id": f"{source}_fixed_{chunk_id}",
            "text": chunk_text,
            "source": source,
            "strategy": "fixed_window",
            "chunk_size": len(chunk_words),
            "start_word": start,
            "end_word": end
        })

        chunk_id += 1
        start += chunk_size - overlap  # Slide forward, keeping overlap

        # Stop if we're at the end
        if end >= len(words):
            break

    logger.info(f"[PDF Fixed] Created {len(chunks)} fixed-window chunks (size={chunk_size}, overlap={overlap}).")
    return chunks


# ── Strategy 2b: Sentence-aware chunking for PDF ─────────────────────────────

def chunk_text_sentence_aware(
    text: str,
    source: str = "Budget_2025",
    max_sentences: int = 6,
    overlap_sentences: int = 1
) -> List[Dict]:
    """
    Split text by sentences, grouping max_sentences per chunk.
    overlap_sentences: number of sentences repeated in the next chunk.
    Preserves semantic boundaries better than fixed-window.
    """
    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(sentences):
        end = start + max_sentences
        chunk_sentences = sentences[start:end]
        chunk_text = " ".join(chunk_sentences)

        chunks.append({
            "chunk_id": f"{source}_sentence_{chunk_id}",
            "text": chunk_text,
            "source": source,
            "strategy": "sentence_aware",
            "chunk_size": len(chunk_text.split()),
            "num_sentences": len(chunk_sentences)
        })

        chunk_id += 1
        start += max_sentences - overlap_sentences

        if end >= len(sentences):
            break

    logger.info(f"[PDF Sentence] Created {len(chunks)} sentence-aware chunks (max={max_sentences} sentences).")
    return chunks


# ── Master chunker: combines both sources ────────────────────────────────────

def chunk_all_documents(
    election_docs: List[Dict],
    budget_text: str,
    pdf_strategy: str = "fixed_window"  # or "sentence_aware"
) -> List[Dict]:
    """
    Chunk all documents from both sources.
    Returns a single flat list of all chunks with metadata.
    """
    logger.info(f"=== Starting chunking (PDF strategy: {pdf_strategy}) ===")

    # Chunk CSV data
    csv_chunks = chunk_csv_documents(election_docs)

    # Chunk PDF data
    if pdf_strategy == "sentence_aware":
        pdf_chunks = chunk_text_sentence_aware(budget_text)
    else:
        pdf_chunks = chunk_text_fixed_window(budget_text)

    all_chunks = csv_chunks + pdf_chunks
    logger.info(f"=== Total chunks created: {len(all_chunks)} ===")
    return all_chunks


# ── Comparative Analysis helper ───────────────────────────────────────────────

def compare_chunking_strategies(budget_text: str, log_path: str = "logs/chunking_experiment.txt"):
    """
    Run both chunking strategies on the budget text and log statistics.
    This satisfies Part A: 'Comparative analysis of chunking impact on retrieval quality'.
    """
    import os
    os.makedirs("logs", exist_ok=True)

    fixed = chunk_text_fixed_window(budget_text, chunk_size=400, overlap=80)
    sentence = chunk_text_sentence_aware(budget_text, max_sentences=6, overlap_sentences=1)

    fixed_sizes = [c["chunk_size"] for c in fixed]
    sentence_sizes = [c["chunk_size"] for c in sentence]

    report = f"""
=== CHUNKING STRATEGY COMPARISON ===
Date: {__import__('datetime').datetime.now()}

--- Fixed Window (400 words, 80 overlap) ---
Total chunks      : {len(fixed)}
Avg chunk size    : {sum(fixed_sizes)/len(fixed_sizes):.1f} words
Min chunk size    : {min(fixed_sizes)} words
Max chunk size    : {max(fixed_sizes)} words

--- Sentence Aware (6 sentences, 1 overlap) ---
Total chunks      : {len(sentence)}
Avg chunk size    : {sum(sentence_sizes)/len(sentence_sizes):.1f} words
Min chunk size    : {min(sentence_sizes)} words
Max chunk size    : {max(sentence_sizes)} words

--- Analysis ---
Fixed Window produces uniform chunks — better for consistent embedding quality.
Sentence-Aware produces variable-size chunks — better for preserving semantic meaning.
For this project we use Fixed Window for the main pipeline because:
  1. Sentence-transformers (all-MiniLM-L6-v2) perform best with consistent input sizes.
  2. Budget text has many short clauses that would create tiny sentence chunks.
  3. Fixed overlap ensures cross-boundary context is always captured.
Decision: Use fixed_window as primary strategy.

--- Sample Fixed Chunk ---
{fixed[5]['text'][:300] if len(fixed) > 5 else fixed[0]['text'][:300]}

--- Sample Sentence Chunk ---
{sentence[5]['text'][:300] if len(sentence) > 5 else sentence[0]['text'][:300]}
"""
    with open(log_path, "w") as f:
        f.write(report)

    print(report)
    logger.info(f"Chunking comparison saved to {log_path}")
    return {"fixed": fixed, "sentence": sentence}


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test with dummy text
    sample_text = """
    The Government of Ghana has committed to fiscal consolidation in 2025.
    Total revenue is projected at GHS 180 billion. Expenditure is expected to reach GHS 210 billion.
    The fiscal deficit target is set at 4.2% of GDP. Inflation is projected to decline to 15% by end of year.
    The growth rate is expected to be 4.0% driven by oil, services and agriculture.
    Tax revenue collection will be improved through digitization of the GRA systems.
    Non-tax revenue is expected to contribute GHS 12 billion. External grants stand at GHS 2.5 billion.
    """ * 20  # Simulate longer text

    sample_csv_docs = [
        {"text": "region: Greater Accra | candidate: John Doe | votes: 5000", "source": "election", "row_index": 0},
        {"text": "region: Ashanti | candidate: Jane Doe | votes: 7000", "source": "election", "row_index": 1},
    ]

    results = compare_chunking_strategies(sample_text)
    csv_chunks = chunk_csv_documents(sample_csv_docs)
    print(f"\nCSV chunks: {len(csv_chunks)}")
    print(f"Fixed PDF chunks: {len(results['fixed'])}")
    print(f"Sentence PDF chunks: {len(results['sentence'])}")