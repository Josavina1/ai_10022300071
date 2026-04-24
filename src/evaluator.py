# evaluator.py
# Author: Josavina - 10022300071
# CS4241 - Introduction to Artificial Intelligence - 2026
# Part E: Critical Evaluation & Adversarial Testing

import os
import json
import logging
import datetime
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# ADVERSARIAL QUERIES (Part E requirement)
#
# Query 1 — AMBIGUOUS:
#   "Who won?" — No region, no election type specified.
#   Tests: Does RAG ask for clarification or pick arbitrary context?
#   Expected failure: Pure LLM may hallucinate a winner.
#
# Query 2 — MISLEADING/INCOMPLETE:
#   "What is Ghana's 2025 budget surplus?" — The budget has a DEFICIT, not surplus.
#   Tests: Does the system correct the false premise, or confirm the error?
#   Expected failure: Pure LLM may fabricate a surplus figure.
# ════════════════════════════════════════════════════════════════════════════════

ADVERSARIAL_QUERIES = [
    {
        "id": "ADV_001",
        "type": "ambiguous",
        "query": "Who won?",
        "description": "Ambiguous — no region or election type specified",
        "expected_behavior": "Should ask for clarification or note ambiguity",
        "failure_mode": "May hallucinate a specific winner"
    },
    {
        "id": "ADV_002",
        "type": "misleading",
        "query": "What is Ghana's 2025 budget surplus amount?",
        "description": "Misleading — the budget has a deficit, not surplus",
        "expected_behavior": "Should correct false premise using context",
        "failure_mode": "May confirm a non-existent surplus figure"
    },
    {
        "id": "ADV_003",
        "type": "out_of_scope",
        "query": "What is the population of Mars?",
        "description": "Completely out of scope for the knowledge base",
        "expected_behavior": "Should say information not in documents",
        "failure_mode": "May hallucinate a scientific answer unrelated to context"
    },
    {
        "id": "ADV_004",
        "type": "complex_cross_domain",
        "query": "How did the 2024 election results influence the 2025 budget allocations?",
        "description": "Requires reasoning across both data sources",
        "expected_behavior": "Should retrieve from both sources and synthesize",
        "failure_mode": "May retrieve only one source and give incomplete answer"
    }
]


# ── Evaluation metrics ────────────────────────────────────────────────────────

def score_response(response: str, expected_behavior: str) -> Dict:
    """
    Manual scoring rubric for evaluating responses.
    Returns scores and flags for the experiment log.
    This is a manual template — fill in actual scores after running experiments.
    """
    return {
        "response_length": len(response.split()),
        "contains_disclaimer": any(phrase in response.lower() for phrase in [
            "i don't", "i do not", "not enough information",
            "cannot find", "not in the", "no information"
        ]),
        "contains_numbers": any(c.isdigit() for c in response),
        "flags_ambiguity": any(word in response.lower() for word in [
            "unclear", "ambiguous", "which", "please specify", "clarify"
        ]),
        "expected_behavior_hint": expected_behavior,
        # Manual score fields — fill in after reading actual output:
        "manual_accuracy_score": None,      # 0-5
        "manual_hallucination_flag": None,  # True/False
        "manual_notes": ""
    }


# ── Adversarial test runner ───────────────────────────────────────────────────

def run_adversarial_tests(pipeline, log_path: str = "logs/adversarial_tests.txt") -> List[Dict]:
    """
    Run all adversarial queries through both RAG and pure LLM.
    Log detailed results for Part E submission.
    """
    os.makedirs("logs", exist_ok=True)
    results = []

    with open(log_path, "w") as f:
        f.write(f"ADVERSARIAL TEST LOG\n")
        f.write(f"Date: {datetime.datetime.now()}\n")
        f.write(f"Model: Google Gemma 2 9b\n")
        f.write("=" * 70 + "\n")

    for test in ADVERSARIAL_QUERIES:
        logger.info(f"Running adversarial test: {test['id']} - {test['type']}")

        # RAG response
        rag_result = pipeline.query(test["query"])
        rag_response = rag_result["response"]

        # Pure LLM response (no retrieval)
        llm_response = pipeline.query_pure_llm(test["query"])

        # Score both
        rag_score = score_response(rag_response, test["expected_behavior"])
        llm_score = score_response(llm_response, test["expected_behavior"])

        result = {
            "test_id": test["id"],
            "type": test["type"],
            "query": test["query"],
            "description": test["description"],
            "rag_response": rag_response,
            "llm_response": llm_response,
            "rag_score": rag_score,
            "llm_score": llm_score,
            "rag_chunks_used": rag_result["metadata"].get("num_chunks_used", 0),
            "rag_sources": rag_result["metadata"].get("sources_used", []),
        }
        results.append(result)

        # Write to log file
        with open(log_path, "a") as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"TEST ID   : {test['id']}\n")
            f.write(f"TYPE      : {test['type']}\n")
            f.write(f"QUERY     : {test['query']}\n")
            f.write(f"DESCRIPTION: {test['description']}\n")
            f.write(f"EXPECTED  : {test['expected_behavior']}\n")
            f.write(f"FAILURE   : {test['failure_mode']}\n")
            f.write(f"\n--- RAG RESPONSE (chunks used: {result['rag_chunks_used']}, sources: {result['rag_sources']}) ---\n")
            f.write(rag_response + "\n")
            f.write(f"[Auto-scored] Disclaimer present: {rag_score['contains_disclaimer']} | "
                    f"Flags ambiguity: {rag_score['flags_ambiguity']}\n")
            f.write(f"\n--- PURE LLM RESPONSE (no retrieval) ---\n")
            f.write(llm_response + "\n")
            f.write(f"[Auto-scored] Disclaimer present: {llm_score['contains_disclaimer']} | "
                    f"Flags ambiguity: {llm_score['flags_ambiguity']}\n")
            f.write(f"\n[MANUAL SCORING NEEDED - fill in after review]\n")
            f.write(f"RAG Accuracy (0-5): ___  Hallucination: ___  Notes: ___\n")
            f.write(f"LLM Accuracy (0-5): ___  Hallucination: ___  Notes: ___\n")

    # Summary
    with open(log_path, "a") as f:
        f.write(f"\n{'='*70}\n")
        f.write("SUMMARY TABLE (fill manually after reviewing outputs)\n")
        f.write(f"{'Test ID':<12} {'Type':<20} {'RAG Disclaimer':<16} {'LLM Disclaimer':<16}\n")
        for r in results:
            f.write(
                f"{r['test_id']:<12} {r['type']:<20} "
                f"{str(r['rag_score']['contains_disclaimer']):<16} "
                f"{str(r['llm_score']['contains_disclaimer']):<16}\n"
            )

    logger.info(f"Adversarial tests complete. Log saved to {log_path}")
    return results


# ── Consistency test: same query multiple times ───────────────────────────────

def test_response_consistency(pipeline, query: str, runs: int = 3,
                               log_path: str = "logs/consistency_test.txt") -> None:
    """
    Run the same query multiple times and check if responses are consistent.
    Part E: 'Response consistency' evaluation.
    """
    os.makedirs("logs", exist_ok=True)
    responses = []

    for i in range(runs):
        result = pipeline.query(query)
        responses.append(result["response"])

    with open(log_path, "a") as f:
        f.write(f"\nCONSISTENCY TEST: {query}\n{'='*50}\n")
        for i, resp in enumerate(responses, 1):
            f.write(f"Run {i}:\n{resp}\n\n")
        f.write(f"[Manual note: Compare the {runs} responses above for consistency]\n")

    logger.info(f"Consistency test logged for: '{query}'")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Evaluator module loaded.")
    print(f"Adversarial queries defined: {len(ADVERSARIAL_QUERIES)}")
    for q in ADVERSARIAL_QUERIES:
        print(f"  [{q['id']}] {q['type']}: {q['query']}")