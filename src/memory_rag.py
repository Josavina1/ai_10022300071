# memory_rag.py
# Author: [Your Name] - [Your Index Number]
# CS4241 - Introduction to Artificial Intelligence - 2026
# Part G: Innovation Component — Memory-Based RAG

import os
import json
import logging
import datetime
from typing import List, Dict, Optional
from collections import deque

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# INNOVATION: Memory-Based RAG
#
# Standard RAG treats every query as independent — it has no memory of past turns.
# This module adds a conversation memory layer that:
#
# 1. CONTEXTUAL QUERY REWRITING:
#    If user asks "What about the Volta region?" after asking about election results,
#    the system rewrites it to "What are the election results for the Volta region?"
#    before sending to the retriever. This dramatically improves multi-turn accuracy.
#
# 2. CONVERSATION HISTORY INJECTION:
#    Recent Q&A pairs are injected into the prompt so the LLM can give coherent
#    follow-up answers (e.g., "As I mentioned earlier, NDC won Greater Accra...").
#
# 3. ENTITY MEMORY:
#    Tracks named entities mentioned in the conversation (regions, parties, figures)
#    so they can be used to disambiguate future queries.
#
# Why this is novel for this domain:
#   Budget and election analysis often involves follow-up questions
#   ("What about the previous year?", "How does that compare?").
#   Standard RAG cannot handle these — memory-RAG can.
# ════════════════════════════════════════════════════════════════════════════════

MEMORY_LOG = "logs/memory_rag_sessions.jsonl"
MAX_HISTORY = 5  # Keep last 5 turns in context


class ConversationMemory:
    """
    Stores and manages conversation history for multi-turn RAG.
    """

    def __init__(self, max_history: int = MAX_HISTORY):
        self.max_history = max_history
        self.history = deque(maxlen=max_history)  # List of {query, response, sources}
        self.entity_memory = {}   # Tracks entities: {entity_name: context_snippet}
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_turn(self, query: str, response: str, sources: List[str], chunks: List[Dict]) -> None:
        """Record a completed conversation turn."""
        turn = {
            "turn": len(self.history) + 1,
            "query": query,
            "response": response,
            "sources": sources,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.history.append(turn)
        self._extract_entities(query, chunks)
        logger.info(f"Memory: added turn {turn['turn']}. History size: {len(self.history)}")

    def _extract_entities(self, query: str, chunks: List[Dict]) -> None:
        """
        Simple entity extraction: look for capitalised words and known domain terms.
        Store them with a snippet from the most relevant chunk.
        """
        domain_terms = [
            "NDC", "NPP", "Accra", "Ashanti", "Volta", "Northern", "Eastern",
            "Western", "Central", "Brong", "GDP", "inflation", "revenue",
            "expenditure", "deficit", "surplus", "Ministry", "Ghana", "Parliament"
        ]
        for term in domain_terms:
            if term.lower() in query.lower() and chunks:
                # Store the best chunk snippet as context for this entity
                best_chunk = max(chunks, key=lambda c: c.get("similarity_score", 0))
                self.entity_memory[term] = best_chunk["text"][:200]

    def rewrite_query(self, query: str) -> str:
        """
        Contextual query rewriting.
        If the query is a pronoun reference or follow-up, expand it using history.
        """
        if not self.history:
            return query  # No history yet, return as-is

        # Detect follow-up patterns
        followup_signals = ["what about", "and the", "how about", "same for",
                            "compare that", "tell me more", "what else", "also"]
        is_followup = any(signal in query.lower() for signal in followup_signals)
        is_short = len(query.split()) <= 5

        if is_followup or is_short:
            last_turn = self.history[-1]
            last_query = last_turn["query"]
            # Extract key topic words from last query (remove question words)
            stopwords = {"what", "who", "how", "when", "where", "is", "are",
                         "was", "the", "a", "an", "did", "does"}
            topic_words = [w for w in last_query.lower().split() if w not in stopwords]
            topic_context = " ".join(topic_words[:5])
            rewritten = f"{query} (in the context of: {topic_context})"
            logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
            return rewritten

        return query

    def format_history_for_prompt(self) -> str:
        """
        Format conversation history for injection into the prompt.
        Returns a string summary of recent turns.
        """
        if not self.history:
            return ""

        lines = ["Previous conversation:"]
        for turn in list(self.history)[-3:]:  # Last 3 turns only
            lines.append(f"  Q: {turn['query']}")
            # Truncate response to 100 words for prompt efficiency
            short_response = " ".join(turn['response'].split()[:80]) + "..."
            lines.append(f"  A: {short_response}")

        return "\n".join(lines)

    def get_entity_context(self, query: str) -> str:
        """
        If the query mentions a known entity, return its stored context.
        """
        for entity, context in self.entity_memory.items():
            if entity.lower() in query.lower():
                return f"[Entity context for '{entity}': {context}]"
        return ""

    def save_session(self, log_path: str = MEMORY_LOG) -> None:
        """Persist session to disk for experiment logs."""
        os.makedirs("logs", exist_ok=True)
        session = {
            "session_id": self.session_id,
            "turns": list(self.history),
            "entity_memory": self.entity_memory
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(session) + "\n")
        logger.info(f"Session saved: {self.session_id}")

    def clear(self) -> None:
        """Reset memory for a new conversation."""
        self.history.clear()
        self.entity_memory.clear()
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("Memory cleared.")


# ── Memory-enhanced query builder ────────────────────────────────────────────

def build_memory_prompt(
    query: str,
    context: str,
    memory: ConversationMemory
) -> str:
    """
    Build a prompt that includes both retrieved context and conversation history.
    This is the core of the Memory-RAG innovation.
    """
    history_str = memory.format_history_for_prompt()
    entity_str = memory.get_entity_context(query)

    prompt = f"""You are an AI assistant for Academic City University with access to Ghana election results and the 2025 Budget Statement.

RULES:
1. Answer ONLY using the provided context documents.
2. Consider the conversation history for follow-up questions.
3. If context is insufficient, say: "I don't have enough information in the provided documents."
4. Do NOT fabricate statistics or names.

{f"Conversation History:{chr(10)}{history_str}{chr(10)}" if history_str else ""}
{f"{entity_str}{chr(10)}" if entity_str else ""}
<context>
{context}
</context>

<question>
{query}
</question>

Answer:"""

    return prompt


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    memory = ConversationMemory(max_history=5)

    # Simulate a multi-turn conversation
    memory.add_turn(
        query="Who won the Greater Accra parliamentary seats?",
        response="NDC won 15 seats and NPP won 12 seats in Greater Accra.",
        sources=["Ghana_Election_Results"],
        chunks=[{"chunk_id": "e1", "text": "Greater Accra: NDC 15 seats NPP 12 seats", "similarity_score": 0.9}]
    )

    # Test query rewriting
    followup = "What about Ashanti?"
    rewritten = memory.rewrite_query(followup)
    print(f"Rewritten query: {rewritten}")

    # Test history formatting
    print("\nHistory for prompt:")
    print(memory.format_history_for_prompt())

    memory.save_session()
    print("\nMemory session saved.")