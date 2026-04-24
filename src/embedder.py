# embedder.py
# Author: Josavina - 10022300071
# IT3241 - Introduction to Artificial Intelligence - 2026
# Part B: Custom Embedding Pipeline & Vector Storage

import os
import json
import numpy as np
import faiss
import logging
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# DESIGN DECISIONS:
# Model: all-MiniLM-L6-v2
#   - Free, runs locally, no API needed
#   - 384-dimensional embeddings — compact and fast
#   - Strong performance on semantic similarity benchmarks
#   - Max 256 word-piece tokens (our 400-word chunks fit well)
#
# Vector Store: FAISS (Facebook AI Similarity Search)
#   - Runs fully locally — no cloud, no cost
#   - IndexFlatIP: Inner Product (cosine similarity after L2 normalization)
#   - Scales to millions of vectors efficiently
#   - We persist the index to disk so we don't re-embed on every run
# ════════════════════════════════════════════════════════════════════════════════

MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_STORE_DIR = "vectorstore"
INDEX_FILE = os.path.join(VECTOR_STORE_DIR, "faiss.index")
METADATA_FILE = os.path.join(VECTOR_STORE_DIR, "metadata.json")


class EmbeddingPipeline:
    """
    Handles embedding generation and FAISS vector store management.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.metadata = []  # Stores chunk metadata aligned with FAISS index positions
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")

    # ── Embed text ────────────────────────────────────────────────────────────

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        Returns a numpy array of shape (N, dimension).
        Normalizes vectors for cosine similarity via inner product.
        """
        logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Enables cosine similarity via inner product
        )
        logger.info(f"Embeddings shape: {embeddings.shape}")
        return embeddings

    # ── Build FAISS index ─────────────────────────────────────────────────────

    def build_index(self, chunks: List[Dict]) -> None:
        """
        Build a FAISS index from a list of chunk dicts.
        Each chunk must have a 'text' key.
        Metadata (all fields except 'text') is stored separately.
        """
        logger.info(f"Building FAISS index for {len(chunks)} chunks...")

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embed_texts(texts)

        # FAISS IndexFlatIP = exact inner product search (cosine sim after normalization)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))
        self.metadata = chunks  # Store full chunk dicts aligned by position

        logger.info(f"FAISS index built. Total vectors: {self.index.ntotal}")

    # ── Save & Load ───────────────────────────────────────────────────────────

    def save(self, index_path: str = INDEX_FILE, metadata_path: str = METADATA_FILE) -> None:
        """Persist the FAISS index and metadata to disk."""
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"Saved index to {index_path} and metadata to {metadata_path}")

    def load(self, index_path: str = INDEX_FILE, metadata_path: str = METADATA_FILE) -> bool:
        """Load FAISS index and metadata from disk. Returns True if successful."""
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            logger.warning("No saved index found. Need to build first.")
            return False
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        logger.info(f"Loaded index with {self.index.ntotal} vectors.")
        return True

    # ── Top-K Retrieval ───────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for top_k most similar chunks to the query.
        Returns list of dicts with chunk metadata + similarity_score.
        """
        if self.index is None:
            raise RuntimeError("Index not built or loaded. Call build_index() or load() first.")

        # Embed query (normalized for cosine similarity)
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        # FAISS search — returns distances and indices
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            chunk = dict(self.metadata[idx])
            chunk["similarity_score"] = float(score)
            results.append(chunk)

        logger.info(f"Retrieved {len(results)} chunks for query: '{query[:60]}...'")
        return results

    # ── Query Expansion (Part B Extension) ───────────────────────────────────

    def search_with_query_expansion(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        EXTENSION: Query Expansion
        Generates additional search terms from the original query,
        runs multiple searches, then deduplicates and re-ranks by score.

        Why: A user asking 'election winner in Accra' might miss chunks that say
        'parliamentary results Greater Accra region'. Expansion bridges vocabulary gaps.
        """
        # Expand query with simple heuristic variations
        expansions = self._expand_query(query)
        logger.info(f"Query expanded into {len(expansions)} variants: {expansions}")

        seen_ids = set()
        all_results = []

        for expanded_query in expansions:
            results = self.search(expanded_query, top_k=top_k)
            for r in results:
                chunk_id = r["chunk_id"]
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    r["matched_query"] = expanded_query
                    all_results.append(r)

        # Re-rank by similarity score descending
        all_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_results = all_results[:top_k]

        logger.info(f"After expansion+dedup: {len(top_results)} unique results.")
        return top_results

    def _expand_query(self, query: str) -> List[str]:
        """
        Simple rule-based query expansion.
        Returns original query + 2 keyword-extracted variants.
        """
        expansions = [query]  # Always include original

        # Expansion 1: keywords only (remove stopwords)
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "what",
                     "who", "how", "in", "of", "for", "and", "or", "to", "did"}
        keywords = [w for w in query.lower().split() if w not in stopwords]
        if keywords:
            expansions.append(" ".join(keywords))

        # Expansion 2: domain-specific synonyms
        synonym_map = {
            "election": "voting results parliamentary",
            "budget": "fiscal policy expenditure revenue",
            "winner": "elected candidate votes",
            "ghana": "Republic of Ghana",
            "economy": "GDP growth fiscal",
            "inflation": "price level monetary policy",
            "region": "constituency district",
        }
        expanded_terms = []
        for word in query.lower().split():
            if word in synonym_map:
                expanded_terms.append(synonym_map[word])
        if expanded_terms:
            expansions.append(query + " " + " ".join(expanded_terms))

        return list(dict.fromkeys(expansions))  # Remove duplicates, preserve order

    # ── Failure Case Demonstration (Part B requirement) ──────────────────────

    def demonstrate_failure_case(self, query: str, top_k: int = 5) -> Dict:
        """
        Shows a case where retrieval returns irrelevant results.
        Compares standard search vs query expansion.
        Logs the results for evidence.
        """
        logger.info(f"\n=== FAILURE CASE DEMO: '{query}' ===")

        standard = self.search(query, top_k=top_k)
        expanded = self.search_with_query_expansion(query, top_k=top_k)

        os.makedirs("logs", exist_ok=True)
        with open("logs/retrieval_failure_cases.txt", "a") as f:
            f.write(f"\n\n=== Query: {query} ===\n")
            f.write("-- Standard Retrieval --\n")
            for r in standard:
                f.write(f"  [{r['similarity_score']:.4f}] {r['text'][:120]}\n")
            f.write("-- Expanded Retrieval --\n")
            for r in expanded:
                f.write(f"  [{r['similarity_score']:.4f}] {r['text'][:120]}\n")

        return {"standard": standard, "expanded": expanded}


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Minimal test with dummy chunks
    dummy_chunks = [
        {"chunk_id": "election_0", "text": "Greater Accra region votes: NDC 150000 NPP 120000", "source": "election"},
        {"chunk_id": "election_1", "text": "Ashanti region votes: NPP 200000 NDC 80000", "source": "election"},
        {"chunk_id": "budget_0", "text": "Ghana 2025 budget targets GDP growth of 4.0 percent", "source": "budget"},
        {"chunk_id": "budget_1", "text": "Total government revenue projected at GHS 180 billion", "source": "budget"},
        {"chunk_id": "budget_2", "text": "Inflation is expected to fall to 15 percent by year end 2025", "source": "budget"},
    ]

    pipeline = EmbeddingPipeline()
    pipeline.build_index(dummy_chunks)
    pipeline.save()

    results = pipeline.search("Who won the election in Accra?", top_k=3)
    print("\n--- Search Results ---")
    for r in results:
        print(f"[{r['similarity_score']:.4f}] {r['text']}")

    print("\n--- Expanded Search ---")
    expanded = pipeline.search_with_query_expansion("government spending 2025", top_k=3)
    for r in expanded:
        print(f"[{r['similarity_score']:.4f}] {r['text']} (via: {r.get('matched_query', 'original')})")