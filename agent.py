"""
rag_augmented/agent.py — RAG-augmented agent pattern.

Domain: Intercompany matching — the agent retrieves relevant context
from a vector store BEFORE reasoning about a new matching exception,
so each period's exceptions are checked against precedent from prior
periods rather than evaluated in isolation.

This is a standalone, dependency-free reference implementation of the
pattern used in Close Command's production RAG layer (which uses
ChromaDB and Voyage embeddings). Here, embeddings are a simple bag-of-
words cosine similarity — swap _embed() for a real embedding model call
in production; the retrieval architecture itself does not change.

The defining trait: institutional memory grows over time WITHOUT
retraining any model. Indexing a new resolved exception immediately
makes it retrievable for the next similar exception — contrast with
continual_learning/agent.py, where improvement requires an explicit
retraining step on accumulated outcomes.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class IndexedPrecedent:
    precedent_id: str
    period: str
    entity_pair: str
    description: str
    resolution: str
    embedding: dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    precedent: IndexedPrecedent
    similarity_score: float


@dataclass
class MatchingDecision:
    exception_id: str
    has_precedent: bool
    retrieved_precedents: list[RetrievalResult]
    suggested_resolution: Optional[str]
    confidence: str  # "HIGH" | "MEDIUM" | "LOW" | "NO_PRECEDENT"


class SimpleVectorStore:
    """
    Minimal in-memory vector store using bag-of-words term-frequency
    vectors and cosine similarity. No external dependencies — this is
    intentionally simple so the RETRIEVAL ARCHITECTURE is the focus,
    not the embedding quality. Swap _embed() for a real embedding API
    call (Voyage, OpenAI, etc.) in production; everything else in this
    class is unchanged by that swap.
    """

    def __init__(self) -> None:
        self.precedents: list[IndexedPrecedent] = []

    def index(self, precedent: IndexedPrecedent) -> None:
        precedent.embedding = self._embed(precedent.description)
        self.precedents.append(precedent)

    def retrieve(self, query: str, top_k: int = 3, min_similarity: float = 0.1) -> list[RetrievalResult]:
        query_embedding = self._embed(query)
        scored = []
        for p in self.precedents:
            sim = self._cosine_similarity(query_embedding, p.embedding)
            if sim >= min_similarity:
                scored.append(RetrievalResult(precedent=p, similarity_score=round(sim, 3)))
        scored.sort(key=lambda r: r.similarity_score, reverse=True)
        return scored[:top_k]

    @staticmethod
    def _embed(text: str) -> dict[str, float]:
        """Bag-of-words term frequency vector — a real embedding model produces a dense vector instead."""
        words = re.findall(r"[a-z]+", text.lower())
        if not words:
            return {}
        counts: dict[str, int] = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
        total = len(words)
        return {w: c / total for w, c in counts.items()}

    @staticmethod
    def _cosine_similarity(v1: dict[str, float], v2: dict[str, float]) -> float:
        if not v1 or not v2:
            return 0.0
        common = set(v1.keys()) & set(v2.keys())
        dot = sum(v1[k] * v2[k] for k in common)
        norm1 = math.sqrt(sum(v ** 2 for v in v1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in v2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)


class ICMatchingAgentWithMemory:
    """
    Agent that retrieves prior-period precedent before suggesting a
    resolution for a new IC matching exception. Architectural rule: the
    retrieved precedent INFORMS the suggestion — it is surfaced to a
    human reviewer as supporting context, never auto-applied as the
    final resolution. The agent never silently resolves an exception
    just because a similar one was resolved before.
    """

    def __init__(self, vector_store: Optional[SimpleVectorStore] = None) -> None:
        self.store = vector_store or SimpleVectorStore()

    def index_resolved_exception(
        self, precedent_id: str, period: str, entity_pair: str, description: str, resolution: str
    ) -> None:
        """Called after a human resolves an exception — makes it retrievable for future periods."""
        self.store.index(IndexedPrecedent(
            precedent_id=precedent_id, period=period, entity_pair=entity_pair,
            description=description, resolution=resolution,
        ))

    def evaluate_exception(self, exception_id: str, entity_pair: str, description: str) -> MatchingDecision:
        retrieved = self.store.retrieve(description, top_k=3)

        # Prefer precedents from the SAME entity pair when available — a
        # generically similar precedent from an unrelated entity pair is
        # weaker evidence than one from the exact pair under review.
        same_pair = [r for r in retrieved if r.precedent.entity_pair == entity_pair]
        relevant = same_pair if same_pair else retrieved

        if not relevant:
            return MatchingDecision(
                exception_id=exception_id, has_precedent=False,
                retrieved_precedents=[], suggested_resolution=None, confidence="NO_PRECEDENT",
            )

        top_match = relevant[0]
        if top_match.similarity_score >= 0.6:
            confidence = "HIGH"
        elif top_match.similarity_score >= 0.3:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return MatchingDecision(
            exception_id=exception_id, has_precedent=True,
            retrieved_precedents=relevant,
            suggested_resolution=top_match.precedent.resolution,
            confidence=confidence,
        )
