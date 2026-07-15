"""
Lightweight BM25 retrieval for MathRAG.

This module avoids an extra runtime dependency while giving the vector
retriever a lexical recall path for formula names and theorem names.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from src.loader.math_text import build_math_search_text
from src.retriever.query_rewriter import rewrite_query_for_retrieval


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """Tokenize Chinese textbook text into single CJK chars and alnum terms."""
    return [token.lower() for token in TOKEN_PATTERN.findall(text or "")]


@dataclass
class BM25Result:
    vector_id: int
    score: float
    metadata: dict[str, Any]


class BM25Retriever:
    """In-memory BM25 retriever over chunk metadata."""

    def __init__(
        self,
        metadata_by_vector_id: dict[int, dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.metadata_by_vector_id = metadata_by_vector_id
        self.k1 = k1
        self.b = b
        self.doc_tokens: dict[int, list[str]] = {}
        self.term_freqs: dict[int, Counter[str]] = {}
        self.doc_freqs: Counter[str] = Counter()
        self.doc_lengths: dict[int, int] = {}
        self.avg_doc_len = 0.0
        self._build()

    def _build(self) -> None:
        total_len = 0
        for vector_id, metadata in self.metadata_by_vector_id.items():
            content = (
                metadata.get("search_text")
                or metadata.get("text")
                or metadata.get("content")
                or ""
            )
            title = metadata.get("title") or ""
            tokens = tokenize(f"{title}\n{content}")
            if not tokens:
                continue

            token_counts = Counter(tokens)
            self.doc_tokens[vector_id] = tokens
            self.term_freqs[vector_id] = token_counts
            self.doc_lengths[vector_id] = len(tokens)
            self.doc_freqs.update(token_counts.keys())
            total_len += len(tokens)

        self.avg_doc_len = total_len / len(self.doc_tokens) if self.doc_tokens else 0.0

    def _idf(self, term: str) -> float:
        doc_count = len(self.doc_tokens)
        doc_freq = self.doc_freqs.get(term, 0)
        return math.log(1 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))

    def score_document(self, query_tokens: list[str], vector_id: int) -> float:
        if vector_id not in self.term_freqs or not self.avg_doc_len:
            return 0.0

        score = 0.0
        term_freq = self.term_freqs[vector_id]
        doc_len = self.doc_lengths[vector_id]
        for term in set(query_tokens):
            freq = term_freq.get(term, 0)
            if freq <= 0:
                continue
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += self._idf(term) * numerator / denominator
        return score

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        *,
        rewrite_query: bool = True,
    ) -> list[BM25Result]:
        query_text = (
            rewrite_query_for_retrieval(query)
            if rewrite_query
            else build_math_search_text(query)
        )
        query_tokens = tokenize(query_text)
        if not query_tokens:
            return []

        scored = []
        for vector_id in self.doc_tokens:
            score = self.score_document(query_tokens, vector_id)
            if score > 0:
                scored.append(
                    BM25Result(
                        vector_id=vector_id,
                        score=score,
                        metadata=self.metadata_by_vector_id[vector_id],
                    )
                )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]
