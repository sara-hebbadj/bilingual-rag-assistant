"""Retrievers: BM25 keyword search (default), embedding search, and a hybrid.

BM25 is written out by hand (about 30 lines) so every step can be explained.
The embedding retriever needs an embeddings API (see llm.EmbeddingClient);
it is optional because the build machine cannot download local models.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .ingest import Chunk
from .textnorm import tokenize


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    rank: int  # 1 = best


class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[SearchHit]: ...


def _to_hits(chunks: list[Chunk], scores: list[float], k: int) -> list[SearchHit]:
    """Sort by score (highest first), drop zero scores, keep the top k."""
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    order = [i for i in order if scores[i] > 0][:k]
    return [SearchHit(chunks[i], float(scores[i]), rank) for rank, i in enumerate(order, start=1)]


class BM25Retriever:
    """Okapi BM25 over chunk.search_text.

    score(q, d) = sum over query terms t of
        idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * len(d) / avg_len))
    k1 limits how much repeating a word helps; b controls the penalty for long chunks.
    """

    def __init__(self, chunks: list[Chunk], analyzer: str = "stemmed", k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.analyzer = analyzer
        self.k1 = k1
        self.b = b
        docs = [tokenize(c.search_text, analyzer) for c in chunks]
        self.term_counts = [Counter(tokens) for tokens in docs]
        self.lengths = [len(tokens) for tokens in docs]
        self.avg_length = sum(self.lengths) / max(len(docs), 1)
        doc_freq = Counter(term for tokens in docs for term in set(tokens))
        n = len(docs)
        # The "+1 inside the log" (Lucene's version) keeps idf positive for common words.
        self.idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    def score(self, query_tokens: list[str], index: int) -> float:
        counts = self.term_counts[index]
        length_norm = 1 - self.b + self.b * self.lengths[index] / self.avg_length
        total = 0.0
        for term in set(query_tokens):
            tf = counts.get(term, 0)
            if tf:
                total += self.idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
        return total

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        query_tokens = tokenize(query, self.analyzer)
        scores = [self.score(query_tokens, i) for i in range(len(self.chunks))]
        return _to_hits(self.chunks, scores, k)


EmbedFn = Callable[[list[str]], list[list[float]]]


class EmbeddingRetriever:
    """Cosine similarity between the query vector and every chunk vector.

    With a few hundred chunks a plain numpy matrix product is instant; this is
    exactly what a FAISS "flat" index does, so FAISS is not needed at this size.
    Chunk vectors are cached on disk so the corpus is embedded only once.
    """

    def __init__(self, chunks: list[Chunk], embed: EmbedFn, cache_path: Path | None = None):
        self.chunks = chunks
        self.embed = embed
        texts = [c.search_text for c in chunks]
        self.matrix = _normalise_rows(np.array(self._embed_with_cache(texts, cache_path), dtype=np.float32))

    def _embed_with_cache(self, texts: list[str], cache_path: Path | None) -> list[list[float]]:
        cache: dict[str, list[float]] = {}
        if cache_path and cache_path.exists():
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
        missing = [t for t, key in zip(texts, keys, strict=True) if key not in cache]
        if missing:
            for text, vector in zip(missing, self.embed(missing), strict=True):
                cache[hashlib.sha1(text.encode("utf-8")).hexdigest()] = vector
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
        return [cache[key] for key in keys]

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        query_vector = _normalise_rows(np.array(self.embed([query]), dtype=np.float32))[0]
        scores = (self.matrix @ query_vector).tolist()
        return _to_hits(self.chunks, scores, k)


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class HybridRetriever:
    """Combine several retrievers with Reciprocal Rank Fusion (RRF).

    Each retriever votes 1 / (rrf_k + rank) for the chunks it returns. RRF only
    uses ranks, so BM25 scores and cosine similarities never need to be put on
    the same scale. rrf_k = 60 is the value from the original RRF paper.
    """

    def __init__(self, retrievers: list[Retriever], rrf_k: int = 60, depth: int = 20):
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.depth = depth

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        fused: dict[str, float] = {}
        chunk_by_id: dict[str, Chunk] = {}
        for retriever in self.retrievers:
            for hit in retriever.search(query, k=self.depth):
                chunk_id = hit.chunk.chunk_id
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1 / (self.rrf_k + hit.rank)
                chunk_by_id[chunk_id] = hit.chunk
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [SearchHit(chunk_by_id[c], fused[c], rank) for rank, c in enumerate(best, start=1)]
