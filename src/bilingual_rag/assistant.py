"""Wire documents, retriever and model together. Used by the app, the CLI and the evals."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .answer import Answer, abstain, generate_answer
from .ingest import REPO_ROOT, Chunk, load_chunks
from .retrieval import BM25Retriever, EmbeddingRetriever, HybridRetriever, Retriever, SearchHit
from .textnorm import detect_language

RETRIEVERS = ("bm25", "embedding", "hybrid")


@dataclass
class AssistantConfig:
    retriever: str = "bm25"  # bm25 is the default: it needs no API and no model download
    analyzer: str = "stemmed"  # plain | normalised | stemmed (see textnorm.tokenize)
    chunking: str = "section"  # section | small (see ingest.CHUNKING)
    k: int = 5  # number of chunks given to the model
    # Optional "retrieval gate": abstain without calling the model when the best
    # BM25 score is below this value. 0 switches it off. Only used with bm25,
    # because hybrid (RRF) scores are on a different scale.
    min_score: float = 0.0
    collections: tuple[str, ...] = ("agency", "public")  # "public" is skipped if not fetched


def build_retriever(chunks: list[Chunk], name: str = "bm25", analyzer: str = "stemmed") -> Retriever:
    if name == "bm25":
        return BM25Retriever(chunks, analyzer=analyzer)
    from .llm import EmbeddingClient  # only needed for embedding/hybrid

    client = EmbeddingClient.from_env()
    slug = re.sub(r"[^A-Za-z0-9]+", "-", client.model)
    cache = REPO_ROOT / "data" / "index" / f"embeddings-{slug}.json"
    dense = EmbeddingRetriever(chunks, client.embed, cache_path=cache)
    if name == "embedding":
        return dense
    if name == "hybrid":
        return HybridRetriever([BM25Retriever(chunks, analyzer=analyzer), dense])
    raise ValueError(f"unknown retriever {name!r}; choose from {RETRIEVERS}")


class RagAssistant:
    def __init__(self, retriever: Retriever, llm=None, config: AssistantConfig | None = None):
        self.retriever = retriever
        self.llm = llm
        self.config = config or AssistantConfig()

    def search(self, question: str, k: int | None = None) -> list[SearchHit]:
        return self.retriever.search(question, k=k or self.config.k)

    def ask(self, question: str, meta: dict | None = None) -> Answer:
        if self.llm is None:
            raise RuntimeError("No language model configured; use search() or set OPENROUTER_API_KEY.")
        hits = self.search(question)
        gate_on = self.config.retriever == "bm25" and self.config.min_score > 0
        if gate_on and (not hits or hits[0].score < self.config.min_score):
            return abstain(question, detect_language(question), "no_sources", hits)
        return generate_answer(question, hits, self.llm, meta)


def build_assistant(config: AssistantConfig | None = None, llm=None) -> RagAssistant:
    config = config or AssistantConfig()
    chunks = load_chunks(config.chunking, config.collections)
    retriever = build_retriever(chunks, config.retriever, config.analyzer)
    return RagAssistant(retriever, llm, config)
