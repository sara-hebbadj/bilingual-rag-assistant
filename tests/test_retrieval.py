import pytest

from bilingual_rag.ingest import Chunk, load_chunks
from bilingual_rag.retrieval import BM25Retriever, EmbeddingRetriever, HybridRetriever


def make_chunk(cid: str, text: str, lang: str = "en") -> Chunk:
    return Chunk(cid, cid, f"{cid}#1", lang, cid, "s", text, "x.md", "agency")


@pytest.fixture(scope="module")
def corpus():
    return load_chunks("section")


def test_bm25_prefers_chunk_with_rare_matching_word():
    chunks = [
        make_chunk("a", "refund refund refund money"),
        make_chunk("b", "baggage bags suitcase"),
        make_chunk("c", "money"),
    ]
    hits = BM25Retriever(chunks).search("refund", k=3)
    assert [h.chunk.chunk_id for h in hits] == ["a"]  # zero-score chunks are dropped
    assert hits[0].rank == 1


def test_bm25_finds_english_answer(corpus):
    hits = BM25Retriever(corpus).search("Can I put a power bank in checked luggage?", k=3)
    assert hits[0].chunk.section_id == "baggage#5"


def test_bm25_normalisation_matches_arabic_spelling_variants(corpus):
    # Typed without hamza and with ه instead of ة: still finds the Arabic baggage page.
    hits = BM25Retriever(corpus, analyzer="stemmed").search("الامتعه المفقوده", k=5)
    assert any(h.chunk.doc_id == "baggage" and h.chunk.lang == "ar" for h in hits)


def test_plain_analyzer_misses_spelling_variant():
    chunk = make_chunk("a", "الأمتعة المفقودة", lang="ar")
    assert BM25Retriever([chunk], analyzer="plain").search("الامتعه", k=1) == []
    assert BM25Retriever([chunk], analyzer="stemmed").search("الامتعه", k=1) != []


def test_embedding_retriever_uses_cosine_and_cache(tmp_path):
    vectors = {"apple": [1.0, 0.0], "banana": [0.0, 1.0]}
    calls = []

    def fake_embed(texts):
        calls.append(list(texts))
        return [vectors["apple"] if "apple" in t else vectors["banana"] for t in texts]

    chunks = [make_chunk("a", "apple"), make_chunk("b", "banana")]
    cache = tmp_path / "emb.json"
    retriever = EmbeddingRetriever(chunks, fake_embed, cache_path=cache)
    assert retriever.search("apple pie", k=1)[0].chunk.chunk_id == "a"
    EmbeddingRetriever(chunks, fake_embed, cache_path=cache)  # second build reads the cache
    assert len(calls) == 2  # corpus once + one query; the rebuild embedded nothing


class FixedRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, k=5):
        from bilingual_rag.retrieval import SearchHit

        return [SearchHit(c, 1.0, rank) for rank, c in enumerate(self.chunks[:k], start=1)]


def test_hybrid_rrf_rewards_agreement():
    a, b, c = make_chunk("a", ""), make_chunk("b", ""), make_chunk("c", "")
    hybrid = HybridRetriever([FixedRetriever([a, b]), FixedRetriever([b, c])])
    assert [h.chunk.chunk_id for h in hybrid.search("q", k=3)][0] == "b"  # b is found by both
