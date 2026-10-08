from bilingual_rag.ingest import (
    chunk_documents,
    load_chunks,
    load_documents,
    pack_units,
    parse_front_matter,
    parse_sections,
    split_units,
)

SAMPLE = """---
doc_id: demo
title: Demo page
lang: en
---

# Demo page

## 1. First topic

One. Two. Three.

## 2. Second topic

- a bullet
- another bullet
"""


def test_front_matter_and_sections():
    meta, body = parse_front_matter(SAMPLE)
    assert meta == {"doc_id": "demo", "title": "Demo page", "lang": "en"}
    sections = parse_sections(body)
    assert [(s.number, s.heading) for s in sections] == [(1, "First topic"), (2, "Second topic")]
    assert sections[1].body == "- a bullet\n- another bullet"


def test_split_units_handles_arabic_question_mark():
    assert split_units("هل يمكن؟ نعم يمكن.") == ["هل يمكن؟", "نعم يمكن."]


def test_pack_units_respects_size_and_overlap():
    units = ["a b c", "d e f", "g h i"]  # 3 words each
    chunks = pack_units(units, max_words=6, overlap_sentences=1)
    assert chunks == ["a b c\nd e f", "d e f\ng h i"]
    for chunk in chunks:
        assert len(chunk.split()) <= 6


def test_pack_units_keeps_long_sentence_whole():
    long = " ".join(["w"] * 10)
    assert pack_units([long], max_words=5, overlap_sentences=1) == [long]


def test_corpus_is_parallel_except_single_language_pages():
    docs = load_documents()
    by_lang = {lang: {d.doc_id: len(d.sections) for d in docs if d.lang == lang} for lang in ("en", "ar")}
    only_en = set(by_lang["en"]) - set(by_lang["ar"])
    only_ar = set(by_lang["ar"]) - set(by_lang["en"])
    assert only_en == {"travel-insurance"} and only_ar == {"umrah-packages"}
    for doc_id in set(by_lang["en"]) & set(by_lang["ar"]):
        assert by_lang["en"][doc_id] == by_lang["ar"][doc_id], doc_id  # same section numbers


def test_every_document_is_marked_fictional():
    for doc in load_documents():
        assert doc.meta.get("notice"), doc.path


def test_chunk_ids_unique_and_section_ids_language_free():
    for chunking in ("section", "small"):
        chunks = load_chunks(chunking)
        assert len({c.chunk_id for c in chunks}) == len(chunks)
        assert all("/" not in c.section_id for c in chunks)


def test_small_chunking_makes_more_chunks():
    docs = load_documents()
    assert len(chunk_documents(docs, "small")) > len(chunk_documents(docs, "section"))
