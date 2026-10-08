"""Checks on the labelled question set, so a typo in a gold label fails CI."""

from collections import Counter

from bilingual_rag.ingest import load_documents
from evals.retrieval_eval import load_questions


def test_question_counts_match_the_spec():
    questions = load_questions()
    assert len(questions) == 60
    assert Counter(q["lang"] for q in questions) == {"en": 30, "ar": 30}
    assert sum(not q["answerable"] for q in questions) == 10
    assert len({q["id"] for q in questions}) == 60


def test_gold_sections_exist_and_unanswerables_have_none():
    sections = {f"{d.doc_id}#{s.number}" for d in load_documents() for s in d.sections}
    for q in load_questions():
        if q["answerable"]:
            assert q["gold_sections"], q["id"]
            assert set(q["gold_sections"]) <= sections, q["id"]
        else:
            assert q["gold_sections"] == [], q["id"]


def test_cross_language_questions_point_to_single_language_pages():
    docs = {(d.doc_id, d.lang) for d in load_documents()}
    for q in load_questions():
        if q["type"] == "cross_lang":
            for gold in q["gold_sections"]:
                doc_id = gold.split("#")[0]
                assert (doc_id, q["lang"]) not in docs, q["id"]  # no page in the question's language


def test_both_splits_contain_unanswerable_questions():
    questions = load_questions()
    for split in ("dev", "test"):
        assert any(q["split"] == split and not q["answerable"] for q in questions)
