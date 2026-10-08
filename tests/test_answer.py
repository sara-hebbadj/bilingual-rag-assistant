from bilingual_rag.answer import ABSTAIN_MESSAGES, NOT_IN_SOURCES, build_messages, generate_answer, parse_citations
from bilingual_rag.assistant import AssistantConfig, RagAssistant
from bilingual_rag.ingest import Chunk
from bilingual_rag.llm import FakeLLM
from bilingual_rag.retrieval import BM25Retriever, SearchHit


def hits_for(*texts):
    chunks = [
        Chunk(f"c{i}", "refunds", f"refunds#{i}", "en", "Refunds", f"Part {i}", t, "x.md", "agency")
        for i, t in enumerate(texts, start=1)
    ]
    return [SearchHit(c, 1.0, i) for i, c in enumerate(chunks, start=1)]


def test_parse_citations_splits_valid_and_invalid():
    assert parse_citations("A [S1]. B [S3][S1]. C [S9].", n_sources=3) == ([1, 3], [9])


def test_prompt_numbers_sources_and_sets_language():
    messages = build_messages("كم المدة؟", hits_for("Refunds take 14 days."), "ar")
    assert "Write the answer in Arabic" in messages[0]["content"]
    assert "[S1] Refunds › Part 1 (EN)\nRefunds take 14 days." in messages[1]["content"]


def test_answer_with_valid_citation_is_shown():
    llm = FakeLLM(responses=["Refunds take 7 to 14 working days [S1]."])
    answer = generate_answer("How long do refunds take?", hits_for("Refunds take 7 to 14 working days."), llm)
    assert answer.status == "answered" and answer.cited == [1]
    assert answer.cited_hits[0].chunk.section_id == "refunds#1"


def test_model_abstention_gives_localised_message():
    llm = FakeLLM(responses=[NOT_IN_SOURCES])
    answer = generate_answer("هل يمكنني اصطحاب قطتي؟", hits_for("unrelated"), llm)
    assert answer.status == "abstained" and answer.abstained
    assert answer.text == ABSTAIN_MESSAGES["ar"]


def test_uncited_answer_is_blocked():
    llm = FakeLLM(responses=["Yes, cats are welcome in the cabin."])  # no [S#]: maybe from model memory
    answer = generate_answer("Can I bring my cat?", hits_for("unrelated"), llm)
    assert answer.status == "ungrounded"
    assert answer.text == ABSTAIN_MESSAGES["en"]
    assert answer.model_text == "Yes, cats are welcome in the cabin."


def test_citation_to_missing_source_is_blocked():
    llm = FakeLLM(responses=["Something [S7]."])
    answer = generate_answer("Question?", hits_for("only one source"), llm)
    assert answer.status == "ungrounded" and answer.invalid_citations == [7]


def test_no_hits_means_no_model_call():
    llm = FakeLLM()
    answer = generate_answer("Question?", [], llm)
    assert answer.status == "no_sources" and llm.calls == []


def test_retrieval_gate_skips_model_when_score_is_low():
    chunks = hits_for("refunds take fourteen days")
    llm = FakeLLM()
    assistant = RagAssistant(BM25Retriever([h.chunk for h in chunks]), llm, AssistantConfig(min_score=100.0))
    assert assistant.ask("refunds").status == "no_sources"
    assert llm.calls == []
