"""LLM client, judge parsing and the eval runner, all without network."""

import json
from types import SimpleNamespace

import pytest

from bilingual_rag import llm as llm_module
from bilingual_rag.assistant import AssistantConfig, build_assistant
from bilingual_rag.llm import FakeLLM, LLMClient, MissingConfigError
from evals.judge import JUDGE_MAX_TOKENS, model_family, parse_verdict
from evals.retrieval_eval import load_questions
from evals.run import evaluate_question, latency_stats, summarise


@pytest.fixture
def no_env_files(monkeypatch):
    """Make sure a real .env on this machine is never read by these tests."""
    monkeypatch.setattr(llm_module, "ENV_FILES", ())
    for name in ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "MODEL_MAIN", "MODEL_CHEAP", "MODEL_JUDGE"):
        monkeypatch.delenv(name, raising=False)


def test_missing_key_gives_clear_error(no_env_files):
    with pytest.raises(MissingConfigError, match="OPENROUTER_API_KEY"):
        LLMClient.from_env("main")


def test_missing_model_gives_clear_error(no_env_files, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with pytest.raises(MissingConfigError, match="MODEL_JUDGE"):
        LLMClient.from_env("judge")


def test_complete_records_usage_and_trace(no_env_files, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_CHEAP", "vendor/model-x")
    trace = tmp_path / "traces.jsonl"
    client = LLMClient.from_env("cheap", trace_path=trace)
    fake_result = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Hello [S1]"), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3, cost=0.0004),
        model="vendor/model-x",
    )
    sent = {}

    def fake_create(**kwargs):
        sent.update(kwargs)
        return fake_result

    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    response = client.complete([{"role": "user", "content": "hi"}], meta={"question_id": "en-01"})
    assert response.text == "Hello [S1]" and response.cost_usd == 0.0004
    assert sent["temperature"] == 0 and sent["extra_body"] == {"usage": {"include": True}}
    record = json.loads(trace.read_text().strip())
    assert record["question_id"] == "en-01" and record["prompt_tokens"] == 12 and record["text"] is None
    assert record["finish_reason"] == "stop"  # "length" would mean the reply was cut off
    assert "test-key" not in trace.read_text()  # the key is never logged


def test_parse_verdict_is_robust():
    text = 'Sure! {"correctness": "Correct", "supported": "partly", "reason": "ok"}'
    assert parse_verdict(text) == {"correctness": "correct", "supported": "partly", "reason": "ok"}
    assert parse_verdict("not json")["correctness"] == "unparsed"
    assert model_family("anthropic/claude-x") == "anthropic"


def test_eval_pipeline_end_to_end_with_fake_model():
    assistant = build_assistant(AssistantConfig(collections=("agency",)), llm=FakeLLM())
    questions = {q["id"]: q for q in load_questions()}
    rows = [evaluate_question(assistant, FakeLLM(), questions[qid]) for qid in ("en-18", "en-26", "ar-01")]
    by_id = {r["id"]: r for r in rows}
    assert by_id["en-18"]["status"] == "answered" and by_id["en-18"]["cited_sections"] == ["baggage#5"]
    assert by_id["en-18"]["citation_precision"] == 1.0 and by_id["en-18"]["judge_correctness"] == "correct"
    summary = summarise(rows)
    assert summary["questions"] == 3
    assert summary["false_abstention_on_answerable"].endswith("/2")


class RecordingFakeLLM(FakeLLM):
    """FakeLLM that remembers the max_tokens it was called with."""

    def complete(self, messages, purpose="answer", meta=None, max_tokens=600):
        self.max_tokens = max_tokens
        return super().complete(messages, purpose, meta, max_tokens)


def test_judge_gets_room_for_reasoning_tokens():
    # Live run 2026-10-08: with the default 600-token cap, a reasoning judge model ran out of
    # tokens before finishing its JSON on 3 of 43 answers ("unparsed" verdicts).
    judge = RecordingFakeLLM()
    assistant = build_assistant(AssistantConfig(collections=("agency",)), llm=FakeLLM())
    question = {q["id"]: q for q in load_questions()}["en-18"]
    row = evaluate_question(assistant, judge, question)
    assert judge.max_tokens == JUDGE_MAX_TOKENS >= 2000
    assert row["judge_correctness"] == "correct" and row["latency_s"] >= 0 and row["cited_langs"] == ["en"]


def test_latency_stats():
    stats = latency_stats([1.0, 2.0, 3.0, 4.0, 10.0])
    assert stats == {"mean": 4.0, "median": 3.0, "p95": 10.0}
    assert latency_stats([]) is None


def test_embedding_client_adds_up_cost_and_skips_repeated_texts():
    from bilingual_rag.llm import EmbeddingClient

    client = EmbeddingClient(model="vendor/embed", api_key="test-key", base_url="https://example.com/v1")
    sent = []

    def fake_create(model, input):
        sent.append(list(input))
        data = [SimpleNamespace(index=i, embedding=[float(len(t)), 1.0]) for i, t in enumerate(input)]
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=len(input), cost=0.001))

    client._client = SimpleNamespace(embeddings=SimpleNamespace(create=fake_create))
    assert client.embed(["ab", "abc", "ab"]) == [[2.0, 1.0], [3.0, 1.0], [2.0, 1.0]]
    assert client.embed(["abc"]) == [[3.0, 1.0]]  # remembered: no second request
    assert sent == [["ab", "abc"]]
    assert client.total_cost_usd == 0.001 and client.total_tokens == 2
