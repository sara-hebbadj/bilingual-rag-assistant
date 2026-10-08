"""LLM client, judge parsing and the eval runner, all without network."""

import json
from types import SimpleNamespace

import pytest

from bilingual_rag import llm as llm_module
from bilingual_rag.assistant import AssistantConfig, build_assistant
from bilingual_rag.llm import FakeLLM, LLMClient, MissingConfigError
from evals.judge import model_family, parse_verdict
from evals.retrieval_eval import load_questions
from evals.run import evaluate_question, summarise


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
        choices=[SimpleNamespace(message=SimpleNamespace(content="Hello [S1]"))],
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
