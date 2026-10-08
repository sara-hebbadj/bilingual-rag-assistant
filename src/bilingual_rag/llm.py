"""The only place that talks to a language model.

Settings come from a .env file (python-dotenv) or from environment variables:
OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODEL_MAIN, MODEL_CHEAP, MODEL_JUDGE,
and optionally EMBEDDING_MODEL (+ EMBEDDING_BASE_URL / EMBEDDING_API_KEY).
Every call is appended to a JSONL trace file with tokens, cost and latency.
The key itself is never printed or logged.

FakeLLM has the same interface and needs no network; tests and --dry-run use it.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from .ingest import REPO_ROOT

# Looked up in this order; values already set in the environment always win.
# The second file is the shared "Portfolio Projects/.env" two folders above the repo.
ENV_FILES = (REPO_ROOT / ".env", REPO_ROOT.parent.parent / ".env")
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TRACE_PATH = REPO_ROOT / "evals" / "traces.jsonl"
MODEL_ROLES = {"main": "MODEL_MAIN", "cheap": "MODEL_CHEAP", "judge": "MODEL_JUDGE"}


class MissingConfigError(RuntimeError):
    """Raised when a key or model id is missing, with a message saying what to set."""


def load_env() -> None:
    for path in ENV_FILES:
        if path.exists():
            load_dotenv(path, override=False)


def has_llm_config() -> bool:
    load_env()
    return bool(os.getenv("OPENROUTER_API_KEY")) and bool(os.getenv("MODEL_CHEAP") or os.getenv("MODEL_MAIN"))


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    latency_s: float = 0.0


def write_trace(trace_path: Path | None, record: dict) -> None:
    if trace_path is None:
        return
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class LLMClient:
    """Chat completions through any OpenAI-compatible API (OpenRouter by default)."""

    def __init__(
        self, model: str, api_key: str, base_url: str = DEFAULT_BASE_URL, trace_path: Path | None = DEFAULT_TRACE_PATH
    ):
        from openai import OpenAI  # imported here so offline tests never need it

        self.model = model
        self.base_url = base_url
        self.trace_path = trace_path
        self.total_cost_usd = 0.0
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=60)

    @classmethod
    def from_env(cls, role: str = "main", trace_path: Path | None = DEFAULT_TRACE_PATH) -> LLMClient:
        load_env()
        api_key = os.getenv("OPENROUTER_API_KEY")
        model = os.getenv(MODEL_ROLES[role])
        if not api_key:
            raise MissingConfigError(
                "OPENROUTER_API_KEY is not set (add it to Portfolio Projects/.env or the repo's .env)."
            )
        if not model:
            raise MissingConfigError(
                f"{MODEL_ROLES[role]} is not set (copy a model id from https://openrouter.ai/models)."
            )
        base_url = os.getenv("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL
        return cls(model=model, api_key=api_key, base_url=base_url, trace_path=trace_path)

    def complete(
        self, messages: list[dict], purpose: str = "answer", meta: dict | None = None, max_tokens: int = 600
    ) -> LLMResponse:
        extra = {}
        if "openrouter.ai" in self.base_url:
            extra["extra_body"] = {"usage": {"include": True}}  # ask OpenRouter to return the cost
        started = time.perf_counter()
        record = {"time": datetime.now(UTC).isoformat(), "purpose": purpose, "model": self.model, **(meta or {})}
        try:
            result = self._client.chat.completions.create(
                model=self.model, messages=messages, temperature=0, max_tokens=max_tokens, **extra
            )
        except Exception as error:  # log the failure, then let the caller decide
            write_trace(
                self.trace_path,
                {
                    **record,
                    "outcome": "error",
                    "error": type(error).__name__,
                    "latency_s": round(time.perf_counter() - started, 3),
                },
            )
            raise
        usage = result.usage
        response = LLMResponse(
            text=result.choices[0].message.content or "",
            model=result.model or self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cost_usd=getattr(usage, "cost", None),
            latency_s=round(time.perf_counter() - started, 3),
        )
        self.total_cost_usd += response.cost_usd or 0.0
        write_trace(self.trace_path, {**record, **asdict(response), "text": None, "outcome": "ok"})
        return response


class FakeLLM:
    """Offline stand-in for LLMClient. Its answers are NOT real model output.

    - If `responses` is given, it returns them in order (a "recorded" client).
    - Otherwise it imitates the answer format: it copies the first sentence of
      source [S1] and cites it, or returns NOT_IN_SOURCES when the prompt has
      no sources; for judge prompts it returns a fixed JSON verdict.
    """

    def __init__(self, responses: list[str] | None = None, trace_path: Path | None = None):
        self.model = "fake-llm"
        self.responses = list(responses or [])
        self.trace_path = trace_path
        self.total_cost_usd = 0.0
        self.calls: list[list[dict]] = []

    def complete(
        self, messages: list[dict], purpose: str = "answer", meta: dict | None = None, max_tokens: int = 600
    ) -> LLMResponse:
        self.calls.append(messages)
        text = self.responses.pop(0) if self.responses else self._imitate(messages, purpose)
        response = LLMResponse(text=text, model=self.model, cost_usd=0.0)
        write_trace(
            self.trace_path,
            {
                "time": datetime.now(UTC).isoformat(),
                "purpose": purpose,
                **(meta or {}),
                **asdict(response),
                "text": None,
                "outcome": "ok",
                "fake": True,
            },
        )
        return response

    @staticmethod
    def _imitate(messages: list[dict], purpose: str) -> str:
        prompt = messages[-1]["content"]
        if purpose == "judge":
            return '{"correctness": "correct", "supported": "yes", "reason": "fake judge"}'
        match = re.search(r"\[S1\][^\n]*\n(.+)", prompt)
        if not match:
            return "NOT_IN_SOURCES"
        first_sentence = re.split(r"(?<=[.!?؟])\s", match.group(1).strip())[0]
        return f"{first_sentence} [S1]"


class EmbeddingClient:
    """Embeddings through an OpenAI-compatible /embeddings endpoint."""

    def __init__(self, model: str, api_key: str, base_url: str, batch_size: int = 64):
        from openai import OpenAI

        self.model = model
        self.batch_size = batch_size
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=60)

    @classmethod
    def from_env(cls) -> EmbeddingClient:
        load_env()
        model = os.getenv("EMBEDDING_MODEL")
        api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL
        if not model or not api_key:
            raise MissingConfigError("Set EMBEDDING_MODEL and an API key to use embedding or hybrid search.")
        return cls(model=model, api_key=api_key, base_url=base_url)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            result = self._client.embeddings.create(model=self.model, input=batch)
            vectors.extend(item.embedding for item in sorted(result.data, key=lambda d: d.index))
        return vectors
