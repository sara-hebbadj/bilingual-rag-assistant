"""LLM-as-judge for answer correctness and citation support.

The judge model (MODEL_JUDGE) should come from a different model family than
the model that wrote the answers, so it is less likely to share its blind spots.
Sara also checks 20 answers by hand (see human_review_*.csv) to see how far the
judge can be trusted.
"""

from __future__ import annotations

import json
import re

JUDGE_PROMPT = """You are grading a customer-support assistant for a fictional travel agency.

Question: {question}

Reference answer: {gold_answer}

Assistant answer: {answer}

Source passages the assistant cited:
{cited_sources}

Reply with JSON only, in this exact shape:
{{"correctness": "correct" | "partial" | "incorrect", "supported": "yes" | "partly" | "no", "reason": "<one short sentence>"}}

- correctness: does the assistant answer give the same key facts as the reference answer? Wording and language may differ.
- supported: is every factual claim in the assistant answer stated in the cited passages?"""

VALID = {"correctness": {"correct", "partial", "incorrect"}, "supported": {"yes", "partly", "no"}}


def build_judge_messages(question: str, gold_answer: str, answer: str, cited_sources: str) -> list[dict]:
    prompt = JUDGE_PROMPT.format(
        question=question, gold_answer=gold_answer, answer=answer, cited_sources=cited_sources or "(none)"
    )
    return [{"role": "user", "content": prompt}]


def parse_verdict(text: str) -> dict:
    """Read the first {...} block; any missing or unexpected value becomes 'unparsed'."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    verdict = {}
    for field, allowed in VALID.items():
        value = str(data.get(field, "")).strip().lower()
        verdict[field] = value if value in allowed else "unparsed"
    verdict["reason"] = str(data.get("reason", ""))[:300]
    return verdict


def model_family(model_id: str) -> str:
    """'anthropic/claude-x' -> 'anthropic'. Used to warn when judge and answerer share a family."""
    return model_id.split("/", 1)[0] if "/" in model_id else model_id
