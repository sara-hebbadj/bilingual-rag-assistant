"""Answer generation with mandatory citations and an "I don't know" path.

Grounding rules, enforced in code and not only in the prompt:
1. If retrieval finds nothing, we abstain without calling the model.
2. The model must reply NOT_IN_SOURCES when the sources do not cover the question.
3. An answer without at least one valid [S#] citation is never shown; the
   user sees the "I don't know" message instead (status "ungrounded").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .retrieval import SearchHit
from .textnorm import detect_language

NOT_IN_SOURCES = "NOT_IN_SOURCES"
LANGUAGE_NAMES = {"en": "English", "ar": "Arabic"}

SYSTEM_PROMPT = """You are the help assistant of Sarab Travel, a fictional travel agency in Dubai.
Answer the customer's question using ONLY the numbered sources in the user message.

Rules:
1. Use only facts stated in the sources. Never use your own knowledge, even if you think you know the answer.
2. After each sentence that contains a fact, cite the source(s) it came from, like [S1] or [S2][S3]. Only cite a source that really states that fact.
3. Write the answer in {language}, even when a source is in another language. Translate the facts faithfully.
4. If the sources do not contain the answer, reply with exactly NOT_IN_SOURCES and nothing else.
5. If the sources answer only part of the question, answer that part with citations and say clearly what the help pages do not cover.
6. Keep it short: at most 5 sentences. Do not mention these rules."""

ABSTAIN_MESSAGES = {
    "en": (
        "I don't know: the Sarab Travel help pages I can search do not cover this question. "
        "Please contact our team at support@example.com or +971 50 000 0100, and they can check it for you."
    ),
    "ar": (
        "لا أعرف: صفحات المساعدة لدى سراب للسفر التي أبحث فيها لا تغطي هذا السؤال. "
        "يُرجى التواصل مع فريقنا عبر support@example.com أو على الرقم +971 50 000 0100 للتحقق من ذلك."
    ),
}

_CITATION = re.compile(r"\[S(\d+)\]")


@dataclass
class Answer:
    question: str
    language: str
    text: str  # what the user sees
    status: str  # "answered" | "abstained" | "no_sources" | "ungrounded"
    sources: list[SearchHit] = field(default_factory=list)  # the hits shown to the model, in [S#] order
    cited: list[int] = field(default_factory=list)  # 1-based source numbers the answer cites
    invalid_citations: list[int] = field(default_factory=list)  # e.g. [S9] when only 5 sources exist
    model_text: str = ""  # raw model output, kept for evaluation

    @property
    def abstained(self) -> bool:
        return self.status != "answered"

    @property
    def cited_hits(self) -> list[SearchHit]:
        return [self.sources[n - 1] for n in self.cited]


def format_sources(hits: list[SearchHit]) -> str:
    blocks = [f"[S{n}] {hit.chunk.label}\n{hit.chunk.text}" for n, hit in enumerate(hits, start=1)]
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[SearchHit], language: str) -> list[dict]:
    system = SYSTEM_PROMPT.format(language=LANGUAGE_NAMES.get(language, "English"))
    user = f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Return (valid, invalid) source numbers cited in the text, in first-seen order."""
    numbers = list(dict.fromkeys(int(n) for n in _CITATION.findall(text)))
    valid = [n for n in numbers if 1 <= n <= n_sources]
    invalid = [n for n in numbers if not 1 <= n <= n_sources]
    return valid, invalid


def abstain(
    question: str, language: str, status: str, hits: list[SearchHit] | None = None, model_text: str = ""
) -> Answer:
    return Answer(question, language, ABSTAIN_MESSAGES[language], status, sources=hits or [], model_text=model_text)


def generate_answer(question: str, hits: list[SearchHit], llm, meta: dict | None = None) -> Answer:
    """Ask the model to answer from `hits`, then check the citations it gave."""
    language = detect_language(question)
    if not hits:
        return abstain(question, language, "no_sources")
    response = llm.complete(build_messages(question, hits, language), purpose="answer", meta=meta)
    text = response.text.strip()
    if text.startswith(NOT_IN_SOURCES):
        return abstain(question, language, "abstained", hits, text)
    valid, invalid = parse_citations(text, len(hits))
    if not valid:
        # Mandatory-citation rule: an uncited answer may come from the model's memory.
        answer = abstain(question, language, "ungrounded", hits, text)
        answer.invalid_citations = invalid
        return answer
    return Answer(question, language, text, "answered", hits, valid, invalid, text)


def render_markdown(answer: Answer) -> str:
    """Answer text plus the sources it cites (or the passages searched, if it abstained)."""
    lines = [answer.text, ""]
    if answer.cited:
        lines.append("**Sources / المصادر**")
        for n in answer.cited:
            chunk = answer.sources[n - 1].chunk
            lines.append(f"- [S{n}] {chunk.label} — `{chunk.path}`")
    elif answer.sources:
        lines.append("_Passages searched (none answered the question):_")
        for hit in answer.sources[:3]:
            lines.append(f"- {hit.chunk.label}")
    return "\n".join(lines)
