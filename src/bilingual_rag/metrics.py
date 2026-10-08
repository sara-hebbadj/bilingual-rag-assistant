"""Evaluation metrics. All deterministic: no model is needed to compute them.

Gold labels are language-independent section ids ("refunds#2"). A retrieved
chunk is relevant if its section id is in the question's gold list.
"""

from __future__ import annotations

from .textnorm import detect_language


def first_relevant_rank(retrieved_sections: list[str], gold: list[str]) -> int | None:
    """1-based rank of the first retrieved chunk from a gold section, or None."""
    for rank, section_id in enumerate(retrieved_sections, start=1):
        if section_id in gold:
            return rank
    return None


def hit_at_k(retrieved_sections: list[str], gold: list[str], k: int) -> float:
    """1.0 if at least one gold section is in the top k, else 0.0 (also called hit rate@k)."""
    rank = first_relevant_rank(retrieved_sections, gold)
    return 1.0 if rank is not None and rank <= k else 0.0


def recall_at_k(retrieved_sections: list[str], gold: list[str], k: int) -> float:
    """Share of the gold sections that appear in the top k (equals hit@k when there is one gold section)."""
    if not gold:
        return 0.0
    found = set(retrieved_sections[:k]) & set(gold)
    return len(found) / len(set(gold))


def reciprocal_rank(retrieved_sections: list[str], gold: list[str], cutoff: int = 10) -> float:
    """1/rank of the first gold chunk within the cutoff, else 0. The mean over questions is MRR."""
    rank = first_relevant_rank(retrieved_sections[:cutoff], gold)
    return 1.0 / rank if rank else 0.0


def citation_precision(cited_sections: list[str], gold: list[str]) -> float | None:
    """Share of an answer's citations that point to a gold section. None if nothing was cited."""
    if not cited_sections:
        return None
    return sum(1 for s in cited_sections if s in gold) / len(cited_sections)


def abstention_correct(abstained: bool, answerable: bool) -> bool:
    """Correct = abstained on an unanswerable question, or answered an answerable one."""
    return abstained != answerable


def language_matches(answer_text: str, question_lang: str) -> bool:
    return detect_language(answer_text) == question_lang


def mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None
