from bilingual_rag.metrics import (
    abstention_correct,
    citation_precision,
    first_relevant_rank,
    hit_at_k,
    language_matches,
    mean,
    recall_at_k,
    reciprocal_rank,
)

RETRIEVED = ["a#1", "b#2", "c#1", "b#2"]


def test_rank_hit_and_mrr():
    assert first_relevant_rank(RETRIEVED, ["c#1"]) == 3
    assert hit_at_k(RETRIEVED, ["c#1"], 2) == 0.0
    assert hit_at_k(RETRIEVED, ["c#1"], 3) == 1.0
    assert reciprocal_rank(RETRIEVED, ["b#2"]) == 0.5
    assert reciprocal_rank(RETRIEVED, ["z#9"]) == 0.0
    assert reciprocal_rank(RETRIEVED, ["c#1"], cutoff=2) == 0.0


def test_recall_counts_distinct_gold_sections():
    assert recall_at_k(RETRIEVED, ["a#1", "c#1"], 2) == 0.5
    assert recall_at_k(RETRIEVED, ["a#1", "c#1"], 3) == 1.0


def test_citation_precision():
    assert citation_precision(["a#1", "b#2"], ["a#1"]) == 0.5
    assert citation_precision([], ["a#1"]) is None


def test_abstention_correct_truth_table():
    assert abstention_correct(abstained=True, answerable=False)
    assert abstention_correct(abstained=False, answerable=True)
    assert not abstention_correct(abstained=True, answerable=True)
    assert not abstention_correct(abstained=False, answerable=False)


def test_language_match_and_mean():
    assert language_matches("يستغرق الاسترداد 14 يوماً [S1]", "ar")
    assert not language_matches("Refunds take 14 days [S1]", "ar")
    assert mean([1.0, None, 0.0]) == 0.5
    assert mean([None]) is None
