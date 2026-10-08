"""Put the saved live runs side by side: one row per (run, question set).

    python -m evals.compare_runs

Reads evals/results/run_*.jsonl and their _summary.json files and writes
evals/results/live_runs_comparison.csv. Every number is recomputed from the
saved per-question rows (no model calls), so the README tables can be checked.
Runs that searched the public pages too are split into the original 60
questions ("agency") and the 10 public-page questions ("public").
"""

from __future__ import annotations

import csv
import json

from .run import EVAL_DIR, _rate, summarise

RESULTS = EVAL_DIR / "results"


def compare_rows() -> list[dict]:
    table = []
    for path in sorted(RESULTS.glob("run_*.jsonl")):
        info = json.loads(path.with_name(f"{path.stem}_summary.json").read_text(encoding="utf-8"))
        with path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for corpus in sorted({r.get("corpus", "agency") for r in rows}):
            chosen = [r for r in rows if r.get("corpus", "agency") == corpus]
            s = summarise(chosen)
            cross = [r for r in chosen if r["type"] == "cross_lang"]
            table.append(
                {
                    "run": path.stem,
                    "answer_model": info["answer_model"],
                    "judge_model": info["judge_model"],
                    "retriever": info["retriever"],
                    "searched": "+".join(info["collections"]),
                    "questions": f"{corpus} ({len(chosen)})",
                    "limit": info.get("limit"),
                    "judge_correct_of_answerable": s["judge_correct_of_all_answerable"],
                    "abstained_on_unanswerable": s["correct_abstention_on_unanswerable"],
                    "abstained_on_answerable": s["false_abstention_on_answerable"],
                    "answered_with_gold_citation": s["answered_with_gold_citation"],
                    "answered_all_citations_gold": s["answered_all_citations_gold"],
                    "judge_supported": s["judge_supported"],
                    "language_match": s["language_match"],
                    "cross_lang_judge_correct": _rate(cross, lambda r: r.get("judge_correctness") == "correct"),
                    "judge_unparsed": s["judge_unparsed"],
                    "latency_mean_s": s["latency_s"]["mean"],
                    "latency_p95_s": s["latency_s"]["p95"],
                    "run_cost_usd": info["cost_usd"],
                }
            )
    return table


def main() -> None:
    table = compare_rows()
    out = RESULTS / "live_runs_comparison.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    for row in table:
        print(" | ".join(str(v) for v in row.values()))
    print(f"\nWrote {out.relative_to(EVAL_DIR.parent).as_posix()}")


if __name__ == "__main__":
    main()
