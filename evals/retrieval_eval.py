"""Retrieval experiment: which chunks does search return for each labelled question?

Runs with no API key (BM25 only):
    python -m evals.retrieval_eval

With EMBEDDING_MODEL and a key set, it also scores embedding and hybrid search:
    python -m evals.retrieval_eval --with-embeddings

Writes to evals/results/:
    retrieval_summary.csv       one row per (config, subset) with hit@k, recall@5, MRR@10
    retrieval_per_question.csv  rank of the first gold chunk for every question and config
    retrieval_hit5.png          chart of hit@5 per analyzer and question group
    abstention_gate.json        can a BM25 score threshold detect unanswerable questions?
    retrieval_run.json          date, command and corpus size of this run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

from bilingual_rag.assistant import build_retriever
from bilingual_rag.ingest import CHUNKING, load_chunks
from bilingual_rag.metrics import first_relevant_rank, hit_at_k, mean, recall_at_k, reciprocal_rank
from bilingual_rag.textnorm import ANALYZERS

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS = EVAL_DIR / "questions.jsonl"
DEPTH = 10  # how many chunks we look at per question (MRR cutoff)


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def configs(with_embeddings: bool) -> list[dict]:
    grid = [{"retriever": "bm25", "analyzer": a, "chunking": c} for c in CHUNKING for a in ANALYZERS]
    if with_embeddings:
        grid += [
            {"retriever": r, "analyzer": "stemmed", "chunking": c} for c in CHUNKING for r in ("embedding", "hybrid")
        ]
    return grid


def config_name(cfg: dict) -> str:
    name = cfg["retriever"] if cfg["retriever"] != "bm25" else f"bm25-{cfg['analyzer']}"
    return f"{name}, {cfg['chunking']}"


def score_question(retriever, question: dict) -> dict:
    hits = retriever.search(question["question"], k=DEPTH)
    sections = [h.chunk.section_id for h in hits]
    gold = question["gold_sections"]
    rank = first_relevant_rank(sections, gold)
    return {
        "id": question["id"],
        "lang": question["lang"],
        "type": question["type"],
        "split": question["split"],
        "gold": " ".join(gold),
        "first_gold_rank": rank or "",
        "hit@1": hit_at_k(sections, gold, 1),
        "hit@3": hit_at_k(sections, gold, 3),
        "hit@5": hit_at_k(sections, gold, 5),
        "recall@5": recall_at_k(sections, gold, 5),
        "rr@10": reciprocal_rank(sections, gold, DEPTH),
        "top_score": round(hits[0].score, 4) if hits else 0.0,
        "top1_lang": hits[0].chunk.lang if hits else "",
        "top5_sections": " ".join(f"{h.chunk.lang}:{h.chunk.section_id}" for h in hits[:5]),
    }


SUBSETS = {
    "all answerable": lambda r: r["type"] != "unanswerable",
    "English questions": lambda r: r["type"] != "unanswerable" and r["lang"] == "en",
    "Arabic questions": lambda r: r["type"] != "unanswerable" and r["lang"] == "ar",
    "same-language": lambda r: r["type"] == "same_lang",
    "English, same-language": lambda r: r["type"] == "same_lang" and r["lang"] == "en",
    "Arabic, same-language": lambda r: r["type"] == "same_lang" and r["lang"] == "ar",
    "cross-language": lambda r: r["type"] == "cross_lang",
}


def plot_hit_at_5(summary: list[dict], path: Path, chunking: str = "section") -> None:
    """Grouped bars: hit@5 of the three BM25 analyzers on three question groups."""
    import matplotlib

    matplotlib.use("Agg")  # no screen needed
    import matplotlib.pyplot as plt

    groups = ["English, same-language", "Arabic, same-language", "cross-language"]
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 of the reference palette
    lookup = {(r["config"], r["subset"]): r for r in summary}
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=150)
    ax.set_facecolor("#fcfcfb")
    width = 0.26
    for i, (analyzer, color) in enumerate(zip(ANALYZERS, colors, strict=True)):
        rows = [lookup[(f"bm25-{analyzer}, {chunking}", g)] for g in groups]
        xs = [g + (i - 1) * (width + 0.02) for g in range(len(groups))]  # small gap between bars
        bars = ax.bar(xs, [r["hit@5"] for r in rows], width, color=color, label=f"BM25 {analyzer}")
        for bar, row in zip(bars, rows, strict=True):
            ax.annotate(
                f"{row['hit@5']:.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#52514e",
            )
    ns = [lookup[(f"bm25-stemmed, {chunking}", g)]["n"] for g in groups]
    ax.set_xticks(
        range(len(groups)), [f"{g}\n(n={n})" for g, n in zip(groups, ns, strict=True)], fontsize=9, color="#0b0b0b"
    )
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("hit@5 (gold section in top 5)", fontsize=9, color="#52514e")
    ax.set_title(f"Retrieval hit@5 by text analyzer ({chunking} chunks, 2026-10-08)", fontsize=10, loc="left")
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", labelsize=8, colors="#52514e", length=0)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, facecolor="#fcfcfb")
    plt.close(fig)


def summarise(name: str, rows: list[dict]) -> list[dict]:
    out = []
    for subset, keep in SUBSETS.items():
        chosen = [r for r in rows if keep(r)]
        out.append(
            {
                "config": name,
                "subset": subset,
                "n": len(chosen),
                **{m: round(mean([r[m] for r in chosen]), 3) for m in ("hit@1", "hit@3", "hit@5", "recall@5")},
                "mrr@10": round(mean([r["rr@10"] for r in chosen]), 3),
            }
        )
    return out


def gate_analysis(rows: list[dict]) -> dict:
    """Pick a BM25 top-score threshold on the dev split; report how it does on the test split.

    "Positive" = the gate abstains. Correct abstention on unanswerable questions is
    good; abstaining on an answerable question is a false abstention.
    """

    def evaluate(subset: list[dict], threshold: float) -> dict:
        unans = [r for r in subset if r["type"] == "unanswerable"]
        ans = [r for r in subset if r["type"] != "unanswerable"]
        caught = sum(r["top_score"] < threshold for r in unans)
        false_abst = sum(r["top_score"] < threshold for r in ans)
        return {
            "unanswerable_abstained": f"{caught}/{len(unans)}",
            "answerable_wrongly_abstained": f"{false_abst}/{len(ans)}",
            "balanced_accuracy": round((caught / len(unans) + 1 - false_abst / len(ans)) / 2, 3),
        }

    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    candidates = sorted({0.0} | {r["top_score"] + 1e-6 for r in dev})
    best = max(candidates, key=lambda t: (evaluate(dev, t)["balanced_accuracy"], -t))
    by_type = {}
    for qtype in ("same_lang", "cross_lang", "unanswerable"):
        scores = sorted(r["top_score"] for r in rows if r["type"] == qtype)
        by_type[qtype] = {"min": scores[0], "median": scores[len(scores) // 2], "max": scores[-1]}
    return {
        "threshold_chosen_on_dev": round(best, 4),
        "dev": evaluate(dev, best),
        "test": evaluate(test, best),
        "top_score_by_type_all_60": by_type,
    }


def _relative(path: Path) -> str:
    """Show paths relative to the repo so saved logs do not contain local folders."""
    try:
        return path.resolve().relative_to(EVAL_DIR.parent).as_posix()
    except ValueError:
        return str(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_table(summary: list[dict], subset: str) -> None:
    print(f"\n{subset}:")
    print("| config | n | hit@1 | hit@3 | hit@5 | MRR@10 |")
    print("|---|---|---|---|---|---|")
    for row in summary:
        if row["subset"] == subset:
            print(
                f"| {row['config']} | {row['n']} | {row['hit@1']:.2f} | {row['hit@3']:.2f} | {row['hit@5']:.2f} | {row['mrr@10']:.2f} |"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval evaluation (no LLM needed for BM25).")
    parser.add_argument(
        "--with-embeddings", action="store_true", help="also run embedding + hybrid (needs EMBEDDING_MODEL)"
    )
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "results")
    parser.add_argument("--no-chart", action="store_true", help="skip the matplotlib chart")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    questions = load_questions()
    per_question, summary, gate = [], [], None
    chunk_counts = {}
    for cfg in configs(args.with_embeddings):
        chunks = load_chunks(cfg["chunking"], collections=("agency",))  # fixed corpus for comparable numbers
        chunk_counts[cfg["chunking"]] = len(chunks)
        retriever = build_retriever(chunks, cfg["retriever"], cfg["analyzer"])
        name = config_name(cfg)
        rows = [{"config": name, **score_question(retriever, q)} for q in questions]
        per_question += rows
        summary += summarise(name, rows)
        if cfg == {"retriever": "bm25", "analyzer": "stemmed", "chunking": "section"}:
            gate = gate_analysis(rows)

    write_csv(args.out / "retrieval_per_question.csv", per_question)
    write_csv(args.out / "retrieval_summary.csv", summary)
    (args.out / "abstention_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    run_info = {
        "date": date.today().isoformat(),
        "command": "python -m evals.retrieval_eval " + " ".join(sys.argv[1:]),
        "questions": len(questions),
        "answerable": sum(q["answerable"] for q in questions),
        "corpus": "data/docs/agency (16 fictional help pages, EN+AR)",
        "chunks": chunk_counts,
    }
    (args.out / "retrieval_run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    if not args.no_chart:
        plot_hit_at_5(summary, args.out / "retrieval_hit5.png")

    for subset in SUBSETS:
        print_table(summary, subset)
    print("\nBM25 score gate (bm25-stemmed, section):", json.dumps(gate, indent=2))
    print(f"\nWrote results to {_relative(args.out)}")


if __name__ == "__main__":
    main()
