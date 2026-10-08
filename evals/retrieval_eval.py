"""Retrieval experiment: which chunks does search return for each labelled question?

Runs with no API key (BM25 only):
    python -m evals.retrieval_eval

With EMBEDDING_MODEL and a key set, it also scores embedding and hybrid search:
    python -m evals.retrieval_eval --with-embeddings

Bigger corpus (agency pages + the CC BY-SA public pages) and the extra public-page questions:
    python -m evals.retrieval_eval --with-embeddings --collections agency public \
        --questions evals/questions.jsonl evals/questions_public.jsonl --out evals/results/agency_plus_public

Writes to evals/results/:
    retrieval_summary.csv       one row per (config, subset) with hit@k, recall@5, MRR@10
    retrieval_per_question.csv  rank of the first gold chunk for every question and config
    retrieval_hit5.png          chart of hit@5 per analyzer and question group
    retrieval_hybrid_hit5.png   chart of hit@5 for BM25 vs embedding vs hybrid (with --with-embeddings)
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
from bilingual_rag.llm import EmbeddingClient
from bilingual_rag.metrics import first_relevant_rank, hit_at_k, mean, recall_at_k, reciprocal_rank
from bilingual_rag.textnorm import ANALYZERS

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS = EVAL_DIR / "questions.jsonl"
DEPTH = 10  # how many chunks we look at per question (MRR cutoff)


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_question_files(paths: list[Path]) -> list[dict]:
    """Several question files in one list. Questions without a "corpus" field are agency questions."""
    questions = [q for path in paths for q in load_questions(path)]
    for q in questions:
        q.setdefault("corpus", "agency")
    return questions


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
        "corpus": question.get("corpus", "agency"),
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


def _agency(row: dict) -> bool:
    """The original 60 questions (about the agency pages); public-page questions are reported apart."""
    return row.get("corpus", "agency") == "agency"


SUBSETS = {
    "all answerable": lambda r: _agency(r) and r["type"] != "unanswerable",
    "English questions": lambda r: _agency(r) and r["type"] != "unanswerable" and r["lang"] == "en",
    "Arabic questions": lambda r: _agency(r) and r["type"] != "unanswerable" and r["lang"] == "ar",
    "same-language": lambda r: _agency(r) and r["type"] == "same_lang",
    "English, same-language": lambda r: _agency(r) and r["type"] == "same_lang" and r["lang"] == "en",
    "Arabic, same-language": lambda r: _agency(r) and r["type"] == "same_lang" and r["lang"] == "ar",
    "cross-language": lambda r: _agency(r) and r["type"] == "cross_lang",
    "public-page questions": lambda r: r.get("corpus") == "public",
    "public-page, cross-language": lambda r: r.get("corpus") == "public" and r["type"] == "cross_lang",
}


def plot_hit_at_5(
    summary: list[dict],
    path: Path,
    chunking: str = "section",
    series: list[tuple[str, str]] | None = None,
    title: str = "Retrieval hit@5 by text analyzer",
) -> None:
    """Grouped bars: hit@5 of up to three configs (default: the BM25 analyzers) on three question groups.

    series = [(legend label, config name), ...]
    """
    import matplotlib

    matplotlib.use("Agg")  # no screen needed
    import matplotlib.pyplot as plt

    series = series or [(f"BM25 {a}", f"bm25-{a}, {chunking}") for a in ANALYZERS]
    groups = ["English, same-language", "Arabic, same-language", "cross-language"]
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 of the reference palette
    lookup = {(r["config"], r["subset"]): r for r in summary}
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=150)
    ax.set_facecolor("#fcfcfb")
    width = 0.26
    for i, ((label, config), color) in enumerate(zip(series, colors, strict=False)):
        rows = [lookup[(config, g)] for g in groups]
        xs = [g + (i - 1) * (width + 0.02) for g in range(len(groups))]  # small gap between bars
        bars = ax.bar(xs, [r["hit@5"] for r in rows], width, color=color, label=label)
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
    ns = [lookup[(series[0][1], g)]["n"] for g in groups]
    ax.set_xticks(
        range(len(groups)), [f"{g}\n(n={n})" for g, n in zip(groups, ns, strict=True)], fontsize=9, color="#0b0b0b"
    )
    ax.set_ylim(0, 1.22)  # headroom above the bars for the legend
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("hit@5 (gold section in top 5)", fontsize=9, color="#52514e")
    ax.set_title(f"{title} ({chunking} chunks, 2026-10-08)", fontsize=10, loc="left")
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", labelsize=8, colors="#52514e", length=0)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3)
    fig.tight_layout()
    fig.savefig(path, facecolor="#fcfcfb")
    plt.close(fig)


def summarise(name: str, rows: list[dict]) -> list[dict]:
    out = []
    for subset, keep in SUBSETS.items():
        chosen = [r for r in rows if keep(r)]
        if not chosen:
            continue  # e.g. no public-page questions in this run
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

    rows = [r for r in rows if _agency(r)]  # the gate is tuned and tested on the original 60 questions only
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
    parser.add_argument(
        "--collections", nargs="+", default=["agency"], help="document folders to search (default: agency only)"
    )
    parser.add_argument("--questions", nargs="+", type=Path, default=[QUESTIONS], help="question files (JSONL)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    questions = load_question_files(args.questions)
    embedder = EmbeddingClient.from_env() if args.with_embeddings else None  # one client: shared cost and memo
    per_question, summary, gate = [], [], None
    chunk_counts = {}
    for cfg in configs(args.with_embeddings):
        # Default corpus is fixed to the agency pages so numbers stay comparable between runs.
        chunks = load_chunks(cfg["chunking"], collections=tuple(args.collections))
        chunk_counts[cfg["chunking"]] = len(chunks)
        retriever = build_retriever(chunks, cfg["retriever"], cfg["analyzer"], embedder)
        name = config_name(cfg)
        rows = [{"config": name, **score_question(retriever, q)} for q in questions]
        per_question += rows
        summary += summarise(name, rows)
        if cfg == {"retriever": "bm25", "analyzer": "stemmed", "chunking": "section"}:
            gate = gate_analysis(rows)

    write_csv(args.out / "retrieval_per_question.csv", per_question)
    write_csv(args.out / "retrieval_summary.csv", summary)
    (args.out / "abstention_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    corpus_names = {
        "agency": "data/docs/agency (16 fictional help pages, EN+AR)",
        "public": "data/docs/public (8 Wikivoyage/Wikipedia pages, CC BY-SA 4.0)",
    }
    run_info = {
        "date": date.today().isoformat(),
        "command": "python -m evals.retrieval_eval " + " ".join(sys.argv[1:]),
        "questions": len(questions),
        "answerable": sum(q["answerable"] for q in questions),
        "corpus": " + ".join(corpus_names.get(c, c) for c in args.collections),
        "chunks": chunk_counts,
        "embedding_model": embedder.model if embedder else None,
        "embedding_cost_usd": round(embedder.total_cost_usd, 6) if embedder else None,
        "embedding_tokens": embedder.total_tokens if embedder else None,
    }
    (args.out / "retrieval_run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    if not args.no_chart:
        plot_hit_at_5(summary, args.out / "retrieval_hit5.png")
        if args.with_embeddings:
            series = [
                ("BM25 (stemmed)", "bm25-stemmed, section"),
                (f"Embeddings ({embedder.model})", "embedding, section"),
                ("Hybrid (RRF)", "hybrid, section"),
            ]
            title = "Retrieval hit@5: keyword vs embedding vs hybrid"
            plot_hit_at_5(summary, args.out / "retrieval_hybrid_hit5.png", series=series, title=title)

    for subset in SUBSETS:
        print_table(summary, subset)
    print("\nBM25 score gate (bm25-stemmed, section):", json.dumps(gate, indent=2))
    print(f"\nWrote results to {_relative(args.out)}")


if __name__ == "__main__":
    main()
