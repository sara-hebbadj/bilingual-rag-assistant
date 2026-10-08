"""End-to-end evaluation: retrieve, answer with a real model, score, and (optionally) judge.

    python -m evals.run --model cheap --limit 10     # real run, needs OPENROUTER_API_KEY
    python -m evals.run --model main                 # all 60 questions
    python -m evals.run --dry-run                    # offline, fake model, NOT real results
    python -m evals.run --retriever embedding --collections agency public \
        --questions evals/questions.jsonl evals/questions_public.jsonl   # with the CC BY-SA public pages

Real runs write to evals/results/ and evals/traces.jsonl.
Dry runs write to evals/dry_run/ so they can never be mistaken for real results.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from bilingual_rag.assistant import RETRIEVERS, AssistantConfig, build_assistant
from bilingual_rag.ingest import CHUNKING
from bilingual_rag.llm import FakeLLM, LLMClient, load_env
from bilingual_rag.metrics import abstention_correct, citation_precision, hit_at_k, language_matches, mean

from .judge import JUDGE_MAX_TOKENS, build_judge_messages, model_family, parse_verdict
from .retrieval_eval import QUESTIONS, load_question_files

EVAL_DIR = Path(__file__).resolve().parent


def make_clients(args) -> tuple:
    """Return (answer_llm, judge_llm or None, output_dir, trace_path)."""
    if args.dry_run:
        out = EVAL_DIR / "dry_run"
        trace = out / "traces.jsonl"
        trace.unlink(missing_ok=True)  # dry runs start fresh; only the latest one is kept
        return FakeLLM(trace_path=trace), (None if args.no_judge else FakeLLM(trace_path=trace)), out, trace
    out = EVAL_DIR / "results"
    trace = EVAL_DIR / "traces.jsonl"
    answer_llm = LLMClient.from_env(args.model, trace_path=trace)
    judge_llm = None if args.no_judge else LLMClient.from_env("judge", trace_path=trace)
    if judge_llm and model_family(judge_llm.model) == model_family(answer_llm.model):
        print(f"WARNING: judge {judge_llm.model} and answer model {answer_llm.model} are from the same family.")
    return answer_llm, judge_llm, out, trace


def evaluate_question(assistant, judge_llm, question: dict, run: str | None = None) -> dict:
    meta = {"question_id": question["id"], "run": run}  # written to every trace record
    cost_before = assistant.llm.total_cost_usd
    started = time.perf_counter()
    answer = assistant.ask(question["question"], meta=meta)
    latency_s = time.perf_counter() - started  # retrieval + model call (+ query embedding), not the judge
    gold = question["gold_sections"]
    cited_sections = [hit.chunk.section_id for hit in answer.cited_hits]
    retrieved_sections = [hit.chunk.section_id for hit in answer.sources]
    precision = citation_precision(cited_sections, gold) if question["answerable"] else None
    row = {
        "id": question["id"],
        "lang": question["lang"],
        "type": question["type"],
        "corpus": question.get("corpus", "agency"),
        "question": question["question"],
        "status": answer.status,
        "abstained": answer.abstained,
        "abstention_correct": abstention_correct(answer.abstained, question["answerable"]),
        "retrieval_hit": hit_at_k(retrieved_sections, gold, len(retrieved_sections)) if gold else None,
        "retrieved_sections": [f"{h.chunk.lang}:{h.chunk.section_id}" for h in answer.sources],
        "cited_sections": cited_sections,
        "cited_langs": [hit.chunk.lang for hit in answer.cited_hits],  # cross-language: did it use the other language?
        "invalid_citations": answer.invalid_citations,
        # Deterministic citation checks against the gold sections (None when nothing was cited):
        "citation_precision": precision,  # share of citations that point to a gold section
        "citation_all_gold": (precision == 1.0) if precision is not None else None,
        "language_match": language_matches(answer.text, question["lang"]) if not answer.abstained else None,
        "latency_s": round(latency_s, 3),
        "answer_cost_usd": round(assistant.llm.total_cost_usd - cost_before, 7),
        "answer": answer.text,
        "model_text": answer.model_text,
        "gold_answer": question["gold_answer"],
    }
    if judge_llm and question["answerable"] and not answer.abstained:
        cited = "\n\n".join(f"[S{n}] {answer.sources[n - 1].chunk.text}" for n in answer.cited)
        messages = build_judge_messages(question["question"], question["gold_answer"], answer.text, cited)
        judge_cost_before = judge_llm.total_cost_usd
        reply = judge_llm.complete(messages, purpose="judge", meta=meta, max_tokens=JUDGE_MAX_TOKENS)
        verdict = parse_verdict(reply.text)
        row.update({f"judge_{k}": v for k, v in verdict.items()})
        row["judge_cost_usd"] = round(judge_llm.total_cost_usd - judge_cost_before, 7)
    return row


def _rate(items: list[dict], test) -> str:
    return f"{sum(1 for r in items if test(r))}/{len(items)}" if items else "0/0"


def summarise(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    answered = [r for r in answerable if not r["abstained"]]
    judged = [r for r in answered if "judge_correctness" in r]
    rate = _rate

    return {
        "questions": len(rows),
        "correct_abstention_on_unanswerable": rate(unanswerable, lambda r: r["abstained"]),
        "false_abstention_on_answerable": rate(answerable, lambda r: r["abstained"]),
        "answered_with_gold_citation": rate(answered, lambda r: (r["citation_precision"] or 0) > 0),
        "answered_all_citations_gold": rate(answered, lambda r: r.get("citation_all_gold") is True),
        "mean_citation_precision": _round(mean([r["citation_precision"] for r in answered])),
        "answers_with_invalid_citation": rate(rows, lambda r: bool(r["invalid_citations"])),
        "language_match": rate(answered, lambda r: r["language_match"]),
        "judge_correct": rate(judged, lambda r: r["judge_correctness"] == "correct"),
        "judge_partial": rate(judged, lambda r: r["judge_correctness"] == "partial"),
        "judge_supported": rate(judged, lambda r: r["judge_supported"] == "yes"),
        "judge_unparsed": rate(judged, lambda r: r["judge_correctness"] == "unparsed"),
        # Strict end-to-end score: an abstention on an answerable question counts as not correct.
        "judge_correct_of_all_answerable": rate(answerable, lambda r: r.get("judge_correctness") == "correct"),
        "status_counts": {s: sum(1 for r in rows if r["status"] == s) for s in sorted({r["status"] for r in rows})},
        "latency_s": latency_stats([r["latency_s"] for r in rows if "latency_s" in r]),
        "by_group": by_group(rows),
    }


def latency_stats(values: list[float]) -> dict | None:
    """Mean, median and 95th percentile (nearest-rank) of the per-question latency."""
    if not values:
        return None
    ordered = sorted(values)
    p95 = ordered[max(0, round(0.95 * len(ordered)) - 1)]
    return {"mean": _round(statistics.mean(ordered)), "median": _round(statistics.median(ordered)), "p95": p95}


def by_group(rows: list[dict]) -> dict:
    """The main numbers per (corpus, question type, language), e.g. 'agency/cross_lang/ar'."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(f"{row.get('corpus', 'agency')}/{row['type']}/{row['lang']}", []).append(row)
    out = {}
    for name, items in sorted(groups.items()):
        answered = [r for r in items if not r["abstained"]]
        out[name] = {"n": len(items), "answered": len(answered)}
        if items[0]["type"] == "unanswerable":
            continue
        out[name].update(
            {
                "judge_correct": _rate(items, lambda r: r.get("judge_correctness") == "correct"),
                "answered_with_gold_citation": _rate(answered, lambda r: (r["citation_precision"] or 0) > 0),
                "language_match": _rate(answered, lambda r: r["language_match"]),
                # Cross-language: share of answers that cite at least one passage in the other language.
                "cites_other_language": _rate(answered, lambda r: any(lang != r["lang"] for lang in r["cited_langs"])),
            }
        )
    return out


def _round(value):
    return round(value, 3) if value is not None else None


def write_human_review(rows: list[dict], path: Path, n: int = 20, seed: int = 42) -> None:
    """A sheet for Sara: 20 random answerable questions with empty columns for her verdict."""
    candidates = [r for r in rows if r["type"] != "unanswerable"]
    sample = random.Random(seed).sample(candidates, min(n, len(candidates)))
    fields = [
        "id",
        "lang",
        "question",
        "gold_answer",
        "answer",
        "status",
        "judge_correctness",
        "sara_correctness",
        "sara_notes",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig so Excel shows Arabic correctly
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in sample:
            writer.writerow({**row, "sara_correctness": "", "sara_notes": ""})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["main", "cheap"], default="cheap")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--split", choices=["all", "dev", "test"], default="all")
    parser.add_argument("--retriever", choices=RETRIEVERS, default="bm25")
    parser.add_argument("--chunking", choices=list(CHUNKING), default="section")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="fake model, no network, writes to evals/dry_run/")
    parser.add_argument(
        "--collections", nargs="+", default=["agency"], help="document folders to search (default: agency only)"
    )
    parser.add_argument("--questions", nargs="+", type=Path, default=[QUESTIONS], help="question files (JSONL)")
    args = parser.parse_args()

    load_env()
    max_cost = float(os.getenv("MAX_COST_PER_RUN_USD") or 3)
    answer_llm, judge_llm, out_dir, trace_path = make_clients(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Default corpus is the agency pages only, so runs stay comparable.
    collections = tuple(args.collections)
    config = AssistantConfig(retriever=args.retriever, chunking=args.chunking, k=args.k, collections=collections)
    assistant = build_assistant(config, llm=answer_llm)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", answer_llm.model)
    corpus_tag = "" if collections == ("agency",) else "_" + "-".join(collections)
    name = "DRY_RUN_fake-llm" if args.dry_run else f"run_{stamp}_{slug}_{args.retriever}{corpus_tag}"

    questions = load_question_files(args.questions)
    questions = [q for q in questions if args.split == "all" or q["split"] == args.split][: args.limit]
    rows, stopped_early = [], False
    for number, question in enumerate(questions, start=1):
        rows.append(evaluate_question(assistant, judge_llm, question, run=name))
        spent = answer_llm.total_cost_usd + (judge_llm.total_cost_usd if judge_llm else 0)
        spent += assistant.embedder.total_cost_usd if assistant.embedder else 0
        print(f"[{number}/{len(questions)}] {question['id']} -> {rows[-1]['status']} (cost so far ${spent:.4f})")
        if spent > max_cost:
            print(f"Stopping: cost ${spent:.2f} passed MAX_COST_PER_RUN_USD=${max_cost}.")
            stopped_early = True
            break

    embedding_cost = assistant.embedder.total_cost_usd if assistant.embedder else 0.0
    answer_cost = answer_llm.total_cost_usd
    judge_cost = judge_llm.total_cost_usd if judge_llm else 0.0
    summary = {
        "run": name,
        "NOT_REAL_RESULTS": args.dry_run,  # True means the fake model wrote the answers
        "date_utc": stamp,
        "answer_model": answer_llm.model,
        "judge_model": judge_llm.model if judge_llm else None,
        "embedding_model": assistant.embedder.model if assistant.embedder else None,
        "retriever": args.retriever,
        "chunking": args.chunking,
        "k": args.k,
        "collections": list(collections),
        "question_files": [_relative(p) for p in args.questions],
        "split": args.split,
        "limit": args.limit,
        "stopped_early_for_cost": stopped_early,
        "cost_usd": round(answer_cost + judge_cost + embedding_cost, 4),
        "cost_breakdown_usd": {
            "answers": round(answer_cost, 5),
            "judge": round(judge_cost, 5),
            "embeddings": round(embedding_cost, 6),
            "answers_per_question": round(answer_cost / len(rows), 6) if rows else None,
        },
        **summarise(rows),
    }
    with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_human_review(rows, out_dir / f"{name}_human_review.csv")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    shown = (out_dir / name).relative_to(EVAL_DIR.parent).as_posix()
    print(f"\nWrote {shown}.jsonl, _summary.json and _human_review.csv; traces in {trace_path.name}")


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(EVAL_DIR.parent).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
