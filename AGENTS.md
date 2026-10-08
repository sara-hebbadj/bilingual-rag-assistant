# Notes for coding agents (bilingual-rag-assistant)

Read the shared rules in `Portfolio Projects/AGENTS.md` first if you have that folder. This file adds what is specific to this repo.

## Commands

```bash
uv pip install -e ".[app,dev]"
pytest -q && ruff check . && ruff format --check .
python -m evals.retrieval_eval            # deterministic, no key; writes evals/results/
python -m evals.run --dry-run             # fake model; writes evals/dry_run/ (NOT results)
python -m evals.run --model cheap --limit 10   # real, needs OPENROUTER_API_KEY + MODEL_CHEAP + MODEL_JUDGE
python app/app.py
```

## Map

- `src/bilingual_rag/textnorm.py` — Arabic/English normalisation, stopwords, stemmers, `tokenize(text, analyzer)`.
- `src/bilingual_rag/ingest.py` — Markdown loading, sections, chunking presets `CHUNKING`.
- `src/bilingual_rag/retrieval.py` — `BM25Retriever`, `EmbeddingRetriever`, `HybridRetriever` (RRF).
- `src/bilingual_rag/answer.py` — prompt, citation parsing, abstention statuses.
- `src/bilingual_rag/assistant.py` — `AssistantConfig`, `build_assistant`.
- `src/bilingual_rag/llm.py` — the ONLY module that calls a model; `FakeLLM` for tests.
- `evals/retrieval_eval.py`, `evals/run.py`, `evals/judge.py` — evaluation.
- `scripts/fetch_wikivoyage.py` — optional CC BY-SA pages into `data/docs/public/`.

## Rules for this repo

- Tests never use the network. Use `FakeLLM(responses=[...])` or a stubbed client.
- Every model call goes through `llm.py` so it is traced (tokens, cost, latency) and never logs the key.
- Do not edit gold labels or questions to improve a metric. If a label is wrong, fix it in a separate change and say why.
- Keep the evaluation corpus fixed to `("agency",)`; fetched public pages change over time.
- Real numbers in README/RESULTS must come from saved files in `evals/results/`, with date, denominator, model ids and command. LLM-based metrics stay "pending live run" until a real run exists.
- Arabic documents: keep the same section numbers as the English version (gold ids depend on it). Arabic wording is reviewed by Sara.
- Keep functions short and explainable; Sara must be able to change any file in an interview.

## Known deviations from the original BUILD_SPEC

- Documents are a fictional travel agency's help pages (per the 8 October build instructions), not a company Travel & Expense handbook.
- Vector search uses numpy (equivalent to a FAISS flat index at this size) instead of FAISS/Chroma.
- No local embedding model: Hugging Face was blocked on the build machine, so embeddings go through an OpenAI-compatible API and BM25 is the default.
