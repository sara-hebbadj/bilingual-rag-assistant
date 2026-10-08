# Retrieval results before the Arabic stemmer fix (2026-10-08): evidence only

These files are the retrieval evaluation as first run on 2026-10-08, kept so the "before"
numbers in the main README can be checked. The current results are one folder up.

What changed: the light Arabic stemmer turned إلغاء ("cancellation") into غاء but الإلغاء
("the cancellation") into الغاء, so the two forms never matched. `light_stem_arabic` in
`src/bilingual_rag/textnorm.py` now strips a second "ال" after a prefix, and
`tests/test_textnorm.py` checks that both forms give the same token. Same commands, same
questions, same documents; only the BM25 "stemmed" and the hybrid rows changed.

- `retrieval_*`: `python -m evals.retrieval_eval --with-embeddings` (agency pages).
- `agency_plus_public/retrieval_*`: the same with `--collections agency public --questions
  evals/questions.jsonl evals/questions_public.jsonl`.
