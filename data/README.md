# Data

## `docs/agency/`: fictional help pages (included)

- **Company:** "Sarab Travel" (سراب للسفر — *sarab* means "mirage"), a **fictional** Dubai travel agency. It is not a real company, and none of these policies are real travel, airline, visa or insurance rules.
- **Written for this repo** on 2026-10-08 by a coding agent, from a brief by Sara Hebbadj. Released under the repository's MIT licence.
- **Contact details** are placeholders: `@example.com` addresses and `+971 50 000 xxxx` numbers.
- **Languages:** 7 topics in English and Arabic (booking changes, cancellations, refunds, baggage, visas: general information, payments, contact and support), plus:
  - `en/travel-insurance.md` — English only, on purpose, to test Arabic questions whose answer exists only in English;
  - `ar/umrah-packages.md` — Arabic only, on purpose, to test the reverse.
- The Arabic pages are written as natural Modern Standard Arabic, not word-for-word translations. **To review:** Sara checks the Arabic wording.

### File format

```markdown
---
doc_id: refunds            # same id in en/ and ar/
title: Refunds
lang: en
version: 2026-10-01
notice: Fictional demo document ...
---

# Refunds

## 2. How long a refund takes      <- section number = part of the section id "refunds#2"
...
```

The English and Arabic versions of a topic use the **same section numbers**, so `refunds#2` names the same content in both languages. The evaluation's gold labels use these ids.

## `docs/public/`: openly licensed pages (not included, optional)

`python scripts/fetch_wikivoyage.py` downloads a few Wikivoyage (English) and Wikipedia (Arabic) pages about Dubai, Abu Dhabi, the UAE, Muscat and Doha through the official MediaWiki API, and writes:

- `docs/public/<lang>/<slug>.md`, with the source URL, revision id, retrieval date, licence and an attribution line in the front matter;
- `docs/public/ATTRIBUTION.md`, listing every page.

These pages are under **CC BY-SA 4.0** (https://creativecommons.org/licenses/by-sa/4.0/), **not** MIT. If you publish them, keep the attribution files and keep them under CC BY-SA. The script was written but not run on the build machine (Wikimedia sites were blocked there). The app includes them automatically when present; the evaluation does not, so its numbers stay comparable.

## `index/` (created on demand, not committed)

Cached chunk embeddings for the optional embedding/hybrid retriever, one file per embedding model.

## Questions

`../evals/questions.jsonl` — 60 questions (30 English, 30 Arabic), written for this repo:

| Field | Meaning |
|---|---|
| `type` | `same_lang` (42), `cross_lang` (8: answer only in the other language), `unanswerable` (10) |
| `gold_sections` | section ids that contain the answer (empty for unanswerable) |
| `gold_answer` | short reference answer, used by the judge model |
| `split` | `dev` (every third question, 20) is only for choosing thresholds; `test` (40) is for reporting |
