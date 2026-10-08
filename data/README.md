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

## `docs/public/`: openly licensed pages (included, CC BY-SA 4.0, NOT MIT)

`python scripts/fetch_wikivoyage.py` downloaded these pages through the official MediaWiki API on **2026-10-08** (after waiting out HTTP 429 rate limits), and the result is committed as a fixed snapshot:

| File | Source | Revision |
|---|---|---|
| `en/dubai.md` | Wikivoyage "Dubai" | 5379926 |
| `en/abu-dhabi.md` | Wikivoyage "Abu Dhabi" | 5376885 |
| `en/united-arab-emirates.md` | Wikivoyage "United Arab Emirates" | 5380906 |
| `en/muscat.md` | Wikivoyage "Muscat" | 5354042 |
| `en/doha.md` | Wikivoyage "Doha" | 5375843 |
| `ar/dubai.md` | Arabic Wikipedia "دبي" | 76810043 |
| `ar/abu-dhabi.md` | Arabic Wikipedia "أبو ظبي" | 76835152 |
| `ar/muscat.md` | Arabic Wikipedia "مسقط" | 76817735 |

- **Licence:** Creative Commons Attribution-ShareAlike 4.0 (https://creativecommons.org/licenses/by-sa/4.0/). The folder has its own `LICENSE.md` and `ATTRIBUTION.md`, and every file keeps the source URL, revision id, retrieval date and attribution line in its front matter. These files are **not** covered by the repository's MIT licence.
- **Why commit them:** CC BY-SA allows redistribution with attribution under the same licence, and a fixed snapshot (with revision ids) makes the public-page evaluation reproducible. Re-running the script fetches newer revisions and can change those numbers.
- **Changes:** plain-text extract (the API drops tables and templates), split into numbered sections at level-2 headings, sections under 20 words dropped, at most 12 sections per page. No rewording.
- **Size:** about 55,600 words (209 section chunks), roughly 12 times the agency pages. Arabic comes from Wikipedia (encyclopedic), the script's original choice because Arabic Wikivoyage coverage looked thin (not re-checked on 2026-10-08), so the English and Arabic public pages are different articles and do not share section ids.
- **Use:** the demo app searches them by default. The main evaluation (`evals/questions.jsonl`) still uses only the agency pages so its numbers stay comparable; a separate run adds them (see `evals/questions_public.jsonl` below).

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

`../evals/questions_public.jsonl` — 10 extra questions (5 English, 5 Arabic; 5 same-language, 5 cross-language) whose answers are in the public pages, added on 2026-10-08. Same fields plus `"corpus": "public"`. They are reported separately and never mixed into the 60-question numbers. Written by the same coding agent that ran the evaluation, after reading the pages, so they share wording with them.
