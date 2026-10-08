# LEARN: walkthrough, interview questions, live exercises

For Sara. Do the walkthrough out loud twice before an interview, then answer the ten questions without looking.

## 10-minute walkthrough script

**0:00 – The problem (1 min).** "Support teams in the Gulf get questions in Arabic, English or both. A chatbot that answers from its own memory is dangerous for refunds and visas. I built an assistant that only answers from the company's help pages, cites the page for each fact, and says 'I don't know' otherwise, and I measured how well it retrieves."

**1:00 – The data (1 min).** Open `data/docs/agency/en/refunds.md` and `ar/refunds.md`. "Sarab Travel is fictional. Each page has numbered sections; the English and Arabic versions share section numbers, so `refunds#2` is the same content in both. Insurance exists only in English and Umrah packages only in Arabic, on purpose, to test cross-language questions."

**2:00 – Arabic normalisation (2 min).** Open `textnorm.py`. Show `normalize_arabic`: diacritics, tatweel, أ/إ/آ → ا, ى → ي, ة → ه. "Why normalise before splitting? Python's `\w` does not include diacritics, so يومَي would become two tokens." Show `light_stem_arabic`: strip و, the article (ال, بال, ...), then suffixes, only if enough letters remain. Run `pytest tests/test_textnorm.py -q`.

**4:00 – Chunking and BM25 (2 min).** `ingest.py`: one chunk per section, whole sentences, one sentence of overlap. `retrieval.py`: walk through the BM25 formula — idf rewards rare words, k1 limits repetition, b penalises long chunks. Run `python -m bilingual_rag "حقيبتي لم تصل" --search-only`.

**6:00 – Answering and grounding (2 min).** `answer.py`: sources are numbered [S1]..[S5]; the rules say answer only from them, cite, reply in the question's language, or say NOT_IN_SOURCES. Then the code checks: no citation → the answer is blocked. "The prompt asks; the code enforces."

**8:00 – Results (2 min).** Open `evals/results/retrieval_hit5.png` and the README table. "Same-language hit@5 is 0.95 in English and 0.91 in Arabic; normalisation took Arabic from 0.76 to 0.91. Cross-language is 0 of 8 with BM25, which is why the next step is embeddings. A score threshold for 'I don't know' worked on dev but not on test, so I left it off." Finish with what is pending (live LLM run) and what you would do next.

## Ten interview questions with short answers

1. **Why start with BM25 instead of embeddings?**
   It needs no model download or API, it is fast, and it is a strong baseline for exact terms such as fees, "Lite fare" or "PIR". It also gives a number to beat: any embedding model must improve on 0.78 hit@5.

2. **What does Arabic do to keyword search?**
   The same word can be typed several ways (أ/ا, ة/ه, ى/ي, with or without diacritics), and words carry attached prefixes and suffixes (و، ال، ب، ها، ات). Without normalisation these never match. Here normalisation took Arabic same-language hit@5 from 0.76 to 0.91.

3. **What is light stemming and where does it fail?**
   It strips common prefixes and suffixes without finding the root. It fails on possessives (مشكلتي vs مشكلة), verb vs noun forms (أصعّد vs تصعيد) and broken plurals (شكوى vs شكاوى); the complaint question in the test set fails for exactly these reasons.

4. **How do you stop the model answering from its own memory?**
   Three layers: (1) the prompt says to use only the numbered sources and to reply NOT_IN_SOURCES otherwise; (2) the code blocks any answer without a valid [S#] citation; (3) the evaluation measures abstention on 10 questions the documents do not cover. A citation can still be wrong, so the judge also checks that the cited passage supports the claim.

5. **What happens when the question is Arabic but the answer is only in English?**
   BM25 cannot match across scripts: 0 of 8 such questions found the gold section. The prompt allows answering in Arabic from an English source, but retrieval must find it first. Fixes: multilingual embeddings, or translating the query before searching.

6. **Why section ids instead of chunk ids as gold labels?**
   Chunk ids change when the chunk size changes. Section ids like `refunds#2` stay the same, so the two chunking settings can be compared on the same labels, and an English chunk can count for an Arabic question.

7. **What are hit@k and MRR?**
   hit@k: share of questions with at least one correct chunk in the top k. MRR: the average of 1/rank of the first correct chunk (1 if it is first, 0.5 if second...), so it rewards putting the right chunk near the top.

8. **Why is the score threshold off?**
   It was chosen on the 20 dev questions, where it caught 4/4 unanswerable ones, but on the 40 test questions it caught only 2/6 and blocked 4/34 answerable ones. Picking a threshold on one set and checking on another exposed that it does not generalise.

9. **How would you trust an LLM judge?**
   Use a judge from a different model family, ask for a fixed JSON verdict, and compare it with my own labels on 20 answers (the run writes a review sheet). If we disagree often, the judge's numbers are not reported.

10. **What would you change for production?**
    Versioned documents and "which policy wins" rules, a larger question set written by people who have not seen the documents, multilingual embeddings for cross-language questions, monitoring of abstention rate and cost from the traces, and a human hand-off for booking-specific questions.

## Three "change it live" exercises

1. **Fix the possessive miss in the stemmer (10 min).**
   In `light_stem_arabic`, before the suffix loop, turn a final "تي" into "ه" (مشكلتي → مشكله). Add a test in `tests/test_textnorm.py`. Run `python -m evals.retrieval_eval` and check `ar-21` in `retrieval_per_question.csv`. Then say why you must also check that no other question got worse (you are changing the system to fit one test question).

2. **Make an unanswerable question answerable (10 min).**
   Add a section "## 6. Pets" to `data/docs/agency/en/baggage.md` and `ar/baggage.md` (same number in both). In `evals/questions.jsonl`, change `en-26` and `ar-26` to `"type": "same_lang"`, `"answerable": true` and `"gold_sections": ["baggage#6"]`. Run `pytest`: the dataset test will fail because there are now 8 unanswerable questions. Decide whether to update the test or write two new unanswerable questions, and explain the choice.

3. **Retrieve fewer chunks (5 min).**
   Run `python -m bilingual_rag "How long does a refund take?" -k 3 --search-only`, then change the default `k` in `AssistantConfig` from 5 to 3. Explain the trade-off: a shorter prompt (cheaper, less distraction) against a lower chance that the gold chunk is included (hit@3 0.76 vs hit@5 0.78 on all answerable questions).
