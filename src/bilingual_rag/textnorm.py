"""Text normalisation and tokenisation for Arabic and English keyword search.

Arabic is written with optional short-vowel marks (diacritics), several forms
of alef, and letters that people type interchangeably (ى/ي, ة/ه). Keyword
search only matches identical strings, so we map all of these variants to one
form before indexing and before searching. The rules follow the widely used
Lucene ArabicNormalizer, plus a "light" stemmer based on Larkey et al. (Light10).
"""

from __future__ import annotations

import re
import unicodedata

import snowballstemmer

# Short vowels, tanween, shadda, sukun, superscript alef and Quranic marks.
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"  # the stretching character in "مـــرحبا"
_ALEF_FORMS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
_LETTER_FORMS = str.maketrans({"ى": "ي", "ة": "ه"})
# Arabic-Indic (٠-٩) and Persian (۰-۹) digits become 0-9 so "٢٤" matches "24".
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

_ARABIC_LETTER = re.compile(r"[ء-ي]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
# A token is a run of letters or digits. Underscore is excluded on purpose.
_TOKEN = re.compile(r"[^\W_]+")

# Light10 affixes, written in their normalised form (ة is already ه).
_AR_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")
_AR_SUFFIXES = ("ها", "ان", "ات", "ون", "ين", "يه", "ه", "ي")

_EN_STEMMER = snowballstemmer.stemmer("english")

# Small, hand-picked stopword lists: question words and function words that
# appear in almost every question and say nothing about the topic.
EN_STOPWORDS = frozenset(
    """a an the and or but if of to in on at by for from with about as into
    is are was were be been being am do does did doing have has had having
    i me my we our you your he she it its they them their this that these those
    what which who whom when where why how can could should would will shall may
    might must there here than then so not no any some all also just very
    please tell know want need""".split()
)

_AR_STOPWORDS_RAW = """في من على الى إلى عن مع هل ما ماذا كيف متى أين اين لماذا كم أي اي
هذا هذه ذلك تلك هناك هنا التي الذي الذين اللذين أن إن ان او أو ثم لا لم لن قد
كان كانت يكون تكون هو هي هم نحن انا أنا انت أنت انتم أنتم لي لك لنا لكم به بها
عند عندي عندما إذا اذا لو كل بعض غير بين بعد قبل حتى أيضا ايضا يمكن يمكنني
ممكن اريد أريد نريد ب و ف ل يا"""


def normalize_arabic(text: str) -> str:
    """Remove diacritics/tatweel and unify alef, ya and ta-marbuta forms."""
    text = _DIACRITICS.sub("", text)
    text = text.replace(_TATWEEL, "")
    text = text.translate(_ALEF_FORMS)
    return text.translate(_LETTER_FORMS)


def normalize(text: str) -> str:
    """Full normalisation used by search: Unicode NFKC, digits, case, Arabic forms."""
    # NFKC turns Arabic "presentation forms" (e.g. the ligature ﻻ) into normal letters.
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_DIGITS).lower()
    return normalize_arabic(text)


AR_STOPWORDS = frozenset(normalize(w) for w in _AR_STOPWORDS_RAW.split())
STOPWORDS = EN_STOPWORDS | AR_STOPWORDS


def is_arabic_word(token: str) -> bool:
    return bool(_ARABIC_LETTER.search(token))


def light_stem_arabic(word: str) -> str:
    """Light10-style stemmer: strip one 'and' prefix, one article, then suffixes.

    It never touches the root letters in the middle of the word, so it is
    "light". Each rule only fires if enough letters remain afterwards.
    """
    if word.startswith("و") and len(word) >= 4:
        word = word[1:]
    for prefix in _AR_PREFIXES:
        if word.startswith(prefix) and len(word) - len(prefix) >= 2:
            word = word[len(prefix) :]
            break
    # Words that begin with إل/أل/آل, like إلغاء ("cancellation"), look like they
    # start with the article once alef is normalised (الغاء), so the bare word
    # already lost "ال" above (-> غاء). Strip one more "ال" so the definite form
    # meets it: الإلغاء -> الالغاء -> الغاء -> غاء.
    if word.startswith("ال") and len(word) - 2 >= 2:
        word = word[2:]
    for suffix in _AR_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            word = word[: -len(suffix)]
    return word


def stem(token: str) -> str:
    if is_arabic_word(token):
        return light_stem_arabic(token)
    return _EN_STEMMER.stemWord(token)


# The three analyzers compared in the retrieval experiment.
ANALYZERS = ("plain", "normalised", "stemmed")


def tokenize(text: str, analyzer: str = "stemmed") -> list[str]:
    """Split text into search tokens.

    plain:      lowercase + split. No Arabic normalisation, no stopwords.
    normalised: + Arabic/Unicode normalisation + stopword removal.
    stemmed:    + light stemming (Light10 for Arabic, Snowball for English).
    """
    if analyzer not in ANALYZERS:
        raise ValueError(f"unknown analyzer {analyzer!r}; choose from {ANALYZERS}")
    if analyzer == "plain":
        return _TOKEN.findall(text.lower())
    # Normalise BEFORE splitting: diacritics are not "word" characters, so a
    # word with diacritics would otherwise be cut into pieces.
    tokens = [t for t in _TOKEN.findall(normalize(text)) if t not in STOPWORDS]
    if analyzer == "stemmed":
        tokens = [stem(t) for t in tokens]
    return tokens


def detect_language(text: str) -> str:
    """Return 'ar' if the text has more Arabic letters than Latin letters, else 'en'."""
    arabic = len(_ARABIC_LETTER.findall(text))
    latin = len(_LATIN_LETTER.findall(text))
    return "ar" if arabic > latin else "en"
