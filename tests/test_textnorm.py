from bilingual_rag.textnorm import detect_language, light_stem_arabic, normalize, normalize_arabic, tokenize


def test_alef_forms_become_bare_alef():
    assert normalize_arabic("أإآٱ") == "اااا"


def test_ya_and_ta_marbuta():
    assert normalize_arabic("مستشفى") == "مستشفي"
    assert normalize_arabic("تذكرة") == "تذكره"


def test_diacritics_and_tatweel_removed():
    assert normalize_arabic("قانونيّة") == "قانونيه"
    assert normalize_arabic("مـــرحبا") == "مرحبا"
    assert normalize_arabic("رَحْلَة") == "رحله"


def test_digits_and_case():
    assert normalize("٥٠٠٠ AED") == "5000 aed"


def test_diacritics_do_not_split_words():
    # Without normalising first, "يومَي" would be split into two tokens.
    assert tokenize("يومَي", "normalised") == ["يومي"]
    assert tokenize("يومَي", "plain") == ["يوم", "ي"]


def test_stopwords_removed_in_both_languages():
    assert tokenize("How can I change the flight?", "normalised") == ["change", "flight"]
    assert tokenize("هل يمكنني تغيير الرحلة؟", "normalised") == ["تغيير", "الرحله"]


def test_light_stemmer_strips_article_and_suffixes():
    assert light_stem_arabic(normalize("والأمتعة")) == "امتع"
    assert light_stem_arabic(normalize("بالبطاقات")) == "بطاق"
    # Short words are left alone so the root survives.
    assert light_stem_arabic("وزن") == "وزن"


def test_stemmed_forms_match_across_spellings():
    assert tokenize("الأمتعة", "stemmed") == tokenize("امتعه", "stemmed")
    assert tokenize("refunds", "stemmed") == tokenize("refund", "stemmed")


def test_detect_language():
    assert detect_language("كم يستغرق الاسترداد؟") == "ar"
    assert detect_language("How long does a refund take?") == "en"
    assert detect_language("ما الذي تغطيه خطة Plus؟") == "ar"  # code-switched, mostly Arabic
