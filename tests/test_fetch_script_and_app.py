"""The Wikivoyage script's parsing (with a recorded API response) and the app handler."""

import importlib.util
from pathlib import Path

import pytest

from bilingual_rag.ingest import load_document

ROOT = Path(__file__).resolve().parents[1]

RECORDED_RESPONSE = {
    "query": {
        "pages": [
            {
                "title": "Dubai",
                "fullurl": "https://en.wikivoyage.org/wiki/Dubai",
                "revisions": [{"revid": 123456, "timestamp": "2026-09-30T10:00:00Z"}],
                "extract": (
                    "Dubai is a city in the United Arab Emirates with many hotels, malls and beaches and a busy airport "
                    "that connects travellers to the rest of the world every day of the year.\n\n\n"
                    "== Understand ==\nDubai grew quickly from a small trading port into a large city, and today it "
                    "is one of the most visited places in the Middle East for shopping, food and business trips.\n"
                    "=== Climate ===\nSummers are very hot and humid, while winters are mild and pleasant for "
                    "walking around the old souks and the creek area in the evening.\n\n"
                    "== See also ==\nShort.\n"
                ),
            }
        ]
    }
}


def load_script():
    spec = importlib.util.spec_from_file_location("fetch_wikivoyage", ROOT / "scripts" / "fetch_wikivoyage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wikivoyage_page_becomes_attributed_markdown(tmp_path):
    script = load_script()
    page = script.parse_api_response(RECORDED_RESPONSE)
    sections = script.split_extract(page["extract"], "Overview")
    assert [h for h, _ in sections] == ["Overview", "Understand"]  # "See also" is too short
    assert "Climate" in sections[1][1]  # subsection kept inside its parent
    markdown = script.to_markdown(page, "en", "wikivoyage", "dubai", sections, "2026-10-08")
    path = tmp_path / "dubai.md"
    path.write_text(markdown, encoding="utf-8")
    doc = load_document(path, "public")
    assert doc.meta["license"].startswith("CC BY-SA 4.0")
    assert "Wikivoyage contributors" in doc.meta["attribution"]
    assert doc.meta["revision_id"] == "123456"
    assert [s.number for s in doc.sections] == [1, 2]


def test_missing_page_raises():
    script = load_script()
    with pytest.raises(LookupError):
        script.parse_api_response({"query": {"pages": [{"title": "Nowhere", "missing": True}]}})


def test_app_search_mode_without_key():
    pytest.importorskip("gradio")
    from bilingual_rag.assistant import AssistantConfig, build_assistant

    spec = importlib.util.spec_from_file_location("demo_app", ROOT / "app" / "app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    respond = app.make_responder(build_assistant(AssistantConfig(collections=("agency",))))
    history, cleared = respond("power bank in checked baggage", [], app.SEARCH_MODE)
    assert cleared == "" and history[0]["role"] == "user"
    assert "Baggage" in history[1]["content"]
