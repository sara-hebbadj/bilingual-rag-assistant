"""Download a few openly licensed travel pages (Wikivoyage / Wikipedia, CC BY-SA 4.0).

It was run on 2026-10-08 and its output is committed in data/docs/public/ (a fixed
snapshot: each file records the revision id it came from). Re-running it fetches the
current revisions, which can change the public-page evaluation numbers.

    python scripts/fetch_wikivoyage.py
    python scripts/fetch_wikivoyage.py --contact "you@example.com"   # recommended

It uses the official MediaWiki API (no HTML scraping), waits between requests,
retries after HTTP 429 (rate limited) as the Retry-After header says, and sends a
descriptive User-Agent, as the Wikimedia API etiquette asks.
For every page it writes data/docs/public/<lang>/<slug>.md with the source URL,
revision id, licence and an attribution line, and it rewrites
data/docs/public/ATTRIBUTION.md and LICENSE.md. The app picks the pages up automatically;
the main evaluation keeps using only the fictional agency pages so its numbers stay comparable.

Licence reminder: the downloaded text stays under CC BY-SA 4.0 (not MIT).
If you publish it, keep LICENSE.md, ATTRIBUTION.md and the front matter of each file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "docs" / "public"
LICENSE = "CC BY-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"

# (language, site, page title, file slug). Edit this list to add destinations.
# Arabic pages come from Arabic Wikipedia because Arabic Wikivoyage coverage is thin.
PAGES = [
    ("en", "wikivoyage", "Dubai", "dubai"),
    ("en", "wikivoyage", "Abu Dhabi", "abu-dhabi"),
    ("en", "wikivoyage", "United Arab Emirates", "united-arab-emirates"),
    ("en", "wikivoyage", "Muscat", "muscat"),
    ("en", "wikivoyage", "Doha", "doha"),
    ("ar", "wikipedia", "دبي", "dubai"),
    ("ar", "wikipedia", "أبوظبي", "abu-dhabi"),
    ("ar", "wikipedia", "مسقط", "muscat"),
]

INTRO_HEADING = {"en": "Overview", "ar": "نظرة عامة"}
SITE_NAMES = {"wikivoyage": "Wikivoyage", "wikipedia": "Wikipedia"}
_HEADING = re.compile(r"^(=+)\s*(.*?)\s*=+\s*$")


def api_url(lang: str, site: str) -> str:
    return f"https://{lang}.{site}.org/w/api.php"


def fetch_page(lang: str, site: str, title: str, user_agent: str) -> dict:
    """Plain-text extract + revision id + canonical URL for one page."""
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
        "maxlag": "5",
        "prop": "extracts|revisions|info",
        "explaintext": "1",
        "exsectionformat": "wiki",
        "rvprop": "ids|timestamp",
        "inprop": "url",
        "titles": title,
    }
    request = urllib.request.Request(
        f"{api_url(lang, site)}?{urllib.parse.urlencode(params)}", headers={"User-Agent": user_agent}
    )
    with open_with_retry(request) as response:
        data = json.load(response)
    return parse_api_response(data)


def open_with_retry(request, attempts: int = 6, opener=urllib.request.urlopen, sleep=time.sleep):
    """Open a URL; on HTTP 429 (rate limited) wait for the server's Retry-After, then try again.

    Wikimedia answered 429 to our first requests on 2026-10-08 (shared network address);
    waiting the number of seconds in the Retry-After header was enough.
    """
    for attempt in range(1, attempts + 1):
        try:
            return opener(request, timeout=30)
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == attempts:
                raise
            retry_after = error.headers.get("Retry-After", "") if error.headers else ""
            wait = int(retry_after) if retry_after.isdigit() else 10
            print(f"rate limited (429), waiting {wait + 1}s (attempt {attempt}/{attempts})", file=sys.stderr)
            sleep(wait + 1)


def parse_api_response(data: dict) -> dict:
    page = data["query"]["pages"][0]
    if page.get("missing"):
        raise LookupError(f"page not found: {page.get('title')}")
    revision = page["revisions"][0]
    return {
        "title": page["title"],
        "extract": page.get("extract", ""),
        "revision_id": revision["revid"],
        "revision_time": revision["timestamp"],
        "url": page["fullurl"],
    }


def split_extract(extract: str, intro_heading: str, min_words: int = 20) -> list[tuple[str, str]]:
    """Turn '== Heading ==' text into (heading, body) pairs.

    Level-2 headings start a new section; deeper headings stay inside it as a
    plain line. Very short sections (e.g. empty "See also") are dropped.
    """
    sections: list[list[str]] = [[intro_heading, ""]]
    for line in extract.splitlines():
        match = _HEADING.match(line.strip())
        if match and len(match.group(1)) == 2:
            sections.append([match.group(2), ""])
        elif match:
            sections[-1][1] += match.group(2) + "\n"
        else:
            sections[-1][1] += line + "\n"
    cleaned = [(h, re.sub(r"\n{3,}", "\n\n", body).strip()) for h, body in sections]
    return [(h, body) for h, body in cleaned if len(body.split()) >= min_words]


def to_markdown(page: dict, lang: str, site: str, slug: str, sections: list[tuple[str, str]], retrieved: str) -> str:
    site_name = SITE_NAMES[site]
    attribution = (
        f'Text from the {site_name} article "{page["title"]}" by {site_name} contributors '
        f"({page['url']}?action=history), {LICENSE}. Converted to plain text and split into "
        f"numbered sections; long pages are cut after the first sections. No other changes."
    )
    header = {
        "doc_id": f"{site}-{slug}",
        "title": f"{page['title']} ({site_name})",
        "lang": lang,
        "version": page["revision_time"][:10],
        "source_url": page["url"],
        "revision_id": str(page["revision_id"]),
        "retrieved": retrieved,
        "license": f"{LICENSE} ({LICENSE_URL})",
        "attribution": attribution,
    }
    lines = ["---", *(f"{k}: {v}" for k, v in header.items()), "---", "", f"# {header['title']}", ""]
    for number, (heading, body) in enumerate(sections, start=1):
        lines += [f"## {number}. {heading}", "", body, ""]
    return "\n".join(lines)


def write_attribution(records: list[dict], out_dir: Path) -> None:
    lines = [
        "# Attribution for downloaded public pages",
        "",
        f"All files in this folder are adapted from Wikimedia projects under {LICENSE} ({LICENSE_URL}).",
        "They are NOT covered by the repository's MIT licence.",
        "",
    ]
    lines += [
        f"- `{r['file']}`: {r['attribution']} Revision {r['revision_id']}, retrieved {r['retrieved']}." for r in records
    ]
    (out_dir / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


LICENSE_NOTE = f"""# Licence of this folder: {LICENSE}

The pages in this folder (`en/`, `ar/`) are text from Wikivoyage and Wikipedia, written by
their contributors and released under the Creative Commons Attribution-ShareAlike 4.0
International licence: {LICENSE_URL} (legal code: https://creativecommons.org/licenses/by-sa/4.0/legalcode).

- They are **not** covered by the repository's MIT licence (that covers the code and the
  fictional agency pages only).
- Changes made: converted to plain text by the MediaWiki API (TextExtracts), split into
  numbered sections, sections under 20 words dropped, pages cut after the first sections.
  Nothing was reworded.
- Who wrote each page, its URL, revision id and retrieval date: see `ATTRIBUTION.md` and the
  front matter at the top of each file.
- If you share or adapt these files, keep this notice and `ATTRIBUTION.md`, credit the
  contributors, and share your version under the same licence.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--max-sections", type=int, default=12, help="keep only the first N sections per page")
    parser.add_argument("--contact", default="github.com/sara-hebbadj", help="contact shown in the User-Agent")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds to wait between requests")
    args = parser.parse_args()

    user_agent = f"bilingual-rag-assistant/0.1 (portfolio project; {args.contact}) python-urllib"
    retrieved = date.today().isoformat()
    records = []
    for lang, site, title, slug in PAGES:
        try:
            page = fetch_page(lang, site, title, user_agent)
        except Exception as error:  # keep going: one missing page should not stop the rest
            print(f"skip {lang}/{title}: {error}", file=sys.stderr)
            continue
        sections = split_extract(page["extract"], INTRO_HEADING[lang])[: args.max_sections]
        path = args.out / lang / f"{slug}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(to_markdown(page, lang, site, slug, sections, retrieved), encoding="utf-8")
        records.append(
            {
                "file": path.relative_to(args.out).as_posix(),
                "revision_id": page["revision_id"],
                "retrieved": retrieved,
                "attribution": f'"{page["title"]}" ({page["url"]}), {SITE_NAMES[site]} contributors, {LICENSE}.',
            }
        )
        print(f"saved {lang}/{slug}.md ({len(sections)} sections, {sum(len(b.split()) for _, b in sections)} words)")
        time.sleep(args.delay)
    if records:
        write_attribution(records, args.out)
        (args.out / "LICENSE.md").write_text(LICENSE_NOTE, encoding="utf-8")
    print(f"{len(records)}/{len(PAGES)} pages saved.")
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
