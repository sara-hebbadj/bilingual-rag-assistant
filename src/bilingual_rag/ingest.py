"""Load the Markdown help pages and split them into chunks.

Each document is a Markdown file with a small front-matter block (doc_id,
title, lang, ...) and numbered sections ("## 2. Refund timelines"). The
section number gives every section a language-independent id such as
"refunds#2": the English and Arabic versions of the same section share it.
The evaluation labels its gold passages with these ids, so the labels stay
valid whatever chunk size we choose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "data" / "docs"

# Two chunking settings compared in the evaluation. Sizes are in words
# (an English word is ~1.3 model tokens; an Arabic word is often 2-3 tokens).
CHUNKING = {
    "section": {"max_words": 350, "overlap_sentences": 1},  # one chunk per section (all sections fit)
    "small": {"max_words": 30, "overlap_sentences": 1},  # about 1-2 sentences per chunk
}

_SECTION_HEADING = re.compile(r"^##\s+(\d+)\.\s+(.+?)\s*$")
# Split after . ! ? or the Arabic question mark, when followed by a space.
_SENTENCE_END = re.compile(r"(?<=[.!?؟])\s+")


@dataclass
class Section:
    number: int
    heading: str
    body: str


@dataclass
class Document:
    doc_id: str
    title: str
    lang: str
    collection: str  # "agency" (fictional help pages) or "public" (Wikivoyage/Wikipedia)
    path: str  # path relative to the repo, shown as the source
    sections: list[Section] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # unique, e.g. "agency/ar/refunds#2.1"
    doc_id: str
    section_id: str  # language-independent, e.g. "refunds#2"
    lang: str
    title: str
    section: str
    text: str
    path: str
    collection: str
    url: str = ""

    @property
    def search_text(self) -> str:
        """Text that is indexed: title and heading are added so they can match."""
        return f"{self.title}. {self.section}. {self.text}"

    @property
    def label(self) -> str:
        """Human-readable source name used in citations."""
        return f"{self.title} › {self.section} ({self.lang.upper()})"


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Split '---' front matter (simple 'key: value' lines) from the body."""
    if not raw.startswith("---"):
        return {}, raw
    _, header, body = raw.split("---", 2)
    meta = {}
    for line in header.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, body


def parse_sections(body: str) -> list[Section]:
    """Collect the text under each '## N. Heading'. Text before the first one is ignored."""
    sections: list[Section] = []
    for line in body.splitlines():
        match = _SECTION_HEADING.match(line)
        if match:
            sections.append(Section(int(match.group(1)), match.group(2), ""))
        elif sections:
            sections[-1].body += line + "\n"
    for section in sections:
        section.body = section.body.strip()
    return sections


def load_document(path: Path, collection: str) -> Document:
    meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
    try:
        relative = path.relative_to(REPO_ROOT).as_posix()
    except ValueError:  # a file outside the repo (used in tests)
        relative = path.name
    return Document(
        doc_id=meta.get("doc_id", path.stem),
        title=meta.get("title", path.stem),
        lang=meta.get("lang", "en"),
        collection=collection,
        path=relative,
        sections=parse_sections(body),
        meta=meta,
    )


def load_documents(docs_dir: Path = DOCS_DIR, collections: tuple[str, ...] = ("agency",)) -> list[Document]:
    """Load every .md file under docs_dir/<collection>/<lang>/. Missing collections are skipped."""
    documents = []
    for collection in collections:
        folder = docs_dir / collection
        if not folder.exists():
            continue  # e.g. public pages not fetched yet
        for path in sorted(folder.glob("*/*.md")):
            documents.append(load_document(path, collection))
    return documents


def split_units(body: str) -> list[str]:
    """Split a section into sentences; each bullet line counts as one unit."""
    units = []
    for line in body.splitlines():
        line = line.strip()
        if line:
            units.extend(u.strip() for u in _SENTENCE_END.split(line) if u.strip())
    return units


def pack_units(units: list[str], max_words: int, overlap_sentences: int) -> list[str]:
    """Greedily pack whole sentences into chunks of at most max_words words.

    The last `overlap_sentences` sentences of a chunk are repeated at the start
    of the next one, so a fact that sits on a boundary is still found. A single
    sentence longer than max_words becomes its own chunk (it is not cut).
    """
    chunks: list[list[str]] = []
    current: list[str] = []
    for unit in units:
        words_now = sum(len(u.split()) for u in current)
        if current and words_now + len(unit.split()) > max_words:
            chunks.append(current)
            current = current[-overlap_sentences:] if overlap_sentences else []
            # Drop the overlap if it would not leave room for the new sentence.
            if sum(len(u.split()) for u in current) + len(unit.split()) > max_words:
                current = []
        current.append(unit)
    if current:
        chunks.append(current)
    return ["\n".join(chunk) for chunk in chunks]


def chunk_documents(documents: list[Document], chunking: str = "section") -> list[Chunk]:
    settings = CHUNKING[chunking]
    chunks = []
    for doc in documents:
        for section in doc.sections:
            section_id = f"{doc.doc_id}#{section.number}"
            pieces = pack_units(split_units(section.body), **settings)
            for part, text in enumerate(pieces, start=1):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.collection}/{doc.lang}/{section_id}.{part}",
                        doc_id=doc.doc_id,
                        section_id=section_id,
                        lang=doc.lang,
                        title=doc.title,
                        section=section.heading,
                        text=text,
                        path=doc.path,
                        collection=doc.collection,
                        url=doc.meta.get("source_url", ""),
                    )
                )
    return chunks


def load_chunks(chunking: str = "section", collections: tuple[str, ...] = ("agency",)) -> list[Chunk]:
    """Convenience: load documents and chunk them in one call."""
    return chunk_documents(load_documents(DOCS_DIR, collections), chunking)
