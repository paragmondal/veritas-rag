"""Document ingestion module for Veritas.

Supports loading .txt, .md, .pdf, and .html/.htm files into structured RawDoc objects.
Normalizes whitespace while preserving essential paragraph structure.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union
from bs4 import BeautifulSoup
import pypdf

from src.config import settings


@dataclass
class RawDoc:
    """Structured representation of a document page or section."""
    doc_id: str
    source: str
    page: int
    text: str


def normalize_whitespace(text: str) -> str:
    """Normalize excessive whitespace while preserving paragraph breaks.

    Replaces Windows newlines, strips per-line padding, collapses multiple
    spaces/tabs into single space, and limits consecutive newlines to at most two.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    # Split into lines and strip each line
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    # Join with newlines
    joined = "\n".join(lines)
    # Collapse 3 or more newlines into 2 (paragraph break)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


def ingest_txt_or_md(file_path: Path) -> List[RawDoc]:
    """Ingest a text or markdown file.

    Supports synthetic page markers such as '--- [Page X] ---' or '-- Page X --'.
    If no page markers exist, treats the document as page 1.
    """
    content = file_path.read_text(encoding="utf-8", errors="replace")
    source_name = file_path.name

    # Check for page delimiter pattern: e.g. --- [Page 1] ---
    page_marker_pattern = re.compile(
        r"^---\s*\[Page\s+(\d+)\]\s*---", re.MULTILINE | re.IGNORECASE
    )
    splits = page_marker_pattern.split(content)

    docs: List[RawDoc] = []
    if len(splits) > 1:
        # splits pattern: [preamble, page_num_1, page_text_1, page_num_2, page_text_2, ...]
        preamble = splits[0].strip()
        if preamble:
            clean_text = normalize_whitespace(preamble)
            if clean_text:
                docs.append(
                    RawDoc(
                        doc_id=f"{source_name}_p1",
                        source=source_name,
                        page=1,
                        text=clean_text,
                    )
                )

        for i in range(1, len(splits), 2):
            page_num = int(splits[i])
            raw_page_text = splits[i + 1] if i + 1 < len(splits) else ""
            clean_text = normalize_whitespace(raw_page_text)
            if clean_text:
                docs.append(
                    RawDoc(
                        doc_id=f"{source_name}_p{page_num}",
                        source=source_name,
                        page=page_num,
                        text=clean_text,
                    )
                )
    else:
        clean_text = normalize_whitespace(content)
        if clean_text:
            docs.append(
                RawDoc(
                    doc_id=f"{source_name}_p1",
                    source=source_name,
                    page=1,
                    text=clean_text,
                )
            )

    return docs


def ingest_pdf(file_path: Path) -> List[RawDoc]:
    """Ingest a PDF file per page using pypdf."""
    docs: List[RawDoc] = []
    source_name = file_path.name
    reader = pypdf.PdfReader(str(file_path))

    for page_idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        clean_text = normalize_whitespace(raw_text)
        if clean_text:
            docs.append(
                RawDoc(
                    doc_id=f"{source_name}_p{page_idx}",
                    source=source_name,
                    page=page_idx,
                    text=clean_text,
                )
            )
    return docs


def ingest_html(file_path: Path) -> List[RawDoc]:
    """Ingest an HTML file using BeautifulSoup.

    Strips <script>, <style>, <nav>, <header>, <footer>, and extracts clean text.
    """
    raw_html = file_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw_html, "html.parser")

    # Remove non-content and layout tags
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n\n")
    clean_text = normalize_whitespace(text)
    source_name = file_path.name

    if not clean_text:
        return []

    return [
        RawDoc(
            doc_id=f"{source_name}_p1",
            source=source_name,
            page=1,
            text=clean_text,
        )
    ]


def load_file(file_path: Union[str, Path]) -> List[RawDoc]:
    """Load a single document based on its extension."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return ingest_txt_or_md(path)
    elif ext == ".pdf":
        return ingest_pdf(path)
    elif ext in (".html", ".htm"):
        return ingest_html(path)
    else:
        # Fallback to plain text reader
        return ingest_txt_or_md(path)


def load_directory(dir_path: Union[str, Path] = None) -> List[RawDoc]:
    """Load all supported documents from a directory."""
    if dir_path is None:
        target_dir = settings.get_raw_dir()
    else:
        target_dir = Path(dir_path)

    all_docs: List[RawDoc] = []
    supported_extensions = {".txt", ".md", ".pdf", ".html", ".htm"}

    for file_path in sorted(target_dir.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            docs = load_file(file_path)
            all_docs.extend(docs)

    return all_docs
