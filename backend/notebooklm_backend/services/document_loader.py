from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document as DocxDocument


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".pdf",
    ".docx",
}


@dataclass
class LoadedDocument:
    path: Path
    text: str


class DocumentLoaderError(RuntimeError):
    """Raised when a document cannot be loaded."""


def iter_supported_files(path: Path, recursive: bool = True) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
        return

    if not path.is_dir():
        raise DocumentLoaderError(f"Unsupported path type: {path}")

    iterator = path.rglob("*") if recursive else path.glob("*")
    for candidate in iterator:
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield candidate


def load_document(path: Path) -> LoadedDocument:
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md", ".markdown"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return LoadedDocument(path=path, text=text)

    if suffix == ".pdf":
        text = extract_pdf_text(str(path))
        return LoadedDocument(path=path, text=text)

    if suffix == ".docx":
        document = DocxDocument(str(path))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        return LoadedDocument(path=path, text=text)

    raise DocumentLoaderError(f"Unsupported file extension: {suffix}")

