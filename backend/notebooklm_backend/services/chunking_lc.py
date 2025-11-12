from __future__ import annotations

from typing import Iterable

try:
    # Keep import local to avoid import-time cost if disabled
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception as e:  # pragma: no cover - handled at runtime
    RecursiveCharacterTextSplitter = None  # type: ignore

from .chunking import TextChunk  # reuse our TextChunk dataclass


def lc_split_text_to_chunks(
    text: str,
    source_path: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> Iterable[TextChunk]:
    """
    Split text using LangChain's RecursiveCharacterTextSplitter and map to our TextChunk type.
    Falls back by raising if LC is not available; the caller should decide the alternative.
    """
    if RecursiveCharacterTextSplitter is None:
        raise RuntimeError("langchain-text-splitters is not installed or failed to import.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    parts = splitter.split_text(text or "")

    order = 0
    from hashlib import md5
    for part in parts:
        content = part.strip()
        if not content:
            continue
        chunk_id = md5(f"{source_path}:{order}:{content[:100]}".encode()).hexdigest()
        yield TextChunk(chunk_id=chunk_id, text=content, source_path=source_path, order=order)
        order += 1

