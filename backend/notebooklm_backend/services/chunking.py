from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Iterable


_WHITESPACE_REGEX = re.compile(r"\s+")


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    source_path: str
    order: int


def normalize_text(text: str) -> str:
    return _WHITESPACE_REGEX.sub(" ", text).strip()


def chunk_text(
    text: str,
    source_path: str,
    chunk_size: int = 750,
    chunk_overlap: int = 150,
) -> Iterable[TextChunk]:
    """
    Splits text into overlapping chunks. Chunk size is expressed in tokens (approximated by words).
    """
    normalized = normalize_text(text)
    if not normalized:
        return []

    words = normalized.split(" ")
    if not words:
        return []

    step = max(1, chunk_size - chunk_overlap)
    chunks: list[TextChunk] = []

    for start in range(0, len(words), step):
        end = min(len(words), start + chunk_size)
        chunk_words = words[start:end]
        if not chunk_words:
            continue
        chunk_text_value = " ".join(chunk_words)
        chunk = TextChunk(
            chunk_id=uuid.uuid4().hex,
            text=chunk_text_value,
            source_path=source_path,
            order=len(chunks),
        )
        chunks.append(chunk)
        if end == len(words):
            break

    return chunks

