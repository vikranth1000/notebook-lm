from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    source_path: str
    order: int


def chunk_text(text: str, source_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> Iterable[TextChunk]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: The text to chunk
        source_path: Path to the source document
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
    
    Yields:
        TextChunk objects
    """
    if not text.strip():
        return

    # Simple character-based chunking
    start = 0
    order = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary if possible
        if end < len(text):
            # Look for sentence endings in the last 100 chars
            for i in range(end - 100, end):
                if i < len(text) and text[i] in '.!?\n':
                    end = i + 1
                    break
        
        chunk_text_content = text[start:end].strip()
        
        if chunk_text_content:
            chunk_id = hashlib.md5(f"{source_path}:{order}:{chunk_text_content[:100]}".encode()).hexdigest()
            yield TextChunk(
                chunk_id=chunk_id,
                text=chunk_text_content,
                source_path=source_path,
                order=order,
            )
            order += 1
        
        # Move start position with overlap
        start = end - chunk_overlap
        if start >= len(text):
            break

