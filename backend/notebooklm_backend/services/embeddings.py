from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Protocol

from ..config import AppConfig

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore


class EmbeddingBackend(Protocol):
    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        ...


@dataclass
class SentenceTransformerBackend:
    model_name: str
    cache_dir: str
    _model: SentenceTransformer | None = None

    def _ensure_model(self) -> SentenceTransformer:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers package is not available.")

        if self._model is None:
            self._model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)  # type: ignore[arg-type]
        return self._model

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        model = self._ensure_model()
        return model.encode(list(texts), convert_to_numpy=False, show_progress_bar=False)  # type: ignore[no-any-return]


class HashEmbeddingBackend:
    """
    Deterministic lightweight embedding fallback that hashes text to a fixed-size vector.
    Intended for tests and environments without local models ready yet.
    """

    dimension: int = 768

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            checksum = hashlib.sha256(text.encode("utf-8")).digest()
            repeated = (checksum * ((self.dimension // len(checksum)) + 1))[: self.dimension]
            vector = [byte / 255.0 for byte in repeated]
            vectors.append(vector)
        return vectors


def create_embedding_backend(settings: AppConfig) -> EmbeddingBackend:
    if settings.embedding_backend == "hash":
        return HashEmbeddingBackend()

    return SentenceTransformerBackend(
        model_name=settings.embedding_model,
        cache_dir=str(settings.models_dir),
    )

