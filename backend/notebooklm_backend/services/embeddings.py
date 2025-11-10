from __future__ import annotations

from typing import Protocol

from ..config import AppConfig


class EmbeddingBackend(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        ...


class SentenceTransformersBackend:
    """Embedding backend using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
    
    def _ensure_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
        return self._model
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using sentence-transformers."""
        model = self._ensure_model()
        embeddings = model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()


class HashEmbeddingBackend:
    """Simple hash-based embedding for testing (not semantic)."""
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate pseudo-embeddings using hashing."""
        import hashlib
        
        embeddings = []
        for text in texts:
            # Create a 768-dim vector from hash
            hash_obj = hashlib.sha256(text.encode())
            hash_bytes = hash_obj.digest()
            
            # Repeat hash to get 768 dimensions
            vector = []
            for i in range(768):
                byte_idx = i % len(hash_bytes)
                # Normalize to [-1, 1]
                val = (hash_bytes[byte_idx] / 255.0) * 2 - 1
                vector.append(val)
            
            embeddings.append(vector)
        
        return embeddings


def create_embedding_backend(settings: AppConfig) -> EmbeddingBackend:
    """Create an embedding backend based on settings."""
    if settings.embedding_backend == "sentence-transformers":
        return SentenceTransformersBackend(model_name=settings.embedding_model)
    elif settings.embedding_backend == "hash":
        return HashEmbeddingBackend()
    else:
        raise ValueError(f"Unknown embedding backend: {settings.embedding_backend}")

