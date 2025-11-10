from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import chromadb
from chromadb import Collection

from ..config import AppConfig
from .chunking import TextChunk
from .embeddings import EmbeddingBackend


@dataclass
class VectorStoreManager:
    client: chromadb.PersistentClient
    embedding_backend: EmbeddingBackend

    def _collection_name(self, notebook_id: str) -> str:
        return f"notebook_{notebook_id}"

    def get_collection(self, notebook_id: str) -> Collection:
        return self.client.get_or_create_collection(name=self._collection_name(notebook_id))

    def add_chunks(self, notebook_id: str, chunks: Iterable[TextChunk]) -> int:
        chunk_list = list(chunks)
        if not chunk_list:
            return 0

        documents = [chunk.text for chunk in chunk_list]
        embeddings = self.embedding_backend.embed(documents)
        metadatas = [
            {
                "source_path": chunk.source_path,
                "order": chunk.order,
            }
            for chunk in chunk_list
        ]
        ids = [chunk.chunk_id for chunk in chunk_list]

        collection = self.get_collection(notebook_id)
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(chunk_list)

    def query(
        self,
        notebook_id: str,
        query: str,
        top_k: int = 5,
    ) -> dict:
        collection = self.get_collection(notebook_id)
        return collection.query(query_texts=[query], n_results=top_k)


def create_vector_store(settings: AppConfig, embedding_backend: EmbeddingBackend) -> VectorStoreManager:
    client = chromadb.PersistentClient(path=str(settings.index_dir))
    return VectorStoreManager(client=client, embedding_backend=embedding_backend)

