#!/usr/bin/env python3
"""
Performance Measurement Script - Baseline v1.0

Measures current performance metrics before implementing two-stage retrieval.
Run this script to establish baseline measurements.

Usage:
    python scripts/measure_baseline.py
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import Dict, List

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from notebooklm_backend.config import get_settings
from notebooklm_backend.services.embeddings import create_embedding_backend
from notebooklm_backend.services.vector_store import create_vector_store
from notebooklm_backend.services.rag import RAGService
from notebooklm_backend.services.llm import create_llm_backend
from notebooklm_backend.services.document_loader import load_document


class PerformanceMeasurer:
    def __init__(self):
        self.settings = get_settings()
        self.embedding_backend = create_embedding_backend(self.settings)
        self.vector_store = create_vector_store(self.settings, self.embedding_backend)
        self.rag_service = RAGService(self.settings, self.vector_store)
        
    async def measure_embedding_time(self, texts: List[str]) -> Dict:
        """Measure embedding generation time."""
        start = time.time()
        embeddings = self.embedding_backend.embed(texts)
        elapsed = time.time() - start
        
        return {
            "texts_count": len(texts),
            "total_time": elapsed,
            "time_per_text": elapsed / len(texts) if texts else 0,
            "texts_per_second": len(texts) / elapsed if elapsed > 0 else 0,
        }
    
    async def measure_ingestion_time(self, document_path: Path, notebook_id: str) -> Dict:
        """Measure document ingestion time."""
        start = time.time()
        
        # Load document
        load_start = time.time()
        document = load_document(document_path)
        load_time = time.time() - load_start
        
        # Chunk document
        chunk_start = time.time()
        from notebooklm_backend.services.chunking import chunk_text
        chunks = list(chunk_text(document.text, source_path=str(document_path)))
        chunk_time = time.time() - chunk_start
        
        # Generate embeddings
        embed_start = time.time()
        documents = [chunk.text for chunk in chunks]
        embeddings = self.embedding_backend.embed(documents)
        embed_time = time.time() - embed_start
        
        # Store in vector DB
        store_start = time.time()
        self.vector_store.add_chunks(notebook_id=notebook_id, chunks=chunks)
        store_time = time.time() - store_start
        
        total_time = time.time() - start
        
        return {
            "document_path": str(document_path),
            "text_length": len(document.text),
            "chunks_count": len(chunks),
            "load_time": load_time,
            "chunk_time": chunk_time,
            "embed_time": embed_time,
            "store_time": store_time,
            "total_time": total_time,
            "chunks_per_second": len(chunks) / embed_time if embed_time > 0 else 0,
        }
    
    async def measure_query_time(self, notebook_id: str, question: str) -> Dict:
        """Measure RAG query time."""
        start = time.time()
        
        # Query embedding
        embed_start = time.time()
        query_emb = self.embedding_backend.embed([question])
        embed_time = time.time() - embed_start
        
        # Vector search
        search_start = time.time()
        result = await self.rag_service.query(notebook_id=notebook_id, question=question, top_k=20)
        search_time = time.time() - search_start
        
        total_time = time.time() - start
        
        return {
            "question": question,
            "query_embed_time": embed_time,
            "search_time": search_time,
            "total_time": total_time,
            "answer_length": len(result.answer),
            "sources_count": len(result.sources),
        }
    
    def get_collection_stats(self, notebook_id: str) -> Dict:
        """Get statistics about a collection."""
        collection = self.vector_store.get_collection(notebook_id)
        count = collection.count()
        
        # Get all documents to count unique sources
        all_docs = collection.get(include=["metadatas"])
        metadatas = all_docs.get("metadatas", [])
        
        from collections import defaultdict
        from pathlib import Path
        doc_sources = defaultdict(int)
        for meta in metadatas:
            if isinstance(meta, dict):
                source_path = meta.get("source_path", "unknown")
                source_name = Path(source_path).name if source_path != "unknown" else "Document"
                doc_sources[source_name] += 1
        
        return {
            "total_chunks": count,
            "unique_documents": len(doc_sources),
            "documents": dict(doc_sources),
        }


async def main():
    print("=" * 60)
    print("Performance Measurement - Baseline v1.0")
    print("=" * 60)
    print()
    
    measurer = PerformanceMeasurer()
    
    # Test embedding performance
    print("1. Measuring Embedding Performance...")
    test_texts = [
        "This is a test document chunk.",
        "Another chunk of text for testing.",
        "Performance measurement is important.",
    ] * 10  # 30 texts total
    
    embed_results = await measurer.measure_embedding_time(test_texts)
    print(f"   Texts processed: {embed_results['texts_count']}")
    print(f"   Total time: {embed_results['total_time']:.3f}s")
    print(f"   Time per text: {embed_results['time_per_text']*1000:.2f}ms")
    print(f"   Texts per second: {embed_results['texts_per_second']:.1f}")
    print()
    
    # Test query performance (requires existing notebook)
    print("2. Collection Statistics...")
    print("   (Run this after ingesting documents)")
    print("   Example: notebook_id = 'test-notebook'")
    print()
    
    # Instructions
    print("=" * 60)
    print("To measure full performance:")
    print("1. Upload documents to create a notebook")
    print("2. Get the notebook_id from the UI")
    print("3. Run: python scripts/measure_baseline.py <notebook_id>")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        notebook_id = sys.argv[1]
        print(f"\nMeasuring query performance for notebook: {notebook_id}")
        
        # Get collection stats
        stats = measurer.get_collection_stats(notebook_id)
        print(f"\nCollection Statistics:")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Unique documents: {stats['unique_documents']}")
        print(f"  Documents: {stats['documents']}")
        
        # Measure query times
        test_queries = [
            "What does this document contain?",
            "Summarize the main points",
            "What is the key information?",
        ]
        
        print(f"\nQuery Performance (averaging {len(test_queries)} queries):")
        query_times = []
        for query in test_queries:
            result = await measurer.measure_query_time(notebook_id, query)
            query_times.append(result['total_time'])
            print(f"  Query: '{query[:50]}...'")
            print(f"    Total time: {result['total_time']:.3f}s")
            print(f"    Answer length: {result['answer_length']} chars")
            print(f"    Sources: {result['sources_count']}")
        
        avg_time = sum(query_times) / len(query_times)
        print(f"\n  Average query time: {avg_time:.3f}s")
        print(f"  Min: {min(query_times):.3f}s, Max: {max(query_times):.3f}s")


if __name__ == "__main__":
    asyncio.run(main())

