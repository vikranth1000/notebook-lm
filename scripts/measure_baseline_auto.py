#!/usr/bin/env python3
"""
Automated Performance Measurement Script - Baseline v1.0

Automatically measures all baseline metrics and generates a report.
No manual intervention required - just run and get results.

Usage:
    python scripts/measure_baseline_auto.py
"""

import asyncio
import time
import sys
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from notebooklm_backend.config import get_settings
from notebooklm_backend.services.embeddings import create_embedding_backend
from notebooklm_backend.services.vector_store import create_vector_store
from notebooklm_backend.services.rag import RAGService
from notebooklm_backend.services.llm import create_llm_backend
from notebooklm_backend.services.document_loader import load_document
from notebooklm_backend.services.chunking import chunk_text


class AutomatedPerformanceMeasurer:
    def __init__(self):
        self.settings = get_settings()
        self.embedding_backend = create_embedding_backend(self.settings)
        self.vector_store = create_vector_store(self.settings, self.embedding_backend)
        self.rag_service = RAGService(self.settings, self.vector_store)
        self.results = {}
        
    def measure_embedding_performance(self) -> Dict:
        """Measure embedding generation performance."""
        print("📊 Measuring Embedding Performance...")
        
        # Test with different batch sizes
        test_cases = [
            (10, "Small batch"),
            (50, "Medium batch"),
            (100, "Large batch"),
        ]
        
        results = {}
        for batch_size, label in test_cases:
            test_texts = [f"This is test document chunk number {i}." for i in range(batch_size)]
            
            # Warm up
            _ = self.embedding_backend.embed(test_texts[:5])
            
            # Measure
            start = time.time()
            embeddings = self.embedding_backend.embed(test_texts)
            elapsed = time.time() - start
            
            results[label] = {
                "batch_size": batch_size,
                "total_time": elapsed,
                "time_per_text": elapsed / batch_size,
                "texts_per_second": batch_size / elapsed if elapsed > 0 else 0,
                "embedding_dim": len(embeddings[0]) if embeddings else 0,
            }
            print(f"   ✓ {label}: {results[label]['texts_per_second']:.1f} texts/sec")
        
        self.results['embedding'] = results
        return results
    
    def measure_chunking_performance(self) -> Dict:
        """Measure text chunking performance."""
        print("📊 Measuring Chunking Performance...")
        
        # Create test documents of different sizes
        test_docs = [
            (1000, "Small document (1K chars)"),
            (5000, "Medium document (5K chars)"),
            (20000, "Large document (20K chars)"),
        ]
        
        results = {}
        for doc_size, label in test_docs:
            test_text = "This is a test sentence. " * (doc_size // 25)
            
            start = time.time()
            chunks = list(chunk_text(test_text, source_path="test.txt"))
            elapsed = time.time() - start
            
            results[label] = {
                "doc_size": doc_size,
                "chunks_count": len(chunks),
                "chunking_time": elapsed,
                "time_per_chunk": elapsed / len(chunks) if chunks else 0,
                "chars_per_second": doc_size / elapsed if elapsed > 0 else 0,
            }
            print(f"   ✓ {label}: {results[label]['chunks_count']} chunks in {elapsed:.3f}s")
        
        self.results['chunking'] = results
        return results
    
    async def measure_ingestion_pipeline(self) -> Dict:
        """Measure full ingestion pipeline performance."""
        print("📊 Measuring Ingestion Pipeline...")
        
        # Create a test document
        test_text = """
        This is a test document for performance measurement.
        It contains multiple paragraphs and sentences.
        We will measure how long it takes to process this document.
        The document will be chunked, embedded, and stored.
        This simulates a real-world document ingestion scenario.
        """ * 50  # ~5K characters
        
        test_notebook_id = "baseline-test-notebook"
        
        # Clean up if exists
        try:
            collection = self.vector_store.get_collection(test_notebook_id)
            if collection.count() > 0:
                # Delete collection by recreating with same name (ChromaDB limitation)
                pass
        except:
            pass
        
        # Measure full pipeline
        start = time.time()
        
        # Step 1: Chunking
        chunk_start = time.time()
        chunks = list(chunk_text(test_text, source_path="test_document.txt"))
        chunk_time = time.time() - chunk_start
        
        # Step 2: Embedding
        embed_start = time.time()
        documents = [chunk.text for chunk in chunks]
        embeddings = self.embedding_backend.embed(documents)
        embed_time = time.time() - embed_start
        
        # Step 3: Storage
        store_start = time.time()
        self.vector_store.add_chunks(notebook_id=test_notebook_id, chunks=chunks)
        store_time = time.time() - store_start
        
        total_time = time.time() - start
        
        results = {
            "document_size": len(test_text),
            "chunks_count": len(chunks),
            "chunking_time": chunk_time,
            "embedding_time": embed_time,
            "storage_time": store_time,
            "total_time": total_time,
            "chunks_per_second": len(chunks) / embed_time if embed_time > 0 else 0,
        }
        
        print(f"   ✓ Processed {len(chunks)} chunks in {total_time:.3f}s")
        print(f"     - Chunking: {chunk_time:.3f}s")
        print(f"     - Embedding: {embed_time:.3f}s")
        print(f"     - Storage: {store_time:.3f}s")
        
        self.results['ingestion'] = results
        self.test_notebook_id = test_notebook_id
        return results
    
    async def measure_query_performance(self) -> Dict:
        """Measure RAG query performance."""
        print("📊 Measuring Query Performance...")
        
        if not hasattr(self, 'test_notebook_id'):
            print("   ⚠ Skipping - no test notebook available")
            return {}
        
        notebook_id = self.test_notebook_id
        
        # Get collection stats
        collection = self.vector_store.get_collection(notebook_id)
        total_chunks = collection.count()
        
        # Test queries of different types
        test_queries = [
            ("What does this document contain?", "General question"),
            ("Summarize the main points", "Summary request"),
            ("What is the key information?", "Information extraction"),
        ]
        
        query_results = []
        for query, query_type in test_queries:
            # Measure query time
            start = time.time()
            
            # Query embedding
            query_embed_start = time.time()
            query_emb = self.embedding_backend.embed([query])
            query_embed_time = time.time() - query_embed_start
            
            # Vector search
            search_start = time.time()
            result = await self.rag_service.query(notebook_id=notebook_id, question=query, top_k=20)
            search_time = time.time() - search_start
            
            total_time = time.time() - start
            
            query_results.append({
                "query": query,
                "query_type": query_type,
                "query_embed_time": query_embed_time,
                "search_time": search_time,
                "total_time": total_time,
                "answer_length": len(result.answer),
                "sources_count": len(result.sources),
            })
        
        avg_time = sum(r['total_time'] for r in query_results) / len(query_results)
        avg_search_time = sum(r['search_time'] for r in query_results) / len(query_results)
        
        results = {
            "collection_size": total_chunks,
            "queries_tested": len(test_queries),
            "average_query_time": avg_time,
            "average_search_time": avg_search_time,
            "query_details": query_results,
        }
        
        print(f"   ✓ Tested {len(test_queries)} queries")
        print(f"     - Average query time: {avg_time:.3f}s")
        print(f"     - Average search time: {avg_search_time:.3f}s")
        print(f"     - Collection size: {total_chunks} chunks")
        
        self.results['queries'] = results
        return results
    
    def measure_multi_document_scenario(self) -> Dict:
        """Measure performance with multiple documents."""
        print("📊 Measuring Multi-Document Scenario...")
        
        if not hasattr(self, 'test_notebook_id'):
            print("   ⚠ Skipping - no test notebook available")
            return {}
        
        notebook_id = self.test_notebook_id
        
        # Get collection stats
        collection = self.vector_store.get_collection(notebook_id)
        all_docs = collection.get(include=["metadatas"])
        metadatas = all_docs.get("metadatas", [])
        
        # Count documents
        doc_sources = defaultdict(int)
        for meta in metadatas:
            if isinstance(meta, dict):
                source_path = meta.get("source_path", "unknown")
                doc_sources[source_path] += 1
        
        results = {
            "total_chunks": collection.count(),
            "unique_documents": len(doc_sources),
            "documents": dict(doc_sources),
            "chunks_per_document_avg": collection.count() / len(doc_sources) if doc_sources else 0,
        }
        
        print(f"   ✓ Collection has {len(doc_sources)} documents")
        print(f"     - Total chunks: {collection.count()}")
        print(f"     - Avg chunks per doc: {results['chunks_per_document_avg']:.1f}")
        
        self.results['multi_document'] = results
        return results
    
    def get_system_info(self) -> Dict:
        """Get system configuration information."""
        print("📊 Collecting System Information...")
        
        info = {
            "embedding_backend": self.settings.embedding_backend,
            "embedding_model": self.settings.embedding_model,
            "llm_provider": self.settings.llm_provider,
            "ollama_model": self.settings.ollama_model,
            "use_langchain_splitter": self.settings.use_langchain_splitter,
            "use_llamaindex_rag": self.settings.use_llamaindex_rag,
            "llm_max_tokens": self.settings.llm_max_tokens,
            "llm_context_window": self.settings.llm_context_window,
        }
        
        print(f"   ✓ Embedding: {info['embedding_model']}")
        print(f"   ✓ LLM: {info['llm_provider']} ({info['ollama_model']})")
        print(f"   ✓ LangChain splitter: {info['use_langchain_splitter']}")
        print(f"   ✓ LlamaIndex RAG: {info['use_llamaindex_rag']}")
        
        self.results['system_info'] = info
        return info
    
    def generate_report(self) -> str:
        """Generate a formatted report."""
        report = []
        report.append("=" * 80)
        report.append("BASELINE PERFORMANCE MEASUREMENT REPORT - v1.0")
        report.append("=" * 80)
        report.append("")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # System Info
        if 'system_info' in self.results:
            report.append("SYSTEM CONFIGURATION")
            report.append("-" * 80)
            for key, value in self.results['system_info'].items():
                report.append(f"  {key}: {value}")
            report.append("")
        
        # Embedding Performance
        if 'embedding' in self.results:
            report.append("EMBEDDING PERFORMANCE")
            report.append("-" * 80)
            for label, metrics in self.results['embedding'].items():
                report.append(f"  {label}:")
                report.append(f"    Batch size: {metrics['batch_size']}")
                report.append(f"    Texts per second: {metrics['texts_per_second']:.1f}")
                report.append(f"    Time per text: {metrics['time_per_text']*1000:.2f}ms")
            report.append("")
        
        # Chunking Performance
        if 'chunking' in self.results:
            report.append("CHUNKING PERFORMANCE")
            report.append("-" * 80)
            for label, metrics in self.results['chunking'].items():
                report.append(f"  {label}:")
                report.append(f"    Chunks generated: {metrics['chunks_count']}")
                report.append(f"    Chunking time: {metrics['chunking_time']:.3f}s")
                report.append(f"    Chars per second: {metrics['chars_per_second']:.0f}")
            report.append("")
        
        # Ingestion Performance
        if 'ingestion' in self.results:
            report.append("INGESTION PIPELINE PERFORMANCE")
            report.append("-" * 80)
            ing = self.results['ingestion']
            report.append(f"  Document size: {ing['document_size']} characters")
            report.append(f"  Chunks generated: {ing['chunks_count']}")
            report.append(f"  Total time: {ing['total_time']:.3f}s")
            report.append(f"    - Chunking: {ing['chunking_time']:.3f}s")
            report.append(f"    - Embedding: {ing['embedding_time']:.3f}s ({ing['chunks_per_second']:.1f} chunks/sec)")
            report.append(f"    - Storage: {ing['storage_time']:.3f}s")
            report.append("")
        
        # Query Performance
        if 'queries' in self.results:
            report.append("QUERY PERFORMANCE")
            report.append("-" * 80)
            q = self.results['queries']
            report.append(f"  Collection size: {q['collection_size']} chunks")
            report.append(f"  Queries tested: {q['queries_tested']}")
            report.append(f"  Average query time: {q['average_query_time']:.3f}s")
            report.append(f"  Average search time: {q['average_search_time']:.3f}s")
            report.append("")
            report.append("  Query Details:")
            for detail in q['query_details']:
                report.append(f"    - {detail['query_type']}: {detail['total_time']:.3f}s ({detail['sources_count']} sources)")
            report.append("")
        
        # Multi-Document Stats
        if 'multi_document' in self.results:
            report.append("MULTI-DOCUMENT STATISTICS")
            report.append("-" * 80)
            md = self.results['multi_document']
            report.append(f"  Total chunks: {md['total_chunks']}")
            report.append(f"  Unique documents: {md['unique_documents']}")
            report.append(f"  Average chunks per document: {md['chunks_per_document_avg']:.1f}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    async def run_all_measurements(self):
        """Run all measurements automatically."""
        print("\n" + "=" * 80)
        print("AUTOMATED BASELINE PERFORMANCE MEASUREMENT")
        print("=" * 80 + "\n")
        
        try:
            # System info
            self.get_system_info()
            print()
            
            # Embedding performance
            self.measure_embedding_performance()
            print()
            
            # Chunking performance
            self.measure_chunking_performance()
            print()
            
            # Ingestion pipeline
            await self.measure_ingestion_pipeline()
            print()
            
            # Query performance (skip if Ollama not available)
            try:
                await self.measure_query_performance()
                print()
            except Exception as e:
                print(f"   ⚠ Skipping query performance - Ollama not available: {e}")
                print("   (Start Ollama to measure query performance)")
                print()
            
            # Multi-document stats
            self.measure_multi_document_scenario()
            print()
            
            # Generate report
            report = self.generate_report()
            print("\n" + report)
            
            # Save to file
            report_file = Path(__file__).parent.parent / "docs" / "baseline_metrics_report.txt"
            report_file.write_text(report)
            print(f"\n✓ Report saved to: {report_file}")
            
            # Save JSON for programmatic access
            json_file = Path(__file__).parent.parent / "docs" / "baseline_metrics.json"
            json_file.write_text(json.dumps(self.results, indent=2))
            print(f"✓ JSON data saved to: {json_file}")
            
        except Exception as e:
            print(f"\n❌ Error during measurement: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True


async def main():
    measurer = AutomatedPerformanceMeasurer()
    success = await measurer.run_all_measurements()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())

