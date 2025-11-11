from __future__ import annotations

from dataclasses import dataclass
from typing import List
import logging

from ..config import AppConfig
from .vector_store import VectorStoreManager
from .rag import RAGResponse, SourceAttribution

logger = logging.getLogger(__name__)


class LlamaIndexRAGService:
    """
    LlamaIndex-based RAG that reuses our existing Chroma collections.
    Stays fully offline by using the local Ollama LLM.
    """

    def __init__(self, settings: AppConfig, vector_store: VectorStoreManager) -> None:
        self.settings = settings
        self.vector_store = vector_store

    async def query(self, notebook_id: str, question: str, top_k: int = 10) -> RAGResponse:
        """
        Query using LlamaIndex, returning RAGResponse for compatibility with ChatService.
        """
        try:
            # Import lazily to avoid cost when disabled
            from llama_index.vector_stores.chroma import ChromaVectorStore
            from llama_index.core import VectorStoreIndex, Settings
            from llama_index.llms.ollama import Ollama

            logger.info(f"LlamaIndex RAG query for notebook {notebook_id[:8]}...")

            collection = self.vector_store.get_collection(notebook_id)
            
            # Check if collection has any documents
            collection_count = collection.count()
            if collection_count == 0:
                logger.warning(f"Collection {notebook_id} is empty")
                return RAGResponse(
                    answer="No documents found in this notebook. Please upload documents first.",
                    sources=[],
                )

            # Use sentence-transformers directly (we already have it installed)
            # This matches the embedding model used during ingestion
            # IMPORTANT: We need to use the same embedding model for queries that was used during ingestion
            try:
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                embedding_model = HuggingFaceEmbedding(model_name=self.settings.embedding_model)
            except ImportError:
                # Fallback: use sentence-transformers directly if HuggingFaceEmbedding not available
                logger.warning("HuggingFaceEmbedding not available, using sentence-transformers directly")
                from sentence_transformers import SentenceTransformer
                from llama_index.core.embeddings import BaseEmbedding
                
                class SentenceTransformerEmbedding(BaseEmbedding):
                    def __init__(self, model_name: str):
                        super().__init__()
                        self._model = SentenceTransformer(model_name)
                    
                    async def _aget_query_embedding(self, query: str) -> list[float]:
                        return self._model.encode(query).tolist()
                    
                    async def _aget_text_embedding(self, text: str) -> list[float]:
                        return self._model.encode(text).tolist()
                    
                    def _get_query_embedding(self, query: str) -> list[float]:
                        return self._model.encode(query).tolist()
                    
                    def _get_text_embedding(self, text: str) -> list[float]:
                        return self._model.encode(text).tolist()
                
                embedding_model = SentenceTransformerEmbedding(self.settings.embedding_model)
            
            # Set the embedding model globally for LlamaIndex
            Settings.embed_model = embedding_model

            # Create ChromaVectorStore - Chroma already has embeddings stored
            # Important: Make sure Chroma collection doesn't have an embedding function
            # (we store embeddings directly, so Chroma shouldn't try to embed)
            if hasattr(collection, 'metadata') and collection.metadata and collection.metadata.get('hnsw:space'):
                # Collection exists and has metadata
                pass
            
            li_store = ChromaVectorStore(
                chroma_collection=collection,
            )

            # Build an index from the existing vector store
            # ChromaVectorStore will use existing embeddings for similarity search
            # The embed_model is only used for encoding the query, not re-embedding documents
            index = VectorStoreIndex.from_vector_store(
                vector_store=li_store,
                embed_model=embedding_model,  # Only for query encoding
            )
            
            # Verify the index can retrieve documents
            logger.info(f"Index created, testing retrieval...")
            test_retriever = index.as_retriever(similarity_top_k=1)
            test_nodes = test_retriever.retrieve("test")
            logger.info(f"Test retrieval returned {len(test_nodes)} nodes")

            # Offline LLM via Ollama
            llm = Ollama(
                model=self.settings.ollama_model,
                base_url=self.settings.ollama_base_url,
                request_timeout=120.0,
            )

            # For multi-document scenarios, we need to ensure we retrieve chunks from ALL documents
            # First, get all unique documents in the collection
            all_docs = collection.get(include=["metadatas"])
            all_metadatas = all_docs.get("metadatas", [])
            
            # Count documents by source
            from collections import defaultdict
            from pathlib import Path
            doc_sources = defaultdict(int)
            for meta in all_metadatas:
                if isinstance(meta, dict):
                    source_path = meta.get("source_path", "unknown")
                    source_name = Path(source_path).name if source_path != "unknown" else "Document"
                    doc_sources[source_name] += 1
            
            logger.info(f"Collection contains {len(doc_sources)} documents: {dict(doc_sources)}")
            
            # Retrieve significantly more chunks to ensure we get content from all documents
            # Use a multiplier based on number of documents
            retrieval_top_k = max(top_k, 20, len(doc_sources) * 10)  # At least 10 chunks per document
            
            # For multi-document scenarios, ensure we retrieve chunks from ALL documents
            # This prevents bias towards documents with more chunks
            # Strategy: Retrieve top-k chunks, then ensure we have at least some chunks from each document
            
            # First, do a normal retrieval
            retriever = index.as_retriever(
                similarity_top_k=retrieval_top_k,
            )
            
            logger.info(f"Retrieving top {retrieval_top_k} chunks for query...")
            retrieved_nodes = retriever.retrieve(question)
            logger.info(f"Retrieved {len(retrieved_nodes)} nodes")
            
            # Group nodes by source to understand document distribution
            source_groups = defaultdict(list)
            for node_with_score in retrieved_nodes:
                node = node_with_score.node if hasattr(node_with_score, "node") else node_with_score
                meta = getattr(node, "metadata", {}) or {}
                source_path = str(meta.get("source_path", "unknown"))
                source_name = Path(source_path).name if source_path != "unknown" else "Document"
                source_groups[source_name].append(node_with_score)
            
            logger.info(f"Found content from {len(source_groups)} different documents: {list(source_groups.keys())}")
            
            # If we have multiple documents but retrieval missed some, do per-document retrieval
            # This ensures we get at least some chunks from each document
            missing_docs = set(doc_sources.keys()) - set(source_groups.keys())
            if len(doc_sources) > 1 and missing_docs:
                logger.info(f"Multi-document scenario detected. Missing chunks from: {missing_docs}")
                logger.info("Performing per-document retrieval to ensure balanced coverage...")
                
                # For each missing document, retrieve at least a few chunks
                chunks_per_doc = max(3, retrieval_top_k // len(doc_sources))  # At least 3 chunks per doc
                for missing_doc_name in missing_docs:
                    # Find the source path for this document
                    missing_source_path = None
                    for meta in all_metadatas:
                        if isinstance(meta, dict):
                            source_path = meta.get("source_path", "unknown")
                            source_name = Path(source_path).name if source_path != "unknown" else "Document"
                            if source_name == missing_doc_name:
                                missing_source_path = source_path
                                break
                    
                    if missing_source_path:
                        # Query specifically for this document by including its name in the query
                        doc_specific_query = f"{question} {missing_doc_name}"
                        doc_nodes = retriever.retrieve(doc_specific_query)
                        
                        # Filter to only nodes from this document
                        for node_with_score in doc_nodes:
                            node = node_with_score.node if hasattr(node_with_score, "node") else node_with_score
                            meta = getattr(node, "metadata", {}) or {}
                            node_source_path = str(meta.get("source_path", "unknown"))
                            node_source_name = Path(node_source_path).name if node_source_path != "unknown" else "Document"
                            
                            if node_source_name == missing_doc_name:
                                source_groups[missing_doc_name].append(node_with_score)
                                if len(source_groups[missing_doc_name]) >= chunks_per_doc:
                                    break
                        
                        logger.info(f"Retrieved {len(source_groups[missing_doc_name])} chunks from {missing_doc_name}")
                
                # Re-log the updated distribution
                logger.info(f"After per-document retrieval, found content from {len(source_groups)} documents: {list(source_groups.keys())}")
            
            # If we still have missing documents after per-document retrieval, 
            # fall back to custom RAG which handles multi-document scenarios better
            final_missing = set(doc_sources.keys()) - set(source_groups.keys())
            if len(doc_sources) > 1 and final_missing:
                logger.warning(f"Still missing chunks from documents: {final_missing} after per-document retrieval.")
                logger.warning(f"This might cause incomplete answers. Falling back to custom RAG for better multi-document handling.")
                if hasattr(self, "_fallback_rag"):
                    return await self._fallback_rag.query(notebook_id, question, top_k)
            
            # Enhance the question with multi-document awareness
            # Include ALL documents (not just retrieved ones) so LLM knows what's available
            all_doc_names = list(doc_sources.keys())
            enhanced_question = question
            if len(all_doc_names) > 1:
                doc_list = ", ".join(all_doc_names)
                enhanced_question = (
                    f"Context: The user has uploaded {len(all_doc_names)} documents: {doc_list}. "
                    f"Pay close attention to which document is most relevant to their question. "
                    f"If they mention a specific document type (e.g., 'resume', 'research paper', 'the other document', 'the resume'), "
                    f"you must identify and focus on content from that specific document type.\n\n"
                    f"User's question: {question}"
                )
                logger.info(f"Enhanced question for multi-document scenario with {len(all_doc_names)} documents: {all_doc_names}")
            
            # Create query engine - it will use the retrieved nodes (already grouped by source)
            query_engine = index.as_query_engine(
                similarity_top_k=retrieval_top_k,
                llm=llm,
                response_mode="compact",
            )
            
            logger.info(f"Executing LlamaIndex query with {collection_count} documents in collection...")
            result = query_engine.query(enhanced_question)

            # Extract answer
            answer = str(getattr(result, "response", result))
            logger.info(f"LlamaIndex query completed, answer length: {len(answer)}")

            # Extract sources if available
            sources: list[SourceAttribution] = []
            try:
                source_nodes = getattr(result, "source_nodes", []) or []
                logger.info(f"Found {len(source_nodes)} source nodes")
                for node_with_score in source_nodes:
                    node = node_with_score.node if hasattr(node_with_score, "node") else node_with_score
                    meta = getattr(node, "metadata", {}) or {}
                    text = getattr(node, "text", "") or str(node)
                    score = getattr(node_with_score, "score", None) if hasattr(node_with_score, "score") else None
                    
                    sources.append(
                        SourceAttribution(
                            source_path=str(meta.get("source_path", "unknown")),
                            content=text,
                            distance=float(score) if score is not None else None,
                        )
                    )
            except Exception as e:
                logger.warning(f"Error extracting sources: {e}")

            return RAGResponse(answer=answer, sources=sources)

        except ImportError as e:
            logger.error(f"LlamaIndex import failed: {e}")
            raise RuntimeError("LlamaIndex dependencies not installed. Run: pip install llama-index-core llama-index-vector-stores-chroma llama-index-llms-ollama") from e
        except Exception as e:
            logger.error(f"LlamaIndex RAG query failed: {e}", exc_info=True)
            # If we have a fallback RAG service, use it instead of raising
            if hasattr(self, "_fallback_rag"):
                logger.info("Falling back to custom RAG service")
                return await self._fallback_rag.query(notebook_id, question, top_k)
            raise


