from __future__ import annotations

from dataclasses import dataclass
from typing import List, TYPE_CHECKING
import logging

from ..config import AppConfig
from .vector_store import VectorStoreManager
from .rag import RAGResponse, SourceAttribution, RAGContext

if TYPE_CHECKING:
    from .rag import RAGService

logger = logging.getLogger(__name__)


class LlamaIndexRAGService:
    """
    LlamaIndex-based RAG that reuses our existing Chroma collections.
    Stays fully offline by using the local Ollama LLM.
    """

    def __init__(self, settings: AppConfig, vector_store: VectorStoreManager) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self._fallback_rag: "RAGService | None" = None

    async def prepare_prompt(self, notebook_id: str, question: str, top_k: int = 5) -> RAGContext:
        # Reuse the custom RAG selection/prompt logic to ensure consistent streaming behavior
        if self._fallback_rag:
            return await self._fallback_rag.prepare_prompt(
                notebook_id=notebook_id,
                question=question,
                top_k=top_k,
            )
        raise RuntimeError("prepare_prompt requires a fallback RAG service")

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
            
            # Skip test retrieval - it can cause ChromaDB errors with empty where clauses
            logger.info(f"Index created successfully")

            # Offline LLM via Ollama
            llm = Ollama(
                model=self.settings.ollama_model,
                base_url=self.settings.ollama_base_url,
                request_timeout=120.0,
            )

            # Two-stage retrieval: First filter documents by summary, then retrieve chunks
            # Stage 1: Query document summaries to find relevant documents
            relevant_summaries = self.vector_store.query_document_summaries(
                notebook_id=notebook_id,
                query=question,
                top_k=3,  # Get top 3 most relevant documents
            )
            
            # Determine which documents to search
            use_two_stage = False
            relevant_source_paths = set()
            if relevant_summaries:
                logger.info(f"Two-stage retrieval: Found {len(relevant_summaries)} relevant documents")
                # Normalize source paths for matching
                for summary in relevant_summaries:
                    # Try both full path and filename matching
                    relevant_source_paths.add(summary.source_path)
                    # Also add filename for matching
                    from pathlib import Path
                    relevant_source_paths.add(Path(summary.source_path).name)
                use_two_stage = True
            else:
                logger.info("No document summaries found, using single-stage retrieval")
                # Check if summaries collection exists but is empty (old documents)
                try:
                    summaries_collection = self.vector_store.client.get_collection(
                        name=self.vector_store._doc_summaries_collection_name(notebook_id)
                    )
                    if summaries_collection.count() == 0:
                        logger.info(
                            "Summaries collection exists but is empty - "
                            "documents uploaded before summary feature"
                        )
                except Exception:
                    logger.info("No summaries collection - documents uploaded before summary feature")
            
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
            
            # If no summaries exist, try keyword-based document matching as fallback
            if not use_two_stage and len(doc_sources) > 1:
                # Extract keywords from query that might indicate document type
                question_lower = question.lower()
                keyword_to_doc_type = {
                    "resume": ["resume", "cv", "curriculum vitae"],
                    "research_paper": ["research", "paper", "publication", "arxiv", "study"],
                }
                
                # Find documents that match keywords
                matching_docs = set()
                for doc_name in doc_sources.keys():
                    doc_name_lower = doc_name.lower()
                    for doc_type, keywords in keyword_to_doc_type.items():
                        if any(kw in question_lower for kw in keywords) and any(kw in doc_name_lower for kw in keywords):
                            matching_docs.add(doc_name)
                            logger.info(f"Keyword match: '{doc_name}' matches query keywords")
                
                # If we found matching documents, prioritize them
                if matching_docs:
                    logger.info(f"Using keyword-based filtering for {len(matching_docs)} documents: {matching_docs}")
                    # Filter source_groups to prioritize matching documents
                    prioritized_groups = defaultdict(list)
                    other_groups = defaultdict(list)
                    
                    for source_name, nodes in source_groups.items():
                        if source_name in matching_docs:
                            prioritized_groups[source_name] = nodes
                        else:
                            other_groups[source_name] = nodes
                    
                    # Rebuild retrieved_nodes with prioritized docs first
                    retrieved_nodes = []
                    for nodes in prioritized_groups.values():
                        retrieved_nodes.extend(nodes)
                    for nodes in other_groups.values():
                        retrieved_nodes.extend(nodes)
                    retrieved_nodes = retrieved_nodes[:retrieval_top_k]
                    
                    # Update source_groups for consistency
                    source_groups = {**prioritized_groups, **other_groups}
                    logger.info(f"Prioritized {len(prioritized_groups)} documents based on keywords")
            
            # If using two-stage retrieval, filter to only relevant documents
            if use_two_stage and relevant_source_paths:
                # Filter source_groups to only include relevant documents
                # Match by both full path and filename
                filtered_source_groups = defaultdict(list)
                for source_name, nodes in source_groups.items():
                    # Check if this document matches any of the relevant summaries
                    matches = False
                    for summary in relevant_summaries:
                        summary_source_name = Path(summary.source_path).name if summary.source_path != "unknown" else "Document"
                        # Match by filename
                        if source_name == summary_source_name:
                            matches = True
                            break
                        # Also check if source_name matches the full path (for absolute paths)
                        if summary.source_path in source_name or source_name in summary.source_path:
                            matches = True
                            break
                    
                    if matches:
                        filtered_source_groups[source_name] = nodes
                
                if filtered_source_groups:
                    logger.info(f"Two-stage filtering: Keeping {len(filtered_source_groups)} relevant documents: {list(filtered_source_groups.keys())}")
                    source_groups = filtered_source_groups
                    # Rebuild retrieved_nodes from filtered groups
                    retrieved_nodes = []
                    for nodes in source_groups.values():
                        retrieved_nodes.extend(nodes)
                    retrieved_nodes = retrieved_nodes[:retrieval_top_k]
                    logger.info(f"After filtering: {len(retrieved_nodes)} nodes from {len(source_groups)} documents")
                else:
                    logger.warning("Two-stage filtering removed all documents, falling back to original results")
                    use_two_stage = False
            
            # If we have multiple documents but retrieval missed some, do per-document retrieval
            # This ensures we get at least some chunks from each document
            missing_docs = set(doc_sources.keys()) - set(source_groups.keys())
            if len(doc_sources) > 1 and missing_docs and not use_two_stage:
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
            
            # Create query engine - use filtered nodes if two-stage retrieval was used
            if use_two_stage and retrieved_nodes:
                # Create a custom retriever that uses our filtered nodes
                from llama_index.core.schema import NodeWithScore
                from llama_index.core.retrievers import BaseRetriever
                
                class FilteredRetriever(BaseRetriever):
                    def __init__(self, nodes: list):
                        self.nodes = nodes
                    
                    def _retrieve(self, query_bundle):
                        return self.nodes
                
                filtered_retriever = FilteredRetriever(retrieved_nodes)
                query_engine = index.as_query_engine(
                    retriever=filtered_retriever,
                    llm=llm,
                    response_mode="compact",
                )
            else:
                # Use standard query engine
                query_engine = index.as_query_engine(
                    similarity_top_k=retrieval_top_k,
                    llm=llm,
                    response_mode="compact",
                )

            logger.info(f"Executing LlamaIndex query...")
            result = query_engine.query(question)
            
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
