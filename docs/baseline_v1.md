# Baseline Documentation - Version 1.0
**Date:** 2024-12-19  
**Purpose:** Establish baseline metrics and architecture before implementing two-stage retrieval optimization

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Current Implementation Details](#current-implementation-details)
3. [Performance Metrics](#performance-metrics)
4. [Technology Stack](#technology-stack)
5. [Configuration](#configuration)
6. [Known Limitations](#known-limitations)
7. [Current Features](#current-features)

---

## System Architecture

### High-Level Overview
```
┌─────────────────────────────────────────────────────────────┐
│                    Electron Desktop App                     │
│  ┌──────────────────┐         ┌──────────────────┐       │
│  │  React Renderer  │         │  Electron Main   │       │
│  │  (Vite Dev)      │◄───────►│  Process         │       │
│  │                  │   IPC    │                  │       │
│  └────────┬─────────┘         └──────────────────┘       │
└───────────┼───────────────────────────────────────────────┘
            │ HTTP (localhost:8000)
            ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Backend (Python)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Chat Service │  │ Ingestion    │  │ RAG Service  │    │
│  │              │  │ Service      │  │             │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                  │                  │            │
│         └──────────────────┼──────────────────┘            │
│                            │                               │
│                    ┌───────▼────────┐                      │
│                    │ Vector Store   │                      │
│                    │ Manager        │                      │
│                    └───────┬────────┘                      │
└────────────────────────────┼───────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   ChromaDB   │    │   Ollama     │    │ Sentence     │
│  (Vector DB) │    │   (LLM)      │    │ Transformers │
│              │    │              │    │ (Embeddings) │
└──────────────┘    └──────────────┘    └──────────────┘
```

### Component Details

#### Frontend (Electron + React)
- **Framework:** Electron 31.7.7 + React 19.2.0
- **Build Tool:** Vite 7.2.2
- **Key Libraries:**
  - `react-markdown` for markdown rendering
  - `react-pdf` for PDF preview
  - `pdfjs-dist` for PDF.js worker
- **Communication:** HTTP REST API to FastAPI backend

#### Backend (FastAPI)
- **Framework:** FastAPI 0.115.6
- **Server:** Uvicorn with auto-reload
- **Language:** Python 3.10+
- **Key Services:**
  - `ChatService`: Handles chat interactions with/without RAG
  - `IngestionService`: Processes and chunks documents
  - `RAGService`: Custom RAG implementation
  - `LlamaIndexRAGService`: LlamaIndex-based RAG (with fallback)
  - `VectorStoreManager`: Manages ChromaDB collections

---

## Current Implementation Details

### Document Ingestion Pipeline

#### 1. Document Loading
- **Supported Formats:** PDF, DOCX, TXT, MD, PPTX, PY
- **Loaders:**
  - PDF: `pypdf` (text extraction)
  - DOCX: `python-docx` (paragraph extraction)
  - PPTX: `python-pptx` (slide text extraction)
  - TXT/MD/PY: Direct text reading

#### 2. Text Chunking
- **Default Strategy:** LangChain `RecursiveCharacterTextSplitter` (enabled by default)
- **Fallback:** Custom chunker if LangChain unavailable
- **Chunk Size:** 800 characters (LangChain) or 500 characters (custom)
- **Chunk Overlap:** 120 characters (LangChain) or 50 characters (custom)
- **Chunk Metadata:** `{source_path, order}`

#### 3. Embedding Generation
- **Model:** `all-MiniLM-L6-v2` (sentence-transformers)
- **Dimensions:** 384
- **Backend:** `SentenceTransformerBackend`
- **Process:** Batch embedding of all chunks

#### 4. Vector Storage
- **Database:** ChromaDB (PersistentClient)
- **Storage Location:** `~/NotebookLM/indexes/`
- **Collection Naming:** `notebook_{notebook_id}`
- **Stored Data:**
  - Document text (chunks)
  - Embeddings (384-dim vectors)
  - Metadata: `{source_path, order}`

### RAG Query Pipeline

#### Current Strategy: Single-Stage Retrieval

**Step 1: Query Embedding**
- User question → Embedding (384-dim vector)
- Uses same embedding model as ingestion

**Step 2: Similarity Search**
- Query embedding compared against all chunk embeddings in collection
- Retrieves top-k chunks (default: 20, scales with document count)
- **Formula:** `retrieval_top_k = max(top_k, 20, num_documents * 10)`

**Step 3: Source Grouping**
- Groups retrieved chunks by source document
- Builds context with source labels: `"From {filename}:"`

**Step 4: LLM Generation**
- Constructs prompt with:
  - System instructions (multi-document awareness)
  - Grouped document excerpts
  - User question
- Sends to Ollama LLM (`qwen2.5:3b`)
- Returns answer with source citations

#### RAG Implementations

**1. Custom RAG (`RAGService`)**
- Direct ChromaDB query
- Manual prompt construction
- Source grouping and citation
- **Default:** Used when LlamaIndex unavailable or fails

**2. LlamaIndex RAG (`LlamaIndexRAGService`)**
- Uses LlamaIndex framework with ChromaVectorStore
- Per-document retrieval fallback for multi-doc scenarios
- Automatic fallback to Custom RAG if retrieval fails
- **Default:** Enabled (`use_llamaindex_rag=True`)

**Multi-Document Handling:**
- Retrieves `max(20, num_documents * 10)` chunks
- Performs per-document retrieval if documents missing
- Falls back to Custom RAG if per-doc retrieval fails

### Chat Service

**Flow:**
1. Receives chat message with optional `notebook_id`
2. If `notebook_id` provided → Uses RAG
3. If no `notebook_id` → Regular chat (no RAG)
4. Combines recent chat history (last 3 messages) with current question for RAG

**RAG Integration:**
- Calls `rag_service.query(notebook_id, question, top_k=20)`
- Returns RAG answer directly
- Falls back to regular chat if RAG fails

---

## Performance Metrics

### Current Benchmarks (To Be Measured)

#### Ingestion Performance
- **Document Processing Time:** TBD (measure per document type)
- **Chunking Time:** TBD
- **Embedding Generation:** TBD (chunks/second)
- **Storage Time:** TBD

#### Query Performance
- **Query Embedding Time:** TBD
- **Vector Search Time:** TBD (varies with collection size)
- **LLM Generation Time:** TBD (varies with response length)
- **Total Query Latency:** TBD

#### Resource Usage
- **Memory:** TBD (embedding model + ChromaDB + Ollama)
- **CPU:** TBD
- **Disk:** TBD (ChromaDB indexes + document storage)

### Test Scenarios (To Be Measured)

#### Scenario 1: Single Document (Small)
- **Document:** Resume (8 chunks)
- **Query:** "What does this document contain?"
- **Expected:** Fast retrieval, accurate answer

#### Scenario 2: Single Document (Large)
- **Document:** Research paper (135 chunks)
- **Query:** "Summarize the main findings"
- **Expected:** Slower retrieval, comprehensive answer

#### Scenario 3: Multi-Document (Mixed Sizes)
- **Documents:** Research paper (135 chunks) + Resume (8 chunks)
- **Query:** "What does the resume contain?"
- **Expected:** Should retrieve from resume, may need per-doc retrieval

#### Scenario 4: Multi-Document (Many Documents)
- **Documents:** 5+ documents with varying chunk counts
- **Query:** General question
- **Expected:** Retrieves from multiple docs, may be slow

---

## Technology Stack

### Frontend
| Component | Version | Purpose |
|-----------|---------|---------|
| Electron | 31.7.7 | Desktop app framework |
| React | 19.2.0 | UI framework |
| Vite | 7.2.2 | Build tool & dev server |
| react-markdown | 9.0.1 | Markdown rendering |
| react-pdf | 10.2.0 | PDF preview |
| pdfjs-dist | 5.4.394 | PDF.js library |

### Backend
| Component | Version | Purpose |
|-----------|---------|---------|
| FastAPI | 0.115.6 | Web framework |
| Uvicorn | 0.32+ | ASGI server |
| ChromaDB | 0.5+ | Vector database |
| sentence-transformers | 2.7+ | Embedding generation |
| langchain-text-splitters | 0.2.0+ | Text chunking |
| llama-index-core | 0.11.0+ | RAG framework |
| llama-index-vector-stores-chroma | 0.2.0+ | Chroma integration |
| llama-index-llms-ollama | 0.3.0+ | Ollama LLM integration |
| pypdf | 5.0+ | PDF parsing |
| python-docx | 1.1+ | DOCX parsing |
| python-pptx | 0.6+ | PPTX parsing |

### External Services
| Service | Version/Model | Purpose |
|---------|---------------|---------|
| Ollama | Latest | LLM inference (qwen2.5:3b) |
| Sentence Transformers | all-MiniLM-L6-v2 | Embedding model |

---

## Configuration

### Default Settings (`AppConfig`)

```python
# Storage Directories
workspace_root: ~/NotebookLM
data_dir: ~/NotebookLM/data
models_dir: ~/NotebookLM/models
index_dir: ~/NotebookLM/indexes
cache_dir: ~/NotebookLM/cache

# Embedding Configuration
embedding_backend: "sentence-transformers"
embedding_model: "all-MiniLM-L6-v2"

# LLM Configuration
llm_provider: "ollama"
ollama_base_url: "http://127.0.0.1:11434"
ollama_model: "qwen2.5:3b"
llm_context_window: 2048
llm_max_tokens: 2048

# Framework Integration
use_langchain_splitter: True
use_llamaindex_rag: True
```

### Environment Variables
- `NOTEBOOKLM_*` prefix for all settings
- Example: `NOTEBOOKLM_OLLAMA_MODEL=qwen2.5:3b`

---

## Known Limitations

### Current Issues

1. **Multi-Document Bias**
   - Documents with more chunks dominate retrieval
   - Small documents (e.g., resume with 8 chunks) may be missed
   - Workaround: Per-document retrieval fallback

2. **No Document-Level Filtering**
   - All chunks searched regardless of document relevance
   - No pre-filtering by document type or content
   - Inefficient for notebooks with many documents

3. **Retrieval Scalability**
   - Retrieves `num_documents * 10` chunks minimum
   - With 10 documents → 100 chunks retrieved
   - All chunks sent to LLM (token limit concerns)

4. **No Document Summaries**
   - No document-level metadata or summaries stored
   - Cannot quickly identify document types
   - Must process all chunks to understand document content

5. **LlamaIndex Integration Issues**
   - Sometimes fails to retrieve from all documents
   - Requires fallback to Custom RAG
   - Per-document retrieval adds latency

6. **Resource Usage**
   - Embedding all chunks for every query
   - No caching of document summaries
   - Memory usage scales with number of chunks

### Performance Bottlenecks

1. **Vector Search:** O(n) where n = total chunks in collection
2. **LLM Context:** All retrieved chunks sent to LLM (token limit)
3. **No Caching:** Document summaries not cached
4. **Sequential Processing:** Per-document retrieval done sequentially

---

## Current Features

### ✅ Implemented Features

1. **Document Upload**
   - Support for PDF, DOCX, TXT, MD, PPTX, PY
   - File preview (PDF viewer with page navigation)
   - Document listing with chunk counts

2. **Document Ingestion**
   - Automatic chunking (LangChain or custom)
   - Embedding generation (sentence-transformers)
   - Vector storage (ChromaDB)

3. **RAG Query**
   - Single-stage retrieval
   - Multi-document support (with limitations)
   - Source citation
   - LlamaIndex integration with fallback

4. **Chat Interface**
   - RAG-enabled chat (with notebook_id)
   - Regular chat (without notebook_id)
   - Chat history context (last 3 messages)
   - Markdown rendering

5. **Document Preview**
   - PDF viewer with zoom and page navigation
   - Keyboard shortcuts (Arrow keys, Escape)
   - Full-screen modal

### ❌ Not Yet Implemented

1. **Document Summaries**
   - No document-level summaries stored
   - No quick document identification

2. **Two-Stage Retrieval**
   - No document-level filtering
   - No coarse-to-fine retrieval

3. **Performance Monitoring**
   - No metrics collection
   - No timing measurements

4. **Caching**
   - No document summary cache
   - No query result cache

---

## Data Flow Examples

### Ingestion Flow
```
User uploads PDF
  ↓
DocumentLoader.load_document() → Extract text
  ↓
IngestionService._chunk_document() → Split into chunks
  ↓
SentenceTransformerBackend.embed() → Generate embeddings
  ↓
VectorStoreManager.add_chunks() → Store in ChromaDB
  ↓
Return: IngestionResult {documents_processed, chunks_indexed}
```

### Query Flow (Current)
```
User asks: "What does the resume contain?"
  ↓
ChatService.generate_reply() → notebook_id provided
  ↓
RAGService.query() or LlamaIndexRAGService.query()
  ↓
VectorStoreManager.query() → Embed query, search all chunks
  ↓
Retrieve top-k chunks (e.g., 20 chunks)
  ↓
Group by source document
  ↓
Build prompt with chunks + question
  ↓
Ollama.generate() → Generate answer
  ↓
Return: RAGResponse {answer, sources}
```

---

## Next Steps: Two-Stage Retrieval Implementation

### Planned Changes

1. **Add Document Summary Generation**
   - Generate one-line summary during ingestion
   - Store summaries in separate collection or metadata

2. **Implement Two-Stage Retrieval**
   - Stage 1: Filter documents by summary similarity
   - Stage 2: Retrieve chunks only from relevant documents

3. **Performance Improvements**
   - Reduce chunks processed per query
   - Faster query times
   - Lower memory usage

### Expected Improvements

- **Query Speed:** 2-3x faster (process fewer chunks)
- **Memory Usage:** Lower (smaller retrieval sets)
- **Accuracy:** Better (focus on relevant documents)
- **Scalability:** Handle 10+ documents efficiently

---

## Measurement Plan

### Before Implementation
1. Measure current query latency (single doc, multi-doc)
2. Measure memory usage (embedding model + ChromaDB)
3. Measure retrieval time vs. collection size
4. Document current limitations with examples

### After Implementation
1. Compare query latency (before vs. after)
2. Compare memory usage
3. Compare retrieval accuracy
4. Measure summary generation time (one-time cost)

---

**Document Version:** 1.0  
**Last Updated:** 2024-12-19  
**Next Review:** After two-stage retrieval implementation

