# Architecture – Offline Notebook LM

## System Overview
```
┌─────────────────────────────┐
│  Electron Desktop Shell     │
│  (React + Vite renderer)    │
│                             │
│  • Notebook browser UI      │
│  • Chat + notes workspace   │
│  • Ingestion + chat UI      │
│  • Invokes backend via IPC  │
└──────────────┬──────────────┘
               │
               │ HTTP / IPC (localhost, signed messages)
               ▼
┌─────────────────────────────┐
│  FastAPI Runtime            │
│  (`backend/`)               │
│                             │
│  • Ingestion scheduler      │
│  • Retrieval/Q&A endpoints  │
│  • Ingestion service        │
│  • RAG endpoint             │
└─────┬──────────┬────────────┘
      │          │
      │          │
      │          │
┌─────▼──────┐ ┌─▼────────────────┐
│ Vector DB  │ │ Model Executors   │
│ (Chroma)   │ │ (`llama.cpp`,     │
│            │ │  Sentence Embd.)  │
│ • Metadata │ │ • LLM inference   │
│ • Chunks   │ │ • Embeddings      │
└────────────┘ └──────────────────┘
```

## Components

- **Desktop Shell (`apps/desktop`)**
  - Electron main process orchestrates backend lifecycle and exposes secure IPC bridges.
  - React renderer provides notebook library, chat canvas, summarisation widgets, and diagnostics dashboard.
  - Talks to the backend over HTTP to ingest documents, request streaming chat responses, export conversations, and manage speech/agent tools.

- **Backend API (`backend/`)**
  - FastAPI app with routers for health, notebooks, ingestion jobs, retrieval, and summaries.
  - Additional routers deliver streaming chat (`/chat/stream`), metrics summaries, notebook exports, speech STT/TTS, and agent planning.
  - Configurable storage directories under `~/NotebookLM` (override via env vars).
  - Uses `sentence-transformers` embeddings persisted via ChromaDB and proxies generation to auto-selected Ollama models (or optional llama.cpp / onnxruntime).
  - Metadata/metrics persisted in a lightweight SQLite database (`metadata.db`) for notebooks, ingestion jobs, chat timings, and agent memory.

- **Data Plane**
  - Raw source documents stored under `data/`.
  - Vector indexes under `indexes/` (split by notebook ID).
  - Model weights cached under `models/` (GGUF, ONNX).
  - Metadata + job state + chat metrics persisted in SQLite (`metadata.db`).

- **Support Services**
  - MetricsStore captures per-chat latency and provider metrics for dashboards/CLI.
  - SpeechService optionally wraps faster-whisper (STT) and Piper (TTS) when enabled.
  - AgentService maintains lightweight plans/memory snippets stored in SQLite for agentic workflows.
  - ModelProfiles auto-select an Ollama model based on detected RAM and what’s already installed, keeping low-powered devices responsive.

## Offline Workflow

1. **Ingestion**
   - User drops files/folders into UI.
   - Renderer requests `/api/notebooks/ingest`.
   - Backend schedules parsing jobs, chunks text, computes embeddings, and stores vectors.

2. **Conversation**
   - Renderer sends messages to `/api/rag/query`.
   - Backend retrieves relevant chunks, feeds them to LLM (Phi-3/Mistral) via `llama.cpp`.
   - Response returned with citation metadata (chunk IDs, offsets).

3. **Summaries + Notes**
   - Batch summarisation jobs produce briefs and persistent notes stored in SQLite / Markdown exports.
   - UI renders note cards, allows user edits offline.

## Security & Privacy

- No outbound network requests after initial setup.
- All telemetry stays on-device; diagnostics limited to local UI.
- IPC bridge whitelists call signatures; future plan for signed messages and binary schema (Cap’n Proto or Protobuf).

## Roadmap Highlights

- Implement ingestion workers with concurrency controls and incremental updates.
- Integrate GPU acceleration for Apple Metal (via `llama.cpp` and `onnxruntime`).
- Bundle optional speech to text (`whisper.cpp`) and TTS (`piper`).
- Add migration system for metadata store and per-notebook access control.
