# Offline Notebook LM

An offline-first desktop RAG assistant for your local documents. Upload PDFs/DOCX/Markdown/TXT, build embeddings locally, and chat with grounded, cited answers via your local Ollama or llama.cpp models—no cloud required.

## Quick Start

### Prerequisites
- [Ollama](https://ollama.ai) installed and running
- A model pulled (e.g., `ollama pull mistral`)

> Tip: Leave `NOTEBOOKLM_OLLAMA_MODEL=auto`. The backend detects your RAM (via psutil) and chooses a lightweight Ollama model (`phi3:mini` for ≤12 GB, `qwen2.5:3b` for midsize, `mistral` for larger machines). If the selected model isn’t installed, run `ollama pull <model>` once.

### Run Everything
```bash
chmod +x scripts/dev.sh
NOTEBOOKLM_LLM_PROVIDER=ollama NOTEBOOKLM_OLLAMA_MODEL=mistral ./scripts/dev.sh
```

This starts:
- Backend API at `http://127.0.0.1:8000`
- Desktop app (Electron window opens automatically)

### Backend Only
```bash
cd backend
uv venv
uv pip install -e ".[dev]"
NOTEBOOKLM_LLM_PROVIDER=ollama NOTEBOOKLM_OLLAMA_MODEL=mistral uv run uvicorn notebooklm_backend.app:create_app --factory --reload
```

### Desktop Only
```bash
cd apps/desktop
npm install
npm run dev
```

## CLI Utilities

List notebooks, inspect ingestion jobs, and dump diagnostics from the terminal:

```bash
python scripts/notebooklm_cli.py notebooks
python scripts/notebooklm_cli.py jobs
python scripts/notebooklm_cli.py diagnostics
python scripts/notebooklm_cli.py metrics
python scripts/notebooklm_cli.py agent-plan --goal "Summarise chapter 3" --notebook <id>
```

## Configuration

Set environment variables or create `backend/.env`:

```bash
NOTEBOOKLM_LLM_PROVIDER=ollama
# Leave as auto for best match, or set a specific model (e.g., qwen2.5:3b, mistral)
NOTEBOOKLM_OLLAMA_MODEL=auto
NOTEBOOKLM_OLLAMA_BASE_URL=http://127.0.0.1:11434
NOTEBOOKLM_LLM_MAX_TOKENS=512
NOTEBOOKLM_LLM_PROVIDER=llama-cpp          # options: none, ollama, llama-cpp, onnx
NOTEBOOKLM_LLM_MODEL_PATH=/path/model.gguf
NOTEBOOKLM_ONNX_MODEL_PATH=/path/model.onnx
NOTEBOOKLM_ONNX_EXECUTION_PROVIDER=metal    # cpu|cuda|metal
NOTEBOOKLM_ENABLE_SPEECH_STT=1              # requires faster-whisper
NOTEBOOKLM_ENABLE_SPEECH_TTS=1              # requires piper-tts + voice files
```

## Features

- ✅ Desktop upload + ingestion pipeline (LangChain splitter fallback, sentence-transformers embeddings, ChromaDB)
- ✅ Two-stage RAG (document summaries + chunk retrieval) with citations and Markdown rendering
- ✅ Offline document preview modal with PDF zoom/pagination and secure file serving
- ✅ Streaming chat responses via SSE with live token rendering, live latency diagnostics, and source previews
- ✅ Configurable LLM backends (auto-selected Ollama models for your RAM, plus llama.cpp Metal hooks, onnxruntime-genai, deterministic offline fallback)
- ✅ Aggregated metrics dashboard + CLI, chat metrics persisted in SQLite for trend analysis
- ✅ Speech extras (optional STT via faster-whisper, TTS via Piper) gated behind config toggles
- ✅ Export utilities (conversation markdown, notebook summaries zip) and agentic planning API for autonomous workflows
- ✅ Automated baseline measurement scripts (`scripts/measure_baseline*.py`) that emit JSON + human-readable reports

## Roadmap

- [ ] Streaming chat responses end-to-end (SSE in FastAPI, token streaming UI)
- [ ] Notebook metadata persistence + job history in SQLite with CLI helpers
- [ ] GPU acceleration via llama.cpp Metal / onnxruntime, model management UI, multi-model switcher
- [ ] Speech add-ons (whisper.cpp STT, piper TTS) with optional hotkeys
- [ ] Conversation + notebook export bundles (Markdown/JSON) and sharing workflow

## License

MIT
