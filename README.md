# Offline Notebook LM

A simple, offline-first chat interface for local LLMs via Ollama. Start chatting with your local models immediately - no ingestion, no embeddings, just pure conversation.

## Quick Start

### Prerequisites
- [Ollama](https://ollama.ai) installed and running
- A model pulled (e.g., `ollama pull mistral`)

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

## Configuration

Set environment variables or create `backend/.env`:

```bash
NOTEBOOKLM_LLM_PROVIDER=ollama
NOTEBOOKLM_OLLAMA_MODEL=mistral
NOTEBOOKLM_OLLAMA_BASE_URL=http://127.0.0.1:11434
NOTEBOOKLM_LLM_MAX_TOKENS=512
```

## Features

- ✅ Simple chat interface
- ✅ Conversation history
- ✅ Works with any Ollama model
- ✅ Fully offline (no cloud calls)
- ✅ Clean, modern UI

## Roadmap

- [ ] Document ingestion and RAG
- [ ] Multiple model switching
- [ ] Export conversations
- [ ] Model management UI

## License

MIT
