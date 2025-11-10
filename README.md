# Offline Notebook LM

An offline-first desktop experience inspired by Google's Notebook LM. Version 1 focuses on the essentials: ingest local documents, build embeddings, and chat with citations using local models (Ollama or llama.cpp) – all without touching the cloud.

## Repository Layout
```
.
├── apps/
│   └── desktop/          # Electron + React desktop shell
├── backend/              # FastAPI-based ingestion and RAG services
├── docs/                 # Planning, architecture, and research notes
└── models/               # (Placeholder) model weights cache
```

## Getting Started

### Quick Start (both backend + desktop)
```bash
chmod +x scripts/dev.sh
./scripts/dev.sh
```

This runs the backend API at `http://127.0.0.1:8000` and opens the desktop app in dev mode.
Use the UI to browse to `samples/getting-started/research-notes.md` and ingest your first notebook in seconds.

### Desktop App (standalone)
```bash
cd apps/desktop
npm install
npm run dev
```

The dev script launches Vite for the renderer and Electron for the main process. The UI currently renders a systems dashboard and will later orchestrate backend workers.

### Backend API
```bash
cd backend
uv venv        # or python -m venv .venv
uv pip install -e ".[dev]"
uv run uvicorn notebooklm_backend.app:create_app --factory --reload
```

FastAPI is exposed at `http://127.0.0.1:8000`; the desktop app will communicate with this service for ingestion, retrieval, and settings.

### Configure LLMs
- **Ollama (default)**: make sure Ollama is running locally and you have pulled a model (e.g. `ollama pull mistral`). The backend will auto-connect if `NOTEBOOKLM_LLM_PROVIDER` is set to `ollama`.
- **llama.cpp (optional)**: install `llama-cpp-python` manually and set `NOTEBOOKLM_LLM_PROVIDER=llama-cpp` plus `NOTEBOOKLM_LLM_MODEL_PATH=/abs/path/model.gguf`.

Set these variables before launching `scripts/dev.sh`, or create a `.env` file inside `backend/`.

### Packaging
```bash
cd apps/desktop
npm run package
```

Electron Builder outputs installers under `apps/desktop/dist` ready for distribution. On macOS, double-click the `.dmg` to install the offline app.

### Tests
```bash
cd backend
uv run pytest
```

## Roadmap
- [ ] Notebook deletion, metadata editing, and re-ingest flows.
- [ ] Export chat answers with citations (Markdown/PDF).
- [ ] First-run wizard for downloading GGUF/ONNX models.
- [ ] Packaging and notarised macOS binaries.
- [ ] Windows/Linux ports and GPU acceleration toggles.

## License
MIT

## Additional Docs
- `docs/project_overview.md` – high-level goals and scope.
- `docs/architecture.md` – system diagram and component breakdown.
- `docs/quickstart.md` – step-by-step walkthrough with the sample notebook.

