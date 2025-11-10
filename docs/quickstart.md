# Quickstart Guide

Follow these steps to try the Offline Notebook LM experience with a local sample notebook.

## 1. Requirements
- macOS (tested on Apple Silicon)
- Node.js 20+
- Python 3.10+ or [uv](https://docs.astral.sh/uv/)
- Ollama running locally (optional, for LLM responses)

## 2. Install project dependencies
```bash
git clone <your-repo-url> offline-notebook-lm
cd offline-notebook-lm

# Desktop dependencies
cd apps/desktop
npm install
cd ../..

# Backend dependencies (uv recommended)
cd backend
uv venv
uv pip install -e ".[dev]"
cd ..
```

## 3. Launch everything together
```bash
chmod +x scripts/dev.sh
./scripts/dev.sh
```

This script starts the FastAPI backend at `http://127.0.0.1:8000` and opens the Electron desktop app in dev mode.

## 4. Ingest the sample notebook
1. In the desktop app, click **Browse…**.
2. Choose `samples/getting-started/research-notes.md`.
3. Click **Ingest notebook**.
4. Within a few seconds you should see a new notebook and ingestion job entry.

## 5. Ask questions
1. Select the new notebook.
2. Ask questions such as “How do alpine plants avoid freeze damage?”
3. Review the cited sources in the response.

## 6. Configure LLMs
- **Ollama** (default): ensure you have pulled a model, e.g. `ollama pull mistral`. The backend reads:
  ```bash
  export NOTEBOOKLM_LLM_PROVIDER=ollama
  export NOTEBOOKLM_OLLAMA_MODEL=mistral
  ```
- **llama.cpp**: download a GGUF model and set
  ```bash
  export NOTEBOOKLM_LLM_PROVIDER=llama-cpp
  export NOTEBOOKLM_LLM_MODEL_PATH=/absolute/path/to/model.gguf
  ```

Restart the backend after changing providers. With no LLM configured, the app returns deterministic summaries with source snippets.

## 7. Package the desktop app (optional)
```bash
cd apps/desktop
npm run package
```

The generated artifacts live under `apps/desktop/dist` and can be distributed for offline installs.

## 8. Run tests and lint checks
```bash
# Backend
cd backend
uv run ruff check .
uv run pytest

# Desktop
cd apps/desktop
npm run lint
npm run build
```

## Next steps
- Add your own document folders under `~/NotebookLM/data`.
- Bundle quantised models under `~/NotebookLM/models` for completely offline use.
- Fork the repo and customise prompts, chat UI, or ingestion logic.

