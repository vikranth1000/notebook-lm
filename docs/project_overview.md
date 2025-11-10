# Offline Notebook LM Clone – Project Overview

## Objectives (Version 1)
- Ship a desktop application that mirrors Notebook LM’s fundamental flow: ingest local docs, build embeddings, chat with citations—all offline.
- Keep the stack simple and reproducible with free/open tooling and local models (Ollama by default).
- Prioritise Apple Silicon macOS for the initial release; document steps for other platforms later.

## Success Criteria
- End-to-end flow: user imports local documents → system processes them → user chats and receives cited answers.
- Application functions without network access after initial model downloads.
- All components (code, models, build instructions) remain free and open-source compatible.
- Reasonable performance on a MacBook Air/Pro M-series using CPU/GPU acceleration.

## Core Features (v1)
- Library management for text, PDF, DOCX, and Markdown files.
- Background ingestion pipeline: chunking, embeddings, vector storage.
- Retrieval-Augmented Generation chat with citation links back to source snippets via Ollama or llama.cpp.
- Desktop UI for ingestion, job monitoring, and question answering.

## Out of Scope (v1)
- Cloud sync or multi-user collaboration.
- Audio transcription / TTS.
- Mobile apps or browser deployment.
- Automated note generation beyond Q&A citations.

## Risks & Mitigations
- **Model availability**: default to Ollama (free) and support manual GGUF paths when users want an alternate runner.
- **Document parsing edge cases**: rely on minimal, well-supported libraries (`pdfminer.six`, `python-docx`, direct Markdown parsing) and surface errors clearly.
- **Index size**: store notebooks separately in Chroma collections; allow users to delete/re-ingest via UI in future updates.
- **Setup friction**: ship quickstart docs, scripts, and sample notebook so first run succeeds without manual wiring.

## Next Steps
1. Polish ingest/chat UX for v1 and validate with sample docs.
2. Package macOS app and document Ollama setup.
3. Collect feedback, then iterate on exports, richer notes, and speech features.

