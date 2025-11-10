import { useEffect, useMemo, useState } from 'react';
import './App.css';

import {
  fetchConfig,
  fetchJobs,
  fetchNotebooks,
  ingestNotebook,
  ragQuery,
} from './api';
import type {
  BackendConfig,
  IngestionJobStatus,
  NotebookMetadata,
  RAGQueryResponse,
  RAGSource,
} from './types';

type StatusState = 'starting' | 'ready' | 'error';

interface Banner {
  type: 'success' | 'error';
  message: string;
}

interface ChatTurn {
  id: string;
  question: string;
  answer: string | null;
  sources: RAGSource[];
  status: 'pending' | 'complete' | 'error';
  createdAt: string;
  error?: string;
}

const POLL_INTERVAL = 4000;

function formatDate(iso: string | null | undefined): string {
  if (!iso) return '–';
  return new Date(iso).toLocaleString();
}

function formatSourcePath(path: string): string {
  if (!path) return 'Unknown source';
  const segments = path.split(/[\\/]/);
  return segments.slice(-2).join('/');
}

const createTurnId = () =>
  typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;

function App() {
  const [status, setStatus] = useState<StatusState>('starting');
  const [config, setConfig] = useState<BackendConfig | null>(null);
  const [bridgeMessage, setBridgeMessage] = useState<string | null>(null);
  const [notebooks, setNotebooks] = useState<NotebookMetadata[]>([]);
  const [jobs, setJobs] = useState<IngestionJobStatus[]>([]);
  const [selectedNotebookId, setSelectedNotebookId] = useState<string | null>(null);
  const [ingestPath, setIngestPath] = useState('');
  const [ingestNotebookId, setIngestNotebookId] = useState('');
  const [recursive, setRecursive] = useState(true);
  const [isIngesting, setIsIngesting] = useState(false);
  const [banner, setBanner] = useState<Banner | null>(null);

  const [question, setQuestion] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [chatTurns, setChatTurns] = useState<ChatTurn[]>([]);

  useEffect(() => {
    async function bootstrap() {
      try {
        if (!window.notebookBridge) {
          setStatus('error');
          setBridgeMessage('Electron bridge unavailable');
          return;
        }

        const response = await window.notebookBridge.ping();
        setBridgeMessage(response);
        const cfg = await fetchConfig();
        setConfig(cfg);
        setStatus('ready');
      } catch (error) {
        console.error(error);
        setStatus('error');
        setBridgeMessage('Failed to talk to backend bridge');
      }
    }

    bootstrap();
  }, []);

  useEffect(() => {
    if (status !== 'ready') return;

    let cancelled = false;

    const poll = async () => {
      try {
        const [nb, jb] = await Promise.all([fetchNotebooks(), fetchJobs()]);
        if (!cancelled) {
          setNotebooks(nb);
          setJobs(jb);
          if (!selectedNotebookId && nb.length > 0) {
            setSelectedNotebookId(nb[0].notebook_id);
          }
        }
      } catch (error) {
        if (!cancelled) {
          console.error(error);
          setStatus('error');
          setBanner({ type: 'error', message: 'Lost connection to backend. Check that it is running.' });
        }
      }
    };

    poll();
    const interval = window.setInterval(poll, POLL_INTERVAL);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [status, selectedNotebookId]);

  const selectedNotebook = useMemo(
    () => notebooks.find((nb) => nb.notebook_id === selectedNotebookId) ?? null,
    [notebooks, selectedNotebookId],
  );

  const statusLabel = useMemo(() => {
    switch (status) {
      case 'ready':
        return 'Ready';
      case 'error':
        return 'Bridge error';
      default:
        return 'Starting';
    }
  }, [status]);

  const handleChoosePath = async () => {
    try {
      const path = await window.notebookBridge?.choosePath({
        title: 'Select file or folder to ingest',
        properties: ['openDirectory', 'openFile'],
      });
      if (path) {
        setIngestPath(path);
      }
    } catch (error) {
      console.error(error);
      setBanner({ type: 'error', message: 'Unable to open file picker.' });
    }
  };

  const handleIngest = async () => {
    if (!ingestPath) {
      setBanner({ type: 'error', message: 'Please select a file or folder to ingest.' });
      return;
    }
    setIsIngesting(true);
    try {
      const result = await ingestNotebook({
        path: ingestPath,
        notebook_id: ingestNotebookId || undefined,
        recursive,
      });
      setBanner({
        type: 'success',
        message: result.status === 'failed' ? result.message ?? 'Ingestion failed.' : result.message ?? 'Ingestion started.',
      });
      setIngestNotebookId('');
    } catch (error) {
      console.error(error);
      setBanner({ type: 'error', message: error instanceof Error ? error.message : 'Failed to ingest notebook.' });
    } finally {
      setIsIngesting(false);
    }
  };

  const handleAsk = async () => {
    if (!selectedNotebook) {
      setBanner({ type: 'error', message: 'Please select a notebook first.' });
      return;
    }
    if (!question.trim()) {
      return;
    }

    const turnId = createTurnId();
    const createdAt = new Date().toISOString();

    setChatTurns((prev) => [
      {
        id: turnId,
        question,
        answer: null,
        sources: [],
        status: 'pending',
        createdAt,
      },
      ...prev,
    ]);

    setQuestion('');
    setIsSubmitting(true);

    try {
      const response: RAGQueryResponse = await ragQuery({
        notebook_id: selectedNotebook.notebook_id,
        question,
        top_k: 5,
      });

      setChatTurns((prev) =>
        prev.map((turn) =>
          turn.id === turnId
            ? {
                ...turn,
                answer: response.answer,
                sources: response.sources,
                status: 'complete',
              }
            : turn,
        ),
      );
    } catch (error) {
      console.error(error);
      const message = error instanceof Error ? error.message : 'Failed to get response from backend.';
      setChatTurns((prev) =>
        prev.map((turn) =>
          turn.id === turnId
            ? {
                ...turn,
                answer: null,
                status: 'error',
                error: message,
              }
            : turn,
        ),
      );
      setBanner({ type: 'error', message });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleResetChat = () => {
    setChatTurns([]);
  };

  const handleOpenDocs = async () => {
    const url = 'https://github.com';
    try {
      await window.notebookBridge?.openExternal?.(url);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="branding">
          <h1>Offline Notebook LM</h1>
          <p>Private research notebooks with fully local ingestion, embeddings, and chat.</p>
        </div>
        <div className="header-meta">
          <div className={`status-indicator status-${status}`}>
            <span className="dot" />
            <span>{statusLabel}</span>
          </div>
          <button className="ghost-button" onClick={handleOpenDocs}>
            View Docs
          </button>
        </div>
      </header>

      {banner && (
        <div className={`banner banner-${banner.type}`} onClick={() => setBanner(null)}>
          {banner.message}
        </div>
      )}

      <div className="content">
        <aside className="sidebar">
          <section className="card">
            <h2>Ingest documents</h2>
            <p className="subtitle">
              Drop folders or files to build a notebook. Supported formats: PDF, DOCX, Markdown, TXT.
            </p>

            <label className="field">
              <span>File or folder path</span>
              <div className="field-row">
                <input
                  type="text"
                  value={ingestPath}
                  onChange={(event) => setIngestPath(event.target.value)}
                  placeholder="/Users/you/Documents/research"
                />
                <button type="button" className="secondary" onClick={handleChoosePath}>
                  Browse…
                </button>
              </div>
            </label>

            <label className="field">
              <span>Notebook ID (optional)</span>
              <input
                type="text"
                value={ingestNotebookId}
                onChange={(event) => setIngestNotebookId(event.target.value)}
                placeholder="auto-generated if left blank"
              />
            </label>

            <label className="checkbox">
              <input
                type="checkbox"
                checked={recursive}
                onChange={(event) => setRecursive(event.target.checked)}
              />
              <span>Include sub-folders</span>
            </label>

            <button
              type="button"
              className="primary-button"
              onClick={handleIngest}
              disabled={isIngesting}
            >
              {isIngesting ? 'Ingesting…' : 'Ingest notebook'}
            </button>
            <p className="status-hint">Bridge: {bridgeMessage ?? 'Waiting…'}</p>
          </section>

          <section className="card">
            <div className="section-header">
              <h2>Notebooks</h2>
              <span className="count">{notebooks.length}</span>
            </div>
            <ul className="notebook-list">
              {notebooks.map((notebook) => (
                <li
                  key={notebook.notebook_id}
                  className={notebook.notebook_id === selectedNotebookId ? 'active' : ''}
                  onClick={() => {
                    setSelectedNotebookId(notebook.notebook_id);
                    setChatTurns([]);
                  }}
                >
                  <div className="notebook-title">{notebook.title || notebook.notebook_id}</div>
                  <div className="notebook-meta">
                    {notebook.source_count} sources · {notebook.chunk_count} chunks
                  </div>
                  <div className="notebook-updated">Updated {formatDate(notebook.updated_at)}</div>
                </li>
              ))}
              {notebooks.length === 0 && (
                <li className="empty-state">No notebooks yet. Ingest a folder to get started.</li>
              )}
            </ul>
          </section>

          <section className="card">
            <div className="section-header">
              <h2>Ingestion jobs</h2>
              <span className="count">{jobs.length}</span>
            </div>
            <ul className="job-list">
              {jobs.map((job) => (
                <li key={job.job_id}>
                  <div className="job-header">
                    <span className={`job-status job-${job.status}`}>{job.status}</span>
                    <span className="job-id">{job.job_id.slice(0, 8)}</span>
                  </div>
                  <div className="job-meta">
                    {job.documents_processed} docs · {job.chunks_indexed} chunks
                  </div>
                  <div className="job-message">{job.message ?? 'Working…'}</div>
                  <div className="job-timestamp">
                    {job.started_at ? `Started ${formatDate(job.started_at)}` : ''}
                    {job.completed_at ? ` • Finished ${formatDate(job.completed_at)}` : ''}
                  </div>
                </li>
              ))}
              {jobs.length === 0 && <li className="empty-state">No jobs yet.</li>}
            </ul>
          </section>
        </aside>

        <main className="workspace">
          <section className="card">
            <div className="section-header">
      <div>
                <h2>Chat with your notebooks</h2>
                {selectedNotebook ? (
                  <p className="subtitle">
                    Currently chatting with <strong>{selectedNotebook.title || selectedNotebook.notebook_id}</strong>
                  </p>
                ) : (
                  <p className="subtitle">Select a notebook to start asking questions.</p>
                )}
              </div>
              <button className="ghost-button" onClick={handleResetChat} disabled={chatTurns.length === 0}>
                Clear chat
              </button>
      </div>

            <div className="question-box">
              <textarea
                placeholder={
                  selectedNotebook
                    ? 'Ask about the contents of this notebook…'
                    : 'Load a notebook before asking a question.'
                }
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                rows={3}
                disabled={!selectedNotebook || isSubmitting}
              />
              <div className="question-actions">
                <button
                  type="button"
                  className="primary-button"
                  onClick={handleAsk}
                  disabled={!selectedNotebook || isSubmitting || !question.trim()}
                >
                  {isSubmitting ? 'Thinking…' : 'Ask question'}
        </button>
              </div>
            </div>

            <div className="chat-log">
              {chatTurns.length === 0 && (
                <div className="empty-chat">
                  Ready when you are. Ask about summaries, cross-reference notes, or request study guides.
                </div>
              )}
              {chatTurns.map((turn) => (
                <article key={turn.id} className="chat-turn">
                  <header>
                    <span className="chat-question">{turn.question}</span>
                    <time>{formatDate(turn.createdAt)}</time>
                  </header>

                  {turn.status === 'pending' && <p className="pending">Generating answer…</p>}
                  {turn.status === 'error' && <p className="error">Failed: {turn.error}</p>}
                  {turn.status === 'complete' && turn.answer && (
                    <p className="answer">{turn.answer}</p>
                  )}

                  {turn.sources.length > 0 && (
                    <ul className="source-list">
                      {turn.sources.map((source, index) => (
                        <li key={`${turn.id}-source-${index}`}>
                          <span className="source-label">Source {index + 1}</span>
                          <span className="source-path">{formatSourcePath(source.source_path)}</span>
                          <p className="source-snippet">
                            {source.content.length > 320 ? `${source.content.slice(0, 320)}…` : source.content}
                          </p>
                        </li>
                      ))}
                    </ul>
                  )}
                </article>
              ))}
            </div>
          </section>

          <section className="card">
            <h2>Configuration</h2>
            {config ? (
              <dl className="config-grid">
                <div>
                  <dt>Models directory</dt>
                  <dd>{config.models_dir}</dd>
                </div>
                <div>
                  <dt>Indexes directory</dt>
                  <dd>{config.indexes_dir}</dd>
                </div>
                <div>
                  <dt>Workspace root</dt>
                  <dd>{config.workspace_root}</dd>
                </div>
                <div>
                  <dt>Audio features</dt>
                  <dd>{config.enable_audio ? 'Enabled' : 'Disabled'}</dd>
                </div>
              </dl>
            ) : (
              <p>Loading configuration…</p>
            )}
          </section>
        </main>
      </div>

      <footer className="app-footer">
        <p>All inference, storage, and retrieval stay on-device. No cloud calls.</p>
      </footer>
    </div>
  );
}

export default App;
