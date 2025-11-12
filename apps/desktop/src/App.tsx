import { useCallback, useEffect, useRef, useState } from 'react';
import './App.css';

import {
  fetchConfig,
  sendChatMessage,
  uploadDocument,
  listDocuments,
  getDocumentPreviewUrl,
  streamChatMessage,
  fetchMetricsSummary,
  exportConversation,
  downloadNotebookSummaries,
  transcribeAudio,
  speakText,
  requestAgentPlan,
} from './api';
import type {
  BackendConfig,
  ChatMessage,
  DocumentInfo,
  StreamSource,
  ChatStreamEvent,
  MetricsSummary,
} from './types';
import ReactMarkdown from 'react-markdown';
import DocumentPreview from './DocumentPreview';

type StatusState = 'starting' | 'ready' | 'error';

function App() {
  const [status, setStatus] = useState<StatusState>('starting');
  const [config, setConfig] = useState<BackendConfig | null>(null);
  const [bridgeMessage, setBridgeMessage] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [notebookId, setNotebookId] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [previewDocument, setPreviewDocument] = useState<DocumentInfo | null>(null);
  const [selectedDocumentIndex, setSelectedDocumentIndex] = useState<number | null>(null);
  const [latestMetrics, setLatestMetrics] = useState<Record<string, number> | null>(null);
  const [streamSources, setStreamSources] = useState<StreamSource[]>([]);
  const [metricsSummary, setMetricsSummary] = useState<MetricsSummary | null>(null);
  const [planGoal, setPlanGoal] = useState('');
  const [agentPlan, setAgentPlan] = useState<string | null>(null);
  const [isPlanning, setIsPlanning] = useState(false);
  const [speechStatus, setSpeechStatus] = useState<string | null>(null);
  const [ttsAudioUrl, setTtsAudioUrl] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const documentItemsRef = useRef<(HTMLDivElement | null)[]>([]);
  const assistantIndexRef = useRef<number | null>(null);
  const assistantBufferRef = useRef<string>('');

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const refreshMetricsSummary = useCallback(async () => {
    try {
      const summary = await fetchMetricsSummary();
      setMetricsSummary(summary);
    } catch (error) {
      console.warn('Failed to load metrics summary', error);
    }
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, isSubmitting]);

  useEffect(() => {
    return () => {
      if (ttsAudioUrl) {
        URL.revokeObjectURL(ttsAudioUrl);
      }
    };
  }, [ttsAudioUrl]);

  useEffect(() => {
    if (status === 'ready') {
      inputRef.current?.focus();
    }
  }, [status]);

  // Auto-focus input when submission completes
  useEffect(() => {
    if (!isSubmitting && status === 'ready') {
      // Small delay to ensure DOM has updated
      const timer = setTimeout(() => {
        inputRef.current?.focus();
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [isSubmitting, status]);

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
        inputRef.current?.focus();
        refreshMetricsSummary();
      } catch (error) {
        console.error(error);
        setStatus('error');
        setBridgeMessage('Failed to connect to backend');
      }
    }

    bootstrap();
  }, [refreshMetricsSummary]);

  const updateAssistantMessage = (content: string) => {
    setMessages((prev) => {
      if (assistantIndexRef.current === null) {
        return prev;
      }
      if (!prev[assistantIndexRef.current]) {
        return prev;
      }
      const next = [...prev];
      next[assistantIndexRef.current] = { ...next[assistantIndexRef.current], content };
      return next;
    });
  };

  const handleSend = async () => {
    if (!input.trim() || isSubmitting) return;

    const prompt = input.trim();
    const userMessage: ChatMessage = { role: 'user', content: prompt };
    const historyPayload = messages.map((m) => ({ role: m.role, content: m.content }));
    setInput('');
    inputRef.current?.focus();
    setIsSubmitting(true);
    setLatestMetrics(null);
    setStreamSources([]);
    assistantBufferRef.current = '';

    setMessages((prev) => {
      const next = [...prev, userMessage];
      const assistantIndex = next.length;
      next.push({ role: 'assistant', content: '' });
      assistantIndexRef.current = assistantIndex;
      return next;
    });

    const body = {
      prompt: userMessage.content,
      history: historyPayload,
      notebook_id: notebookId,
    };

    const handleStreamEvent = (event: ChatStreamEvent) => {
      switch (event.type) {
        case 'meta':
          setStreamSources(event.sources ?? []);
          if (event.metrics) {
            setLatestMetrics(event.metrics);
          }
          break;
        case 'token':
          assistantBufferRef.current += event.delta;
          updateAssistantMessage(assistantBufferRef.current);
          break;
        case 'done':
          assistantBufferRef.current = event.reply;
          updateAssistantMessage(assistantBufferRef.current);
          if (event.metrics) {
            setLatestMetrics(event.metrics);
          }
          setIsSubmitting(false);
          assistantIndexRef.current = null;
          refreshMetricsSummary();
          break;
        case 'error':
          updateAssistantMessage(`Error: ${event.message}`);
          setIsSubmitting(false);
          assistantIndexRef.current = null;
          break;
        default:
          break;
      }
    };

    try {
      await streamChatMessage(body, handleStreamEvent);
    } catch (streamError) {
      console.error('Streaming failed, falling back to standard request', streamError);
      try {
        const response = await sendChatMessage(body);
        assistantBufferRef.current = response.reply;
        updateAssistantMessage(response.reply);
        setLatestMetrics(response.metrics ?? null);
      } catch (error) {
        console.error(error);
        updateAssistantMessage(`Error: ${error instanceof Error ? error.message : 'Failed to get response'}`);
        setLatestMetrics(null);
      } finally {
        setIsSubmitting(false);
        assistantIndexRef.current = null;
        refreshMetricsSummary();
      }
    }
  };

  const loadDocuments = async (id: string) => {
    try {
      const result = await listDocuments(id);
      setDocuments(result.documents);
    } catch (error) {
      console.error('Failed to load documents:', error);
    }
  };

  useEffect(() => {
    if (notebookId) {
      loadDocuments(notebookId);
    } else {
      setDocuments([]);
    }
  }, [notebookId]);

  const handleExportConversation = async () => {
    try {
      await exportConversation('Offline Notebook LM Conversation', messages);
    } catch (error) {
      console.error('Export failed', error);
    }
  };

  const handleDownloadNotebook = async () => {
    if (!notebookId) return;
    try {
      await downloadNotebookSummaries(notebookId);
    } catch (error) {
      console.error('Notebook export failed', error);
    }
  };

  const handleAudioUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setSpeechStatus('Transcribing audio...');
    try {
      const transcript = await transcribeAudio(file);
      setInput((prev) => (prev ? `${prev}\n${transcript}` : transcript));
      setSpeechStatus('Transcript added to composer.');
    } catch (error) {
      setSpeechStatus(error instanceof Error ? error.message : 'Transcription failed');
    } finally {
      event.target.value = '';
    }
  };

  const handleSpeakLast = async () => {
    const lastAssistant = [...messages].reverse().find((msg) => msg.role === 'assistant');
    if (!lastAssistant) return;
    setSpeechStatus('Generating audio...');
    try {
      const url = await speakText(lastAssistant.content);
      setTtsAudioUrl(url);
      setSpeechStatus('Playing audio');
    } catch (error) {
      setSpeechStatus(error instanceof Error ? error.message : 'TTS failed');
    }
  };

  const handleAgentPlan = async () => {
    if (!planGoal.trim()) return;
    setIsPlanning(true);
    try {
      const response = await requestAgentPlan(planGoal.trim(), notebookId);
      setAgentPlan(response.plan);
    } catch (error) {
      setAgentPlan(error instanceof Error ? error.message : 'Failed to create plan');
    } finally {
      setIsPlanning(false);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setUploadStatus('Uploading and processing document...');

    try {
      const result = await uploadDocument(file, notebookId || undefined);
      setNotebookId(result.notebook_id);
      setUploadStatus(
        `✅ Document processed! ${result.documents_processed} document(s), ${result.chunks_indexed} chunk(s) indexed.`,
      );
      // Reload documents list
      await loadDocuments(result.notebook_id);
      setTimeout(() => setUploadStatus(null), 5000);
    } catch (error) {
      setUploadStatus(`❌ Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setTimeout(() => setUploadStatus(null), 5000);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleDocumentClick = (doc: DocumentInfo) => {
    setPreviewDocument(doc);
  };

  const handleDocumentKeyDown = (e: React.KeyboardEvent, doc: DocumentInfo, index: number) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      setPreviewDocument(doc);
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      const nextIndex = index < documents.length - 1 ? index + 1 : 0;
      setSelectedDocumentIndex(nextIndex);
      documentItemsRef.current[nextIndex]?.focus();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      const prevIndex = index > 0 ? index - 1 : documents.length - 1;
      setSelectedDocumentIndex(prevIndex);
      documentItemsRef.current[prevIndex]?.focus();
    }
  };

  // Global keyboard handler for spacebar when a document is focused
  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      // Only handle spacebar if we're not typing in an input
      if (e.key === ' ' && selectedDocumentIndex !== null && !previewDocument) {
        const target = e.target as HTMLElement;
        if (target.tagName !== 'INPUT' && target.tagName !== 'TEXTAREA') {
          e.preventDefault();
          const doc = documents[selectedDocumentIndex];
          if (doc) {
            setPreviewDocument(doc);
          }
        }
      }
    };

    window.addEventListener('keydown', handleGlobalKeyDown);
    return () => window.removeEventListener('keydown', handleGlobalKeyDown);
  }, [selectedDocumentIndex, documents, previewDocument]);

  const statusLabel = status === 'ready' ? 'Ready' : status === 'error' ? 'Error' : 'Starting';

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="branding">
          <h1>Offline Notebook LM</h1>
          <p>Chat with local models via Ollama</p>
        </div>
        <div className="header-meta">
          <div className={`status-indicator status-${status}`}>
            <span className="dot" />
            <span>{statusLabel}</span>
          </div>
        </div>
      </header>

      <main className="chat-main">
        <div className="chat-container">
          <div className="chat-messages">
            {messages.length === 0 && (
              <div className="empty-chat">
                <p>Start a conversation with your local model.</p>
                {config && (
                  <p className="config-hint">
                    Using {config.ollama_model} via {config.ollama_base_url}
                  </p>
                )}
              </div>
            )}
            {messages.map((msg, idx) => (
              <div key={idx} className={`message message-${msg.role}`}>
                <div className="message-role">{msg.role === 'user' ? 'You' : 'Assistant'}</div>
                {msg.content ? (
                  <ReactMarkdown className="message-content markdown">{msg.content}</ReactMarkdown>
                ) : (
                  <div className="message-content thinking-bubble">
                    <span className="thinking-text" data-text="thinking…">
                      thinking…
                    </span>
                  </div>
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          <div className="chat-input-area">
            <textarea
              value={input}
              ref={inputRef}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your message... (Press Enter to send, Shift+Enter for new line)"
              rows={3}
              disabled={isSubmitting || status !== 'ready'}
            />
            <button
              type="button"
              className="send-button"
              onClick={handleSend}
              disabled={!input.trim() || isSubmitting || status !== 'ready'}
            >
              {isSubmitting ? 'Sending...' : 'Send'}
            </button>
          </div>
        </div>

        <aside className="chat-sidebar">
          <div className="card">
            <h2>Upload Document</h2>
            <p className="upload-hint">Upload PDF, DOCX, TXT, or MD files for RAG-enabled chat.</p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx,.txt,.md"
              onChange={handleFileUpload}
              disabled={isUploading || status !== 'ready'}
              style={{ display: 'none' }}
            />
            <button
              type="button"
              className="upload-button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading || status !== 'ready'}
            >
              {isUploading ? 'Uploading...' : 'Choose File'}
            </button>
            {uploadStatus && <p className="upload-status">{uploadStatus}</p>}
            {notebookId && (
              <p className="notebook-id">
                <strong>Notebook ID:</strong> <code>{notebookId.slice(0, 8)}...</code>
              </p>
            )}
          </div>

          {documents.length > 0 && (
            <div className="card">
              <h2>Uploaded Documents ({documents.length})</h2>
              <div className="documents-list">
                {documents.map((doc, idx) => (
                  <div
                    key={idx}
                    ref={(el) => {
                      documentItemsRef.current[idx] = el;
                    }}
                    className={`document-item ${selectedDocumentIndex === idx ? 'document-item-selected' : ''}`}
                    onClick={() => handleDocumentClick(doc)}
                    onKeyDown={(e) => handleDocumentKeyDown(e, doc, idx)}
                    onFocus={() => setSelectedDocumentIndex(idx)}
                    onBlur={() => {
                      // Only clear selection if focus is moving to another document
                      setTimeout(() => {
                        if (!documentItemsRef.current.some((ref) => ref === document.activeElement)) {
                          setSelectedDocumentIndex(null);
                        }
                      }, 0);
                    }}
                    tabIndex={0}
                    role="button"
                    aria-label={`Preview ${doc.filename}. Press space or enter to open.`}
                  >
                    <div className="document-header">
                      <span className="document-name" title={doc.filename}>
                        {doc.filename}
                      </span>
                      <span className="document-chunks">{doc.chunk_count} chunks</span>
                    </div>
                    <div className="document-hint">Click or press Space to preview</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="card">
            <h2>Configuration</h2>
            {config ? (
              <dl className="config-list">
                <div>
                  <dt>Provider</dt>
                  <dd>{config.llm_provider || 'none'}</dd>
                </div>
                <div>
                  <dt>Model</dt>
                  <dd>
                    {config.resolved_ollama_model ?? config.ollama_model}
                    {config.model_selection_reason && (
                      <span className="config-hint"> ({config.model_selection_reason})</span>
                    )}
                  </dd>
                </div>
                <div>
                  <dt>Base URL</dt>
                  <dd>{config.ollama_base_url}</dd>
                </div>
                <div>
                  <dt>LangChain Splitter</dt>
                  <dd>{config.use_langchain_splitter ? 'enabled' : 'disabled'}</dd>
                </div>
                <div>
                  <dt>LlamaIndex RAG</dt>
                  <dd>{config.use_llamaindex_rag ? 'enabled' : 'disabled'}</dd>
                </div>
                {config.embedding_model && (
                  <div>
                    <dt>Embedding Model</dt>
                    <dd>{config.embedding_model}</dd>
                  </div>
                )}
              </dl>
            ) : (
              <p>Loading...</p>
            )}
          </div>

          <div className="card">
            <h2>Status</h2>
            <p>Bridge: {bridgeMessage ?? 'Waiting...'}</p>
            {status === 'error' && (
              <p className="error-hint">Make sure the backend is running on http://127.0.0.1:8000</p>
            )}
            {latestMetrics && (
              <div className="metrics-list">
                {Object.entries(latestMetrics).map(([key, value]) => (
                  <div key={key} className="metric-row">
                    <span className="metric-key">{key}</span>
                    <span className="metric-value">{value.toFixed(1)} ms</span>
                  </div>
                ))}
              </div>
            )}
            {streamSources.length > 0 && (
              <div className="sources-list">
                <div className="sources-title">Sources</div>
                {streamSources.map((src, idx) => (
                  <div key={`${src.source_path}-${idx}`} className="source-row">
                    <div className="source-path" title={src.source_path}>
                      {src.source_path.split(/[/\\]/).pop() ?? src.source_path}
                    </div>
                    <div className="source-preview">{src.preview}</div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {metricsSummary && (
            <div className="card">
              <h2>Analytics</h2>
              <div className="metrics-list">
                <div className="metric-row">
                  <span className="metric-key">Sessions</span>
                  <span className="metric-value">{metricsSummary.conversations}</span>
                </div>
                <div className="metric-row">
                  <span className="metric-key">Avg Total</span>
                  <span className="metric-value">
                    {metricsSummary.avg_total_ms ? metricsSummary.avg_total_ms.toFixed(1) : '--'} ms
                  </span>
                </div>
                <div className="metric-row">
                  <span className="metric-key">Avg Retrieval</span>
                  <span className="metric-value">
                    {metricsSummary.avg_retrieval_ms ? metricsSummary.avg_retrieval_ms.toFixed(1) : '--'} ms
                  </span>
                </div>
                <div className="metric-row">
                  <span className="metric-key">Avg LLM</span>
                  <span className="metric-value">
                    {metricsSummary.avg_llm_ms ? metricsSummary.avg_llm_ms.toFixed(1) : '--'} ms
                  </span>
                </div>
              </div>
              <div className="provider-list">
                {Object.entries(metricsSummary.provider_breakdown).map(([provider, count]) => (
                  <div key={provider} className="provider-row">
                    <span>{provider}</span>
                    <span>{count}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="card">
            <h2>Conversation Tools</h2>
            <button type="button" className="secondary-button" onClick={handleExportConversation} disabled={messages.length === 0}>
              Export Conversation
            </button>
            <button
              type="button"
              className="secondary-button"
              onClick={handleDownloadNotebook}
              disabled={!notebookId}
            >
              Download Notebook Summaries
            </button>
          </div>

          <div className="card">
            <h2>Agent Workspace</h2>
            <textarea
              value={planGoal}
              onChange={(e) => setPlanGoal(e.target.value)}
              placeholder="Goal (e.g., Summarize chapter 3)"
              rows={3}
            />
            <button type="button" className="secondary-button" onClick={handleAgentPlan} disabled={isPlanning || !planGoal.trim()}>
              {isPlanning ? 'Planning...' : 'Generate Plan'}
            </button>
            {agentPlan && <pre className="agent-plan">{agentPlan}</pre>}
          </div>

          {(config?.enable_speech_stt || config?.enable_speech_tts) && (
            <div className="card">
              <h2>Speech</h2>
              {config?.enable_speech_stt && (
                <>
                  <p className="upload-hint">Drop an audio clip to transcribe it into the composer.</p>
                  <input type="file" accept="audio/*" onChange={handleAudioUpload} />
                </>
              )}
              {config?.enable_speech_tts && (
                <button type="button" className="secondary-button" onClick={handleSpeakLast} disabled={!messages.some((m) => m.role === 'assistant')}>
                  Speak Last Answer
                </button>
              )}
              {speechStatus && <p className="upload-status">{speechStatus}</p>}
              {ttsAudioUrl && <audio src={ttsAudioUrl} controls autoPlay />}
            </div>
          )}
        </aside>
      </main>

      <footer className="app-footer">
        <p>All inference stays on-device. No cloud calls.</p>
      </footer>

      {previewDocument && notebookId && (
        <DocumentPreview
          isOpen={!!previewDocument}
          onClose={() => setPreviewDocument(null)}
          documentUrl={getDocumentPreviewUrl(notebookId, previewDocument.source_path)}
          filename={previewDocument.filename}
        />
      )}
    </div>
  );
}

export default App;
