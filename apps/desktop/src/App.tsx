import { useEffect, useRef, useState } from 'react';
import './App.css';

import { fetchConfig, sendChatMessage, uploadDocument, listDocuments, getDocumentPreviewUrl } from './api';
import type { BackendConfig, ChatMessage, DocumentInfo } from './types';
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
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const documentItemsRef = useRef<(HTMLDivElement | null)[]>([]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isSubmitting]);

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
      } catch (error) {
        console.error(error);
        setStatus('error');
        setBridgeMessage('Failed to connect to backend');
      }
    }

    bootstrap();
  }, []);

  const handleSend = async () => {
    if (!input.trim() || isSubmitting) return;

    const userMessage: ChatMessage = { role: 'user', content: input.trim() };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    inputRef.current?.focus();
    setIsSubmitting(true);

    try {
      const response = await sendChatMessage({
        prompt: userMessage.content,
        history: messages.map((m) => ({ role: m.role, content: m.content })),
        notebook_id: notebookId,
      });

      const assistantMessage: ChatMessage = { role: 'assistant', content: response.reply };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error(error);
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Failed to get response'}`,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsSubmitting(false);
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
                <ReactMarkdown className="message-content markdown">{msg.content}</ReactMarkdown>
              </div>
            ))}
            {isSubmitting && (
              <div className="message message-assistant">
                <div className="message-role">Assistant</div>
                <div className="message-content">Thinking...</div>
              </div>
            )}
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
                  <dd>{config.ollama_model}</dd>
                </div>
                <div>
                  <dt>Base URL</dt>
                  <dd>{config.ollama_base_url}</dd>
                </div>
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
          </div>
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
