import { useEffect, useState } from 'react';
import './App.css';

import { fetchConfig, sendChatMessage } from './api';
import type { BackendConfig, ChatMessage } from './types';

type StatusState = 'starting' | 'ready' | 'error';

function App() {
  const [status, setStatus] = useState<StatusState>('starting');
  const [config, setConfig] = useState<BackendConfig | null>(null);
  const [bridgeMessage, setBridgeMessage] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

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
    setIsSubmitting(true);

    try {
      const response = await sendChatMessage({
        prompt: userMessage.content,
        history: messages.map((m) => ({ role: m.role, content: m.content })),
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

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

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
                <div className="message-content">{msg.content}</div>
              </div>
            ))}
            {isSubmitting && (
              <div className="message message-assistant">
                <div className="message-role">Assistant</div>
                <div className="message-content">Thinking...</div>
              </div>
            )}
          </div>

          <div className="chat-input-area">
            <textarea
              value={input}
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
    </div>
  );
}

export default App;
