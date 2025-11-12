import type {
  BackendConfig,
  ChatRequest,
  ChatResponse,
  IngestionResponse,
  DocumentsListResponse,
  ChatStreamEvent,
  MetricsSummary,
  AgentPlanResponse,
  ChatMessage,
} from './types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000/api';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
    ...init,
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with status ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export function fetchConfig(): Promise<BackendConfig> {
  return request<BackendConfig>('/config');
}

export function fetchMetricsSummary(): Promise<MetricsSummary> {
  return request<MetricsSummary>('/metrics/summary');
}

export function sendChatMessage(body: ChatRequest): Promise<ChatResponse> {
  return request<ChatResponse>('/chat/', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export async function uploadDocument(file: File, notebookId?: string): Promise<IngestionResponse> {
  const formData = new FormData();
  formData.append('file', file);
  if (notebookId) {
    formData.append('notebook_id', notebookId);
  }

  const response = await fetch(`${API_BASE_URL}/documents/ingest`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Upload failed with status ${response.status}`);
  }

  return response.json() as Promise<IngestionResponse>;
}

export function listDocuments(notebookId: string): Promise<DocumentsListResponse> {
  return request<DocumentsListResponse>(`/documents/list?notebook_id=${encodeURIComponent(notebookId)}`);
}

export function getDocumentPreviewUrl(notebookId: string, sourcePath: string): string {
  const params = new URLSearchParams({
    notebook_id: notebookId,
    source_path: sourcePath,
  });
  return `${API_BASE_URL}/documents/preview?${params.toString()}`;
}

export async function streamChatMessage(
  body: ChatRequest,
  onEvent: (event: ChatStreamEvent) => void,
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok || !response.body) {
    const detail = await response.text();
    throw new Error(detail || `Stream failed with status ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  const processBuffer = () => {
    let boundary = buffer.indexOf('\n\n');
    while (boundary !== -1) {
      const rawEvent = buffer.slice(0, boundary).trim();
      buffer = buffer.slice(boundary + 2);
      if (rawEvent.startsWith('data:')) {
        const payload = rawEvent.replace(/^data:\s*/, '');
        if (payload) {
          try {
            const parsed = JSON.parse(payload) as ChatStreamEvent;
            onEvent(parsed);
          } catch (error) {
            console.warn('Failed to parse stream payload', error);
          }
        }
      }
      boundary = buffer.indexOf('\n\n');
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    processBuffer();
  }

  buffer += decoder.decode();
  processBuffer();
}

export async function exportConversation(title: string, messages: ChatMessage[]): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/export/conversation`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ title, messages }),
  });
  if (!response.ok) {
    throw new Error(`Export failed (${response.status})`);
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${title.replace(/\s+/g, '_')}.md`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export async function downloadNotebookSummaries(notebookId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/export/notebook/${encodeURIComponent(notebookId)}`);
  if (!response.ok) {
    throw new Error(`Export failed (${response.status})`);
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `notebook-${notebookId.slice(0, 8)}.zip`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export async function transcribeAudio(file: File): Promise<string> {
  const form = new FormData();
  form.append('audio', file);
  const response = await fetch(`${API_BASE_URL}/speech/transcribe`, {
    method: 'POST',
    body: form,
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const data = await response.json();
  return data.transcript as string;
}

export async function speakText(text: string): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/speech/speak`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  return url;
}

export async function requestAgentPlan(goal: string, notebookId?: string | null): Promise<AgentPlanResponse> {
  return request<AgentPlanResponse>('/agent/plan', {
    method: 'POST',
    body: JSON.stringify({ goal, notebook_id: notebookId ?? null }),
  });
}
