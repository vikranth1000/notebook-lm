import type {
  BackendConfig,
  ChatRequest,
  ChatResponse,
  IngestionResponse,
  DocumentsListResponse,
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
