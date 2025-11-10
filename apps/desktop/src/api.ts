import type {
  BackendConfig,
  IngestionJobStatus,
  NotebookMetadata,
  RAGQueryResponse,
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

export function fetchNotebooks(): Promise<NotebookMetadata[]> {
  return request<NotebookMetadata[]>('/notebooks/');
}

export function fetchJobs(): Promise<IngestionJobStatus[]> {
  return request<IngestionJobStatus[]>('/notebooks/jobs');
}

export function ingestNotebook(body: { path: string; notebook_id?: string; recursive: boolean }) {
  return request<IngestionJobStatus>('/notebooks/ingest', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export function ragQuery(body: { notebook_id: string; question: string; top_k: number }) {
  return request<RAGQueryResponse>('/notebooks/rag/query', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

