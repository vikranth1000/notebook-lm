import type { BackendConfig, ChatRequest, ChatResponse } from './types';

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
