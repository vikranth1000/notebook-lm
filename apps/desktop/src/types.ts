export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatRequest {
  prompt: string;
  history?: ChatMessage[];
}

export interface ChatResponse {
  reply: string;
  provider: string;
}

export interface BackendConfig {
  llm_provider: string;
  ollama_model: string;
  ollama_base_url: string;
}
