export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatRequest {
  prompt: string;
  history?: ChatMessage[];
  notebook_id?: string | null;
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

export interface IngestionResponse {
  notebook_id: string;
  documents_processed: number;
  chunks_indexed: number;
}

export interface DocumentInfo {
  filename: string;
  source_path: string;
  chunk_count: number;
  preview: string;
}

export interface DocumentsListResponse {
  documents: DocumentInfo[];
}
