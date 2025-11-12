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
  metrics?: Record<string, number>;
}

export interface StreamSource {
  source_path: string;
  preview: string;
  distance?: number | null;
}

export type ChatStreamEvent =
  | {
      type: 'meta';
      provider: string;
      sources: StreamSource[];
      metrics?: Record<string, number>;
    }
  | { type: 'token'; delta: string }
  | { type: 'done'; reply: string; metrics?: Record<string, number> }
  | { type: 'error'; message: string };

export interface MetricsSummary {
  conversations: number;
  avg_total_ms?: number | null;
  avg_llm_ms?: number | null;
  avg_retrieval_ms?: number | null;
  provider_breakdown: Record<string, number>;
}

export interface AgentPlanResponse {
  plan: string;
}

export interface BackendConfig {
  llm_provider: string;
  ollama_model: string;
  resolved_ollama_model?: string;
  model_selection_reason?: string;
  ollama_base_url: string;
  use_langchain_splitter?: boolean;
  use_llamaindex_rag?: boolean;
  embedding_model?: string;
  enable_speech_stt?: boolean;
  enable_speech_tts?: boolean;
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
