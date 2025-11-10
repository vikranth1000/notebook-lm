export interface BackendConfig {
  models_dir: string;
  indexes_dir: string;
  workspace_root: string;
  enable_audio: boolean;
}

export interface NotebookMetadata {
  notebook_id: string;
  title: string;
  description?: string | null;
  created_at: string;
  updated_at: string;
  source_count: number;
  chunk_count: number;
}

export interface IngestionJobStatus {
  job_id: string;
  notebook_id: string;
  status: 'queued' | 'running' | 'failed' | 'completed';
  message?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  documents_processed: number;
  chunks_indexed: number;
}

export interface RAGSource {
  source_path: string;
  content: string;
  distance?: number | null;
}

export interface RAGQueryResponse {
  answer: string;
  sources: RAGSource[];
}

