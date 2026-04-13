export interface RetrievalHit {
  chunk_id: string
  doc_id: string
  score: number
  rerank_score: number | null
  platform: string
  title: string
  excerpt: string
  path: string
  created_at: string
  url: string | null
}

export interface Citation {
  source_id: string
  reason: string
}

export interface SourceRef {
  source_id: string
  platform: string
  title: string
  path: string
  score: number
  rerank_score: number | null
  url: string | null
  excerpt: string
}

export interface QAResponse {
  answer: string
  citations: Citation[]
  uncertainty: string | null
  sources: SourceRef[]
  debug?: Record<string, unknown>
}

export interface KbStatus {
  total_chats: number
  total_chunks: number
  by_platform: Record<string, number>
  vectorstore_size_bytes: number
  last_index_time: string | null
  is_indexing: boolean
}

export interface OllamaStatus {
  available: boolean
  models: string[]
  current_model: string
  base_url: string
}

export interface IndexProgress {
  task_id: string
  status: 'pending' | 'running' | 'done' | 'error'
  total_files: number
  processed_files: number
  total_chunks: number
  elapsed_seconds: number
  error: string | null
}

export interface ChatItem {
  id: string
  platform: string
  model: string | null
  title: string
  url: string | null
  tags: string[]
  message_count?: number
  created_at: string
  saved_at: string
  file_path: string
}

export interface ChatDetail {
  id: string
  meta: Record<string, unknown>
  content: string
}

export interface SSEEvent {
  type: 'token' | 'sources' | 'done' | 'error'
  content?: string
  sources?: SourceRef[]
  message?: string
}
