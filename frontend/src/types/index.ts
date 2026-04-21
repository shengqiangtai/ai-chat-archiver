export type RerankMode = 'auto' | 'off' | 'on'
export type GraphMode = 'auto' | 'off'

export interface RetrievalHit {
  chunk_id: string
  doc_id: string
  score: number
  rerank_score: number | null
  keyword_score?: number | null
  fused_score?: number | null
  entity_score?: number | null
  platform: string
  title: string
  excerpt: string
  path: string
  created_at: string
  url: string | null
  role_summary?: string
  message_range?: string
  model_name?: string | null
  tags?: string[]
  entity_names?: string[]
  turn_index?: number
  chunk_index?: number
}

export interface QueryAnalysis {
  query_type: string
  enable_rewrite: boolean
  enable_rerank: boolean
  enable_graph: boolean
  reasons: string[]
}

export interface Citation {
  source_id: string
  reason: string
}

export interface SourceRef {
  source_id: string
  chunk_id: string
  platform: string
  title: string
  path: string
  score: number
  rerank_score: number | null
  url: string | null
  excerpt: string
  message_range?: string
  turn_index?: number
}

export interface RetrievalDebug {
  cache_hit: boolean
  retrieval_mode: string
  dense_count: number | null
  keyword_count: number | null
  entity_count: number | null
  graph_routed: boolean
  graph_hit_count: number
  graph_hits: RetrievalHit[]
  candidate_count: number
  final_count: number
  embed_time?: number
  search_time?: number
  total_time?: number
  original_query?: string
  rewritten_query?: string
  rewrite_applied?: boolean
  rewrite_strategy?: string
  rerank_requested_mode?: RerankMode
  rerank_effective_mode?: RerankMode | 'off'
  graph_requested_mode?: GraphMode
  graph_effective_mode?: 'off' | 'on'
  rerank_applied?: boolean
  rerank_status?: 'skipped' | 'applied' | 'fallback'
  rerank_reason?: string
  rerank_message?: string
  rerank_fallback?: boolean
  rerank_timed_out?: boolean
  rerank_elapsed_ms?: number
  rerank_candidate_limit?: number
  rerank_candidate_count?: number
  query_entities?: string[]
  expanded_entities?: string[]
  query_analysis?: QueryAnalysis
  analysis_scope?: string
  dense_hits: RetrievalHit[]
  keyword_hits: RetrievalHit[]
  entity_hits: RetrievalHit[]
  candidate_hits: RetrievalHit[]
  final_hits: RetrievalHit[]
}

export interface KbSearchResponse {
  query: string
  rewritten_query?: string
  rewrite_applied?: boolean
  rewrite_strategy?: string
  hits: RetrievalHit[]
  total: number
  debug?: RetrievalDebug
}

export interface QAResponse {
  answer: string
  citations: Citation[]
  uncertainty: string | null
  sources: SourceRef[]
  debug?: RetrievalDebug
}

export interface KbStatus {
  total_chats: number
  total_chunks: number
  total_entities: number
  top_entities: Array<{ name: string; entity_type: string; mention_count: number }>
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
  scanned_files?: number
  total_files: number
  processed_files: number
  skipped_files?: number
  skip_reasons?: Record<string, number>
  total_chunks: number
  elapsed_seconds: number
  error: string | null
}

export interface CleanupResult {
  ok: boolean
  message: string
  orphan_chat_count: number
  stale_file_record_count: number
  orphan_doc_count: number
  removed_chats: number
  removed_file_records: number
  removed_vector_docs: number
  removed_chunks: number
  removed_chunk_hash_docs: number
  sample_orphan_chat_ids: string[]
  sample_orphan_doc_ids: string[]
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
