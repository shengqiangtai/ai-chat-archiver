const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8765'

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  })
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const data = await res.json()
      detail = data.detail || detail
    } catch {
      // ignore
    }
    throw new Error(detail)
  }
  return res.json()
}

export const api = {
  // 聊天记录
  getChats: (platform?: string, limit = 200) => {
    const params = new URLSearchParams({ limit: String(limit) })
    if (platform) params.set('platform', platform)
    return request<{ chats: any[]; count: number }>(`/chats?${params}`)
  },
  getChat: (id: string) => request<any>(`/chats/${id}`),
  deleteChat: (id: string) => request<any>(`/chats/${id}`, { method: 'DELETE' }),
  search: (query: string, platform?: string) =>
    request<{ results: any[]; count: number }>('/search', {
      method: 'POST',
      body: JSON.stringify({ query, platform: platform || null, limit: 100 }),
    }),
  getStats: () => request<any>('/stats'),

  // 知识库
  getKbStatus: () => request<any>('/api/kb/status'),
  reindex: () => request<any>('/api/kb/reindex', { method: 'POST' }),
  reindexIncremental: () => request<any>('/api/kb/reindex/incremental', { method: 'POST' }),
  getIndexProgress: (taskId: string) => request<any>(`/api/kb/reindex/progress/${taskId}`),

  kbSearch: (query: string, topK = 10, platformFilter?: string) =>
    request<any>('/api/kb/search', {
      method: 'POST',
      body: JSON.stringify({
        query,
        top_k: topK,
        platform_filter: platformFilter || null,
        score_threshold: 0.25,
      }),
    }),

  kbQA: (query: string, mode = 'concise', topK = 15, topN = 5) =>
    request<any>('/api/kb/qa', {
      method: 'POST',
      body: JSON.stringify({ query, mode, top_k: topK, top_n: topN }),
    }),

  kbQAStreamUrl: () => `${API_BASE}/api/kb/qa/stream`,

  getOllamaStatus: () => request<any>('/api/kb/ollama/status'),
  switchOllamaModel: (model: string) =>
    request<any>('/api/kb/ollama/model', {
      method: 'PUT',
      body: JSON.stringify({ model }),
    }),

  getLLMStatus: () => request<any>('/api/kb/llm/status'),
  switchLLMBackend: (backend: string, model?: string) =>
    request<any>('/api/kb/llm/backend', {
      method: 'PUT',
      body: JSON.stringify({ backend, model: model || null }),
    }),
}
