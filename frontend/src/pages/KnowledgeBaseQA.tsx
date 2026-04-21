import { useState, useRef, useEffect, useCallback, type KeyboardEvent } from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import 'highlight.js/styles/github-dark.css'
import { api } from '../api/client'
import type { KbStatus, RetrievalDebug, RerankMode, SourceRef, SSEEvent } from '../types'
import SourcePreview from '../components/SourcePreview'
import ReindexPanel from '../components/ReindexPanel'
import SearchBox from '../components/SearchBox'

type SubTab = 'qa' | 'manage' | 'semantic'

interface QAMessage {
  role: 'user' | 'ai'
  content: string
  sources?: SourceRef[]
  citations?: { source_id: string; reason: string }[]
  uncertainty?: string | null
}

function rerankStatusTone(status?: 'skipped' | 'applied' | 'fallback') {
  if (status === 'applied') return 'text-emerald-300 border-emerald-700/60 bg-emerald-950/30'
  if (status === 'fallback') return 'text-amber-300 border-amber-700/60 bg-amber-950/30'
  return 'text-slate-300 border-slate-700/60 bg-slate-900/40'
}

function rerankStatusLabel(status?: 'skipped' | 'applied' | 'fallback') {
  if (status === 'applied') return '已执行'
  if (status === 'fallback') return '已回退'
  return '已跳过'
}

export default function KnowledgeBaseQA() {
  const [subTab, setSubTab] = useState<SubTab>('qa')
  const [messages, setMessages] = useState<QAMessage[]>([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [platform, setPlatform] = useState('')
  const [mode, setMode] = useState<'concise' | 'detailed'>('concise')
  const [qaRetrievalMode, setQaRetrievalMode] = useState<'hybrid' | 'vector' | 'keyword' | 'entity' | 'mix'>('mix')
  const [qaRerankMode, setQaRerankMode] = useState<RerankMode>('auto')
  const [kbStatus, setKbStatus] = useState<KbStatus | null>(null)
  const [kbStatusError, setKbStatusError] = useState<string | null>(null)
  const listRef = useRef<HTMLDivElement>(null)

  // 语义搜索状态
  const [searchResults, setSearchResults] = useState<any[]>([])
  const [searchLoading, setSearchLoading] = useState(false)
  const [searchPlatform, setSearchPlatform] = useState('')
  const [searchRetrievalMode, setSearchRetrievalMode] = useState<'hybrid' | 'vector' | 'keyword' | 'entity' | 'mix'>('mix')
  const [searchRerankMode, setSearchRerankMode] = useState<RerankMode>('auto')
  const [searchDebug, setSearchDebug] = useState<RetrievalDebug | null>(null)
  const [searchRewrite, setSearchRewrite] = useState<{ rewritten?: string; applied?: boolean; strategy?: string } | null>(null)
  const [searchError, setSearchError] = useState<string | null>(null)
  const [hasSearched, setHasSearched] = useState(false)

  const scrollToBottom = useCallback(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight
    }
  }, [])

  useEffect(() => { scrollToBottom() }, [messages, scrollToBottom])
  useEffect(() => {
    api.getKbStatus()
      .then((data) => {
        setKbStatus(data)
        setKbStatusError(null)
      })
      .catch((err: any) => {
        setKbStatusError(err.message || '知识库状态加载失败')
      })
  }, [])

  const handleSend = async () => {
    const question = input.trim()
    if (!question || sending) return
    if (kbStatus && kbStatus.total_chunks === 0) {
      setMessages((prev) => [
        ...prev,
        { role: 'ai', content: '知识库当前还没有可检索的内容。请先归档聊天并执行索引。' },
      ])
      return
    }

    setInput('')
    setSending(true)
    setMessages((prev) => [...prev, { role: 'user', content: question }])

    const aiMsg: QAMessage = { role: 'ai', content: '' }
    setMessages((prev) => [...prev, aiMsg])

    try {
      const res = await fetch(api.kbQAStreamUrl(), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: question,
          mode,
          top_k: 15,
          top_n: 5,
          platform_filter: platform || null,
          retrieval_mode: qaRetrievalMode,
          rerank_mode: qaRerankMode,
          rewrite_query: true,
        }),
      })

      if (!res.ok || !res.body) throw new Error('流式请求失败')

      const reader = res.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffer = ''
      let raw = ''
      let sources: SourceRef[] = []

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        const blocks = buffer.split('\n\n')
        buffer = blocks.pop() || ''

        for (const block of blocks) {
          const line = block.split('\n').find((l) => l.startsWith('data:'))
          if (!line) continue

          let payload: SSEEvent
          try {
            payload = JSON.parse(line.slice(5).trim())
          } catch {
            continue
          }

          if (payload.type === 'token') {
            raw += payload.content || ''
            setMessages((prev) => {
              const updated = [...prev]
              updated[updated.length - 1] = { ...updated[updated.length - 1], content: raw }
              return updated
            })
          } else if (payload.type === 'sources') {
            sources = payload.sources || []
          } else if (payload.type === 'done') {
            break
          } else if (payload.type === 'error') {
            raw += `\n\n错误: ${payload.message || '未知错误'}`
          }
        }
      }

      setMessages((prev) => {
        const updated = [...prev]
        updated[updated.length - 1] = {
          ...updated[updated.length - 1],
          content: raw,
          sources,
        }
        return updated
      })
    } catch (err: any) {
      setMessages((prev) => {
        const updated = [...prev]
        updated[updated.length - 1] = {
          ...updated[updated.length - 1],
          content: `请求失败: ${err.message || '未知错误'}`,
        }
        return updated
      })
    } finally {
      setSending(false)
    }
  }

  const handleKeyDown = (e: KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      handleSend()
    }
  }

  const handleSemanticSearch = async (query: string) => {
    setSearchLoading(true)
    setHasSearched(true)
    setSearchError(null)
    try {
      const data = await api.kbSearch(query, {
        topK: 10,
        platformFilter: searchPlatform || undefined,
        retrievalMode: searchRetrievalMode,
        rerankMode: searchRerankMode,
        includeDebug: true,
        rewriteQuery: true,
      })
      setSearchResults(data.hits || [])
      setSearchDebug(data.debug || null)
      setSearchRewrite({
        rewritten: data.rewritten_query,
        applied: data.rewrite_applied,
        strategy: data.rewrite_strategy,
      })
    } catch {
      setSearchResults([])
      setSearchDebug(null)
      setSearchRewrite(null)
      setSearchError('检索请求失败。请确认后端正在运行，且知识库已经建立。')
    } finally {
      setSearchLoading(false)
    }
  }

  const qaEmptyState = (() => {
    if (kbStatusError) {
      return {
        title: '知识库状态读取失败',
        detail: kbStatusError,
      }
    }
    if (kbStatus?.total_chats === 0) {
      return {
        title: '还没有归档聊天记录',
        detail: '先归档至少一条 AI 聊天，再回来做语义搜索和知识库问答。',
      }
    }
    if (kbStatus && kbStatus.total_chats > 0 && kbStatus.total_chunks === 0) {
      return {
        title: '聊天已归档，但知识库未建立',
        detail: '请先到“管理”页执行一次全量重建索引。',
      }
    }
    return null
  })()

  return (
    <div className="space-y-4">
      {/* 子标签 */}
      <div className="flex gap-2">
        {(['qa', 'manage', 'semantic'] as SubTab[]).map((tab) => (
          <button
            key={tab}
            onClick={() => setSubTab(tab)}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition ${
              subTab === tab
                ? 'bg-[#3b82f6]/20 border border-[#3b82f6] text-[#93c5fd]'
                : 'bg-[#1f2937] border border-[#374151] text-[#9ca3af] hover:text-[#e5e7eb]'
            }`}
          >
            {tab === 'qa' ? '💬 问答' : tab === 'manage' ? '⚙️ 管理' : '🔍 语义搜索'}
          </button>
        ))}
      </div>

      {/* 问答页面 */}
      {subTab === 'qa' && (
        <div className="bg-[#111827] border border-[#374151] rounded-xl flex flex-col" style={{ height: '72vh' }}>
          {/* 设置栏 */}
          <div className="flex items-center gap-3 px-4 py-2 border-b border-[#374151] text-xs">
            <select
              value={platform}
              onChange={(e) => setPlatform(e.target.value)}
              className="bg-[#1f2937] border border-[#374151] rounded px-2 py-1 text-[#e5e7eb]"
            >
              <option value="">全部平台</option>
              {['ChatGPT', 'Claude', 'Gemini', 'DeepSeek', 'Poe'].map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
            <label className="flex items-center gap-1 text-[#9ca3af]">
              <input
                type="radio"
                checked={mode === 'concise'}
                onChange={() => setMode('concise')}
                className="accent-[#3b82f6]"
              />
              简洁
            </label>
            <label className="flex items-center gap-1 text-[#9ca3af]">
              <input
                type="radio"
                checked={mode === 'detailed'}
                onChange={() => setMode('detailed')}
                className="accent-[#3b82f6]"
              />
              详细
            </label>
            <select
              value={qaRetrievalMode}
              onChange={(e) => setQaRetrievalMode(e.target.value as 'hybrid' | 'vector' | 'keyword' | 'entity' | 'mix')}
              className="bg-[#1f2937] border border-[#374151] rounded px-2 py-1 text-[#e5e7eb]"
            >
              <option value="hybrid">Hybrid</option>
              <option value="vector">Vector</option>
              <option value="keyword">Keyword</option>
              <option value="entity">Entity</option>
              <option value="mix">Mix</option>
            </select>
            <select
              value={qaRerankMode}
              onChange={(e) => setQaRerankMode(e.target.value as RerankMode)}
              className="bg-[#1f2937] border border-[#374151] rounded px-2 py-1 text-[#e5e7eb]"
            >
              <option value="auto">Rerank Auto</option>
              <option value="off">Rerank Off</option>
              <option value="on">Rerank On</option>
            </select>
          </div>

          {/* 消息列表 */}
          <div ref={listRef} className="flex-1 overflow-y-auto p-4 space-y-3">
            {messages.length === 0 && (
              <div className="text-center text-[#6b7280] py-20">
                <div className="text-4xl mb-3">{qaEmptyState ? '📦' : '🔍'}</div>
                <div className="text-sm">{qaEmptyState?.title || '输入你的问题，比如：'}</div>
                <div className="text-xs mt-2 space-y-1">
                  {qaEmptyState ? (
                    <div className="text-[#fbbf24]">{qaEmptyState.detail}</div>
                  ) : (
                    <>
                      <div className="text-[#3b82f6]">"我之前讨论过哪些关于机器学习的话题？"</div>
                      <div className="text-[#3b82f6]">"上次关于 Python 异步编程的对话内容"</div>
                      <div className="text-[#3b82f6]">"帮我找一下之前关于数据库设计的建议"</div>
                    </>
                  )}
                </div>
              </div>
            )}
            {messages.map((msg, i) => (
              <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div
                  className={`max-w-[80%] rounded-xl px-4 py-2.5 text-sm leading-relaxed ${
                    msg.role === 'user'
                      ? 'bg-[#1d4ed8] text-white'
                      : 'bg-[#1f2937] border border-[#374151] text-[#e5e7eb]'
                  }`}
                >
                  {msg.role === 'ai' ? (
                    <>
                      <div className="prose prose-invert prose-sm max-w-none
                        prose-p:my-1.5 prose-p:leading-relaxed
                        prose-headings:text-[#e5e7eb] prose-headings:font-semibold prose-headings:mt-3 prose-headings:mb-1
                        prose-code:text-[#f472b6] prose-code:bg-[#0f172a] prose-code:px-1 prose-code:rounded prose-code:text-xs
                        prose-pre:bg-[#0f172a] prose-pre:border prose-pre:border-[#374151] prose-pre:rounded-lg prose-pre:text-xs prose-pre:my-2
                        prose-ul:my-1.5 prose-li:my-0.5
                        prose-ol:my-1.5
                        prose-blockquote:border-[#3b82f6] prose-blockquote:text-[#9ca3af] prose-blockquote:my-2
                        prose-strong:text-[#e5e7eb]
                        prose-table:text-xs prose-th:bg-[#0f172a] prose-th:text-[#9ca3af] prose-td:border-[#374151]
                      ">
                        <ReactMarkdown rehypePlugins={[rehypeHighlight]}>
                          {msg.content || ''}
                        </ReactMarkdown>
                      </div>
                      {sending && i === messages.length - 1 && (
                        <span className="inline-block w-2 h-4 bg-[#3b82f6] animate-pulse rounded-sm ml-0.5" />
                      )}
                      {msg.sources && msg.sources.length > 0 && (
                        <div className="border-t border-dashed border-[#374151] pt-2 mt-3">
                          <div className="text-xs text-[#9ca3af] mb-1.5 font-medium">📎 来源引用</div>
                          {msg.sources.map((s) => {
                            const score = Math.round((s.rerank_score ?? s.score) * 100)
                            return (
                              <div key={s.source_id} className="bg-[#0f172a] border border-[#374151] hover:border-[#3b82f6]/50 rounded-lg p-2 mt-1 text-xs transition">
                                <div className="font-medium text-[#e5e7eb]">{s.platform} · {s.title}</div>
                                <div className="flex items-center gap-2 mt-1">
                                  <span className="text-[#9ca3af]">相关度</span>
                                  <div className="flex-1 h-1.5 bg-[#1e293b] rounded-full overflow-hidden max-w-[100px]">
                                    <div
                                      className="h-full rounded-full"
                                      style={{
                                        width: `${score}%`,
                                        background: score > 70 ? '#22c55e' : score > 40 ? '#f59e0b' : '#ef4444',
                                      }}
                                    />
                                  </div>
                                  <span className="text-[#9ca3af]">{score}%</span>
                                </div>
                                {s.excerpt && (
                                  <div className="text-[#6b7280] mt-1 line-clamp-2">{s.excerpt}</div>
                                )}
                              </div>
                            )
                          })}
                        </div>
                      )}
                    </>
                  ) : (
                    msg.content
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* 输入区 */}
          <div className="border-t border-[#374151] p-3 flex gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={2}
              placeholder="输入你的问题，Ctrl+Enter 发送..."
              className="flex-1 bg-[#1f2937] border border-[#374151] rounded-lg px-3 py-2 text-sm text-[#e5e7eb] placeholder:text-[#6b7280] resize-none focus:outline-none focus:border-[#3b82f6]"
            />
            <button
              onClick={handleSend}
              disabled={sending || !input.trim() || (!!kbStatus && kbStatus.total_chunks === 0)}
              className="px-4 py-2 bg-[#3b82f6] hover:bg-[#2563eb] disabled:opacity-50 text-white rounded-lg text-sm font-medium self-end transition"
            >
              {sending ? '生成中...' : '发送'}
            </button>
          </div>
        </div>
      )}

      {/* 管理页面 */}
      {subTab === 'manage' && <ReindexPanel />}

      {/* 语义搜索 */}
      {subTab === 'semantic' && (
        <div className="bg-[#111827] border border-[#374151] rounded-xl p-4 space-y-4">
          <div className="space-y-3">
            <SearchBox
              placeholder="语义搜索，例如：异步编程的最佳实践"
              onSearch={handleSemanticSearch}
              loading={searchLoading}
              platformFilter={searchPlatform}
              onPlatformChange={setSearchPlatform}
            />
            <div className="flex items-center gap-2 text-xs text-[#9ca3af]">
              <span>检索模式</span>
              <select
                value={searchRetrievalMode}
                onChange={(e) => setSearchRetrievalMode(e.target.value as 'hybrid' | 'vector' | 'keyword' | 'entity' | 'mix')}
                className="bg-[#1f2937] border border-[#374151] rounded px-2 py-1 text-[#e5e7eb]"
              >
                <option value="hybrid">Hybrid</option>
                <option value="vector">Vector</option>
                <option value="keyword">Keyword</option>
                <option value="entity">Entity</option>
                <option value="mix">Mix</option>
              </select>
              <select
                value={searchRerankMode}
                onChange={(e) => setSearchRerankMode(e.target.value as RerankMode)}
                className="bg-[#1f2937] border border-[#374151] rounded px-2 py-1 text-[#e5e7eb]"
              >
                <option value="auto">Rerank Auto</option>
                <option value="off">Rerank Off</option>
                <option value="on">Rerank On</option>
              </select>
            </div>
          </div>

          {searchRewrite && (
            <div className="bg-[#0f172a] border border-[#334155] rounded-xl p-3 text-xs space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[#cbd5e1] font-medium">检索改写</span>
                <span className={`px-2 py-0.5 rounded ${searchRewrite.applied ? 'bg-[#1d4ed8]/20 text-[#93c5fd]' : 'bg-[#334155] text-[#94a3b8]'}`}>
                  {searchRewrite.strategy || 'identity'}
                </span>
              </div>
              <div className="text-[#94a3b8]">
                {searchRewrite.applied ? '已改写为更适合检索的独立查询。' : '本次查询未做实质改写。'}
              </div>
              {searchRewrite.rewritten && (
                <div className="text-[#e2e8f0]">{searchRewrite.rewritten}</div>
              )}
            </div>
          )}

          {searchError && (
            <div className="bg-[#7f1d1d]/20 border border-[#7f1d1d] rounded-xl p-3 text-xs text-[#fecaca]">
              <div className="font-medium mb-1">检索失败</div>
              <div>{searchError}</div>
            </div>
          )}

          {searchDebug && (searchDebug.query_entities?.length || searchDebug.expanded_entities?.length) && (
            <div className="bg-[#0f172a] border border-[#334155] rounded-xl p-3 text-xs space-y-2">
              <div className="text-[#cbd5e1] font-medium">实体扩展</div>
              {searchDebug.query_entities && searchDebug.query_entities.length > 0 && (
                <div className="text-[#94a3b8]">
                  Query entities: <span className="text-[#e2e8f0]">{searchDebug.query_entities.join(', ')}</span>
                </div>
              )}
              {searchDebug.expanded_entities && searchDebug.expanded_entities.length > 0 && (
                <div className="text-[#94a3b8]">
                  Expanded entities: <span className="text-[#e2e8f0]">{searchDebug.expanded_entities.join(', ')}</span>
                </div>
              )}
            </div>
          )}

          {searchDebug && (
            <div className="bg-[#0b1220] border border-[#243244] rounded-xl p-4 space-y-4">
              {searchDebug.query_analysis && (
                <div className="grid gap-3 lg:grid-cols-2">
                  <div className="rounded-xl border border-[#334155] bg-[#0f172a] px-3 py-3 text-xs space-y-2">
                    <div className="text-[#cbd5e1] font-medium">Query Analysis</div>
                    <div className="text-[#94a3b8]">
                      type: <span className="text-[#e2e8f0]">{searchDebug.query_analysis.query_type}</span>
                      {' · '}
                      scope: <span className="text-[#e2e8f0]">{searchDebug.analysis_scope || 'n/a'}</span>
                    </div>
                    <div className="text-[#94a3b8]">
                      rewrite: <span className="text-[#e2e8f0]">{searchDebug.query_analysis.enable_rewrite ? 'on' : 'off'}</span>
                      {' · '}
                      rerank: <span className="text-[#e2e8f0]">{searchDebug.query_analysis.enable_rerank ? 'on' : 'off'}</span>
                      {' · '}
                      graph: <span className="text-[#e2e8f0]">{searchDebug.query_analysis.enable_graph ? 'on' : 'off'}</span>
                    </div>
                    {searchDebug.query_analysis.reasons.length > 0 && (
                      <div className="flex flex-wrap gap-2">
                        {searchDebug.query_analysis.reasons.map((reason) => (
                          <span
                            key={reason}
                            className="inline-flex items-center rounded-full border border-[#334155] bg-[#111827] px-2 py-0.5 text-[11px] text-[#cbd5e1]"
                          >
                            {reason}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                  <div className="rounded-xl border border-[#334155] bg-[#0f172a] px-3 py-3 text-xs space-y-2">
                    <div className="text-[#cbd5e1] font-medium">Graph Routing</div>
                    <div className="text-[#94a3b8]">
                      requested: <span className="text-[#e2e8f0]">{searchDebug.graph_requested_mode || 'auto'}</span>
                      {' · '}
                      effective: <span className="text-[#e2e8f0]">{searchDebug.graph_effective_mode || 'off'}</span>
                    </div>
                    <div className="text-[#94a3b8]">
                      routed: <span className="text-[#e2e8f0]">{searchDebug.graph_routed ? 'yes' : 'no'}</span>
                      {' · '}
                      graph hits: <span className="text-[#e2e8f0]">{searchDebug.graph_hit_count}</span>
                    </div>
                  </div>
                </div>
              )}
              <div className="flex flex-wrap gap-2 text-xs">
                {[
                  ['dense', searchDebug.dense_count ?? '-'],
                  ['keyword', searchDebug.keyword_count ?? '-'],
                  ['entity', searchDebug.entity_count ?? '-'],
                  ['candidate', searchDebug.candidate_count],
                  ['final', searchDebug.final_count],
                  ['rerank', searchDebug.rerank_applied ? 'on' : searchDebug.rerank_reason || 'off'],
                  ['cache', searchDebug.cache_hit ? 'hit' : 'miss'],
                ].map(([label, value]) => (
                  <div key={label} className="px-2 py-1 rounded bg-[#162033] text-[#cbd5e1]">
                    {label}: {value}
                  </div>
                  ))}
              </div>
              <div className={`rounded-xl border px-3 py-2 text-xs ${rerankStatusTone(searchDebug.rerank_status)}`}>
                <div className="font-medium mb-1">
                  Rerank {rerankStatusLabel(searchDebug.rerank_status)}
                </div>
                <div>{searchDebug.rerank_message || '本次检索未执行 rerank。'}</div>
              </div>
              <div className="text-xs text-[#94a3b8]">
                rerank requested: <span className="text-[#e2e8f0]">{searchDebug.rerank_requested_mode || 'auto'}</span>
                {' · '}
                effective: <span className="text-[#e2e8f0]">{searchDebug.rerank_effective_mode || 'off'}</span>
                {' · '}
                reason: <span className="text-[#e2e8f0]">{searchDebug.rerank_reason || 'disabled'}</span>
                {' · '}
                elapsed: <span className="text-[#e2e8f0]">{searchDebug.rerank_elapsed_ms ?? 0}ms</span>
                {' · '}
                candidates: <span className="text-[#e2e8f0]">{searchDebug.rerank_candidate_count ?? 0}/{searchDebug.rerank_candidate_limit ?? 0}</span>
              </div>
              <div className="grid gap-4 lg:grid-cols-3">
                {[
                  { title: 'Dense Top', hits: searchDebug.dense_hits },
                  { title: 'Keyword Top', hits: searchDebug.keyword_hits },
                  { title: 'Entity Top', hits: searchDebug.entity_hits },
                  { title: 'Graph Top', hits: searchDebug.graph_hits || [] },
                  { title: 'Final Top', hits: searchDebug.final_hits },
                ].map(({ title, hits }) => (
                  <div key={title} className="space-y-2">
                    <div className="text-xs font-medium text-[#93c5fd]">{title}</div>
                    {hits.length === 0 ? (
                      <div className="text-xs text-[#64748b]">无结果</div>
                    ) : (
                      hits.slice(0, 3).map((hit, i) => (
                        <SourcePreview key={`${title}-${hit.chunk_id}-${i}`} hit={hit} index={i} />
                      ))
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {searchResults.length === 0 && !searchLoading && (
            <div className="text-center text-[#6b7280] py-10 text-sm space-y-2">
              {kbStatus?.total_chats === 0 ? (
                <>
                  <div className="text-[#fbbf24]">还没有归档聊天，当前无法进行语义检索。</div>
                  <div>请先归档一条聊天记录。</div>
                </>
              ) : kbStatus && kbStatus.total_chats > 0 && kbStatus.total_chunks === 0 ? (
                <>
                  <div className="text-[#fbbf24]">聊天已归档，但知识库还没有建立。</div>
                  <div>请先到“管理”页执行重建索引。</div>
                </>
              ) : hasSearched ? (
                <>
                  <div className="text-[#e5e7eb]">没有找到相关结果。</div>
                  <div>可以尝试换一个说法、切换检索模式，或关闭 rerank 再试。</div>
                </>
              ) : (
                <div>输入关键词进行语义检索</div>
              )}
            </div>
          )}

          <div className="space-y-3">
            {searchResults.map((hit, i) => (
              <SourcePreview key={hit.chunk_id || i} hit={hit} index={i} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
