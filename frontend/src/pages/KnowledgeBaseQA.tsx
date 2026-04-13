import { useState, useRef, useEffect, useCallback, type KeyboardEvent } from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import 'highlight.js/styles/github-dark.css'
import { api } from '../api/client'
import type { SourceRef, SSEEvent } from '../types'
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

export default function KnowledgeBaseQA() {
  const [subTab, setSubTab] = useState<SubTab>('qa')
  const [messages, setMessages] = useState<QAMessage[]>([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [platform, setPlatform] = useState('')
  const [mode, setMode] = useState<'concise' | 'detailed'>('concise')
  const listRef = useRef<HTMLDivElement>(null)

  // 语义搜索状态
  const [searchResults, setSearchResults] = useState<any[]>([])
  const [searchLoading, setSearchLoading] = useState(false)
  const [searchPlatform, setSearchPlatform] = useState('')

  const scrollToBottom = useCallback(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight
    }
  }, [])

  useEffect(() => { scrollToBottom() }, [messages, scrollToBottom])

  const handleSend = async () => {
    const question = input.trim()
    if (!question || sending) return

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
    try {
      const data = await api.kbSearch(query, 10, searchPlatform || undefined)
      setSearchResults(data.hits || [])
    } catch {
      setSearchResults([])
    } finally {
      setSearchLoading(false)
    }
  }

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
          </div>

          {/* 消息列表 */}
          <div ref={listRef} className="flex-1 overflow-y-auto p-4 space-y-3">
            {messages.length === 0 && (
              <div className="text-center text-[#6b7280] py-20">
                <div className="text-4xl mb-3">🔍</div>
                <div className="text-sm">输入你的问题，比如：</div>
                <div className="text-xs mt-2 space-y-1">
                  <div className="text-[#3b82f6]">"我之前讨论过哪些关于机器学习的话题？"</div>
                  <div className="text-[#3b82f6]">"上次关于 Python 异步编程的对话内容"</div>
                  <div className="text-[#3b82f6]">"帮我找一下之前关于数据库设计的建议"</div>
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
              disabled={sending || !input.trim()}
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
          <SearchBox
            placeholder="语义搜索，例如：异步编程的最佳实践"
            onSearch={handleSemanticSearch}
            loading={searchLoading}
            platformFilter={searchPlatform}
            onPlatformChange={setSearchPlatform}
          />

          {searchResults.length === 0 && !searchLoading && (
            <div className="text-center text-[#6b7280] py-10 text-sm">输入关键词进行语义检索</div>
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
