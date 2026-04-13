import { useState, useEffect, useCallback } from 'react'
import { api } from './api/client'
import KnowledgeBaseQA from './pages/KnowledgeBaseQA'
import DocumentDetail from './pages/DocumentDetail'
import type { ChatItem } from './types'

type MainTab = 'chats' | 'kb'

export default function App() {
  const [tab, setTab] = useState<MainTab>('chats')
  const [stats, setStats] = useState<any>(null)
  const [chats, setChats] = useState<ChatItem[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [platform, setPlatform] = useState('')
  const [searchQuery, setSearchQuery] = useState('')

  const loadStats = useCallback(async () => {
    try {
      setStats(await api.getStats())
    } catch { /* ignore */ }
  }, [])

  const loadChats = useCallback(async () => {
    try {
      const data = await api.getChats(platform || undefined)
      setChats(data.chats || [])
    } catch { /* ignore */ }
  }, [platform])

  useEffect(() => {
    loadStats()
    loadChats()
  }, [loadStats, loadChats])

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      loadChats()
      return
    }
    try {
      const data = await api.search(searchQuery, platform || undefined)
      setChats((data.results || []).map((x: any) => ({
        id: x.id,
        title: x.title,
        platform: x.platform,
        model: x.model,
        saved_at: x.saved_at,
        created_at: x.created_at || '',
        tags: x.tags || [],
        url: x.url,
        file_path: x.file_path || '',
      })))
    } catch { /* ignore */ }
  }

  const handleRefresh = () => {
    loadStats()
    loadChats()
  }

  return (
    <div className="max-w-[1440px] mx-auto px-4 py-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-bold text-[#e5e7eb]">AI Chat Archiver</h1>
        <button
          onClick={handleRefresh}
          className="px-3 py-1.5 bg-[#3b82f6] hover:bg-[#2563eb] text-white rounded-lg text-sm font-medium transition"
        >
          刷新
        </button>
      </div>

      {/* 主 Tab */}
      <div className="flex gap-2 mb-4">
        <button
          onClick={() => setTab('chats')}
          className={`px-4 py-2 rounded-xl text-sm font-semibold transition ${
            tab === 'chats'
              ? 'bg-[#3b82f6] border border-[#3b82f6] text-white'
              : 'bg-[#111827] border border-[#374151] text-[#e5e7eb] hover:border-[#3b82f6]/50'
          }`}
        >
          聊天记录
        </button>
        <button
          onClick={() => setTab('kb')}
          className={`px-4 py-2 rounded-xl text-sm font-semibold transition ${
            tab === 'kb'
              ? 'bg-[#3b82f6] border border-[#3b82f6] text-white'
              : 'bg-[#111827] border border-[#374151] text-[#e5e7eb] hover:border-[#3b82f6]/50'
          }`}
        >
          🔍 知识库
        </button>
      </div>

      {/* 聊天记录页 */}
      {tab === 'chats' && (
        <>
          {/* 统计卡片 */}
          {stats && (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3 mb-4">
              <StatCard label="总对话数" value={stats.total || 0} />
              {Object.entries(stats.by_platform || {}).map(([p, c]) => (
                <StatCard key={p} label={p} value={c as number} />
              ))}
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-[380px_1fr] gap-4">
            {/* 列表侧栏 */}
            <div className="bg-[#111827] border border-[#374151] rounded-xl p-3 space-y-3">
              <div className="flex gap-2">
                <select
                  value={platform}
                  onChange={(e) => setPlatform(e.target.value)}
                  className="bg-[#1f2937] border border-[#374151] rounded-lg px-2 py-1.5 text-sm text-[#e5e7eb] flex-1"
                >
                  <option value="">全部平台</option>
                  {['ChatGPT', 'Claude', 'Gemini', 'DeepSeek', 'Poe'].map((p) => (
                    <option key={p} value={p}>{p}</option>
                  ))}
                </select>
                <button
                  onClick={loadChats}
                  className="px-3 py-1.5 bg-[#1f2937] border border-[#374151] hover:border-[#3b82f6] rounded-lg text-sm transition"
                >
                  加载
                </button>
              </div>

              <div className="flex gap-2">
                <input
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                  placeholder="全文搜索..."
                  className="flex-1 bg-[#1f2937] border border-[#374151] rounded-lg px-3 py-1.5 text-sm text-[#e5e7eb] placeholder:text-[#6b7280] focus:outline-none focus:border-[#3b82f6]"
                />
                <button
                  onClick={handleSearch}
                  className="px-3 py-1.5 bg-[#3b82f6] hover:bg-[#2563eb] text-white rounded-lg text-sm font-medium transition"
                >
                  搜索
                </button>
              </div>

              <div className="max-h-[70vh] overflow-y-auto space-y-2">
                {chats.length === 0 && (
                  <div className="text-sm text-[#6b7280] text-center py-8">暂无记录</div>
                )}
                {chats.map((chat) => (
                  <div
                    key={chat.id}
                    onClick={() => setSelectedId(chat.id)}
                    className={`p-3 rounded-xl cursor-pointer transition ${
                      chat.id === selectedId
                        ? 'bg-[#1f2937] border border-[#3b82f6]'
                        : 'bg-[#1f2937] border border-[#374151] hover:border-[#3b82f6]/40'
                    }`}
                  >
                    <div className="font-medium text-sm truncate">{chat.title}</div>
                    <div className="text-xs text-[#9ca3af] mt-1">
                      {chat.platform} · {chat.model || 'N/A'} · {chat.message_count || 0}条
                    </div>
                    <div className="text-xs text-[#6b7280] mt-0.5">
                      {formatDate(chat.saved_at || chat.created_at)}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* 详情 */}
            <DocumentDetail chatId={selectedId} onDelete={handleRefresh} />
          </div>
        </>
      )}

      {/* 知识库页 */}
      {tab === 'kb' && <KnowledgeBaseQA />}
    </div>
  )
}

function StatCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="bg-[#1f2937] border border-[#374151] rounded-xl p-3">
      <div className="text-xs text-[#9ca3af]">{label}</div>
      <div className="text-xl font-bold mt-1">{value}</div>
    </div>
  )
}

function formatDate(value: string) {
  if (!value) return ''
  const d = new Date(value)
  if (isNaN(d.getTime())) return value
  return d.toLocaleString()
}
