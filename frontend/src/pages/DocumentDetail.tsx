import { useEffect, useState } from 'react'
import { marked } from 'marked'
import { api } from '../api/client'
import type { ChatDetail } from '../types'

interface Props {
  chatId: string | null
  onDelete?: () => void
}

export default function DocumentDetail({ chatId, onDelete }: Props) {
  const [chat, setChat] = useState<ChatDetail | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!chatId) {
      setChat(null)
      return
    }
    setLoading(true)
    api.getChat(chatId)
      .then(setChat)
      .catch(() => setChat(null))
      .finally(() => setLoading(false))
  }, [chatId])

  const handleDelete = async () => {
    if (!chatId || !window.confirm('确认删除该聊天记录？')) return
    try {
      await api.deleteChat(chatId)
      setChat(null)
      onDelete?.()
    } catch {
      // ignore
    }
  }

  if (!chatId) {
    return (
      <div className="bg-[#111827] border border-[#374151] rounded-xl p-6 flex items-center justify-center text-[#6b7280] min-h-[300px]">
        请选择左侧聊天记录
      </div>
    )
  }

  if (loading) {
    return (
      <div className="bg-[#111827] border border-[#374151] rounded-xl p-6 flex items-center justify-center min-h-[300px]">
        <div className="animate-pulse text-[#6b7280]">加载中...</div>
      </div>
    )
  }

  if (!chat) return null

  const meta = (chat.meta || {}) as Record<string, any>

  return (
    <div className="bg-[#111827] border border-[#374151] rounded-xl p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold">{meta.title || chatId}</h2>
          <div className="text-xs text-[#9ca3af] mt-1">
            {meta.platform || ''} · {meta.model || 'N/A'} · {meta.saved_at || meta.created_at || ''}
          </div>
        </div>
        <button
          onClick={handleDelete}
          className="px-3 py-1 bg-[#7f1d1d] hover:bg-[#991b1b] text-[#fca5a5] rounded-lg text-xs transition"
        >
          删除
        </button>
      </div>

      <div
        className="prose prose-invert prose-sm max-w-none max-h-[75vh] overflow-y-auto leading-relaxed"
        dangerouslySetInnerHTML={{ __html: marked.parse(chat.content || '') as string }}
      />
    </div>
  )
}
