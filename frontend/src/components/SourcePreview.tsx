import type { RetrievalHit } from '../types'

interface Props {
  hit: RetrievalHit
  index: number
  onViewChat?: (docId: string) => void
}

const PLATFORM_COLORS: Record<string, string> = {
  ChatGPT: '#10a37f',
  Claude: '#d97706',
  Gemini: '#4285f4',
  DeepSeek: '#3b82f6',
  Poe: '#8b5cf6',
}

export default function SourcePreview({ hit, index, onViewChat }: Props) {
  const score = Math.round((hit.rerank_score ?? hit.score) * 100)
  const color = PLATFORM_COLORS[hit.platform] || '#6b7280'
  const text = hit.excerpt.replace(/\n+/g, ' ').slice(0, 200)

  return (
    <div className="bg-[#1f2937] border border-[#374151] rounded-xl p-4 hover:border-[#3b82f6]/40 transition">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span
            className="w-2 h-2 rounded-full"
            style={{ backgroundColor: color }}
          />
          <span className="font-medium text-sm">{hit.platform}</span>
          <span className="text-[#9ca3af] text-xs">·</span>
          <span className="text-sm text-[#e5e7eb]">{hit.title}</span>
        </div>
        <span className="text-xs text-[#9ca3af]">#{index + 1}</span>
      </div>

      <div className="flex items-center gap-2 mb-2 text-xs">
        <span className="text-[#9ca3af]">相关度</span>
        <div className="w-20 h-1.5 bg-[#0f172a] rounded-full overflow-hidden">
          <div
            className="h-full rounded-full"
            style={{
              width: `${score}%`,
              background: score > 70 ? '#22c55e' : score > 40 ? '#f59e0b' : '#ef4444',
            }}
          />
        </div>
        <span className={score > 70 ? 'text-[#22c55e]' : score > 40 ? 'text-[#f59e0b]' : 'text-[#ef4444]'}>
          {score}%
        </span>
        {hit.created_at && (
          <>
            <span className="text-[#374151]">|</span>
            <span className="text-[#6b7280]">{hit.created_at}</span>
          </>
        )}
      </div>

      <p className="text-xs text-[#9ca3af] leading-relaxed line-clamp-3">{text}...</p>

      {onViewChat && (
        <button
          onClick={() => onViewChat(hit.doc_id)}
          className="mt-2 text-xs text-[#3b82f6] hover:text-[#60a5fa] transition"
        >
          查看完整对话 →
        </button>
      )}
    </div>
  )
}
