import type { Citation, SourceRef } from '../types'

interface Props {
  citations: Citation[]
  sources: SourceRef[]
  onSourceClick?: (sourceId: string) => void
}

const PLATFORM_ICONS: Record<string, string> = {
  ChatGPT: '🤖',
  Claude: '🧠',
  Gemini: '✨',
  DeepSeek: '🔍',
  Poe: '💬',
}

export default function CitationList({ citations, sources, onSourceClick }: Props) {
  if (sources.length === 0) return null

  return (
    <div className="border-t border-dashed border-[#374151] pt-3 mt-3">
      <div className="text-xs text-[#9ca3af] mb-2 font-medium">📎 来源引用</div>
      <div className="space-y-2">
        {sources.map((src) => {
          const score = Math.round((src.rerank_score ?? src.score) * 100)
          const icon = PLATFORM_ICONS[src.platform] || '📄'
          const cited = citations.find((c) => c.source_id === src.source_id)

          return (
            <div
              key={src.source_id}
              className="bg-[#0f172a] border border-[#374151] rounded-lg p-3 text-xs hover:border-[#3b82f6]/50 transition cursor-pointer"
              onClick={() => onSourceClick?.(src.source_id)}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="font-medium text-[#e5e7eb]">
                  {icon} {src.platform} · {src.title}
                </span>
                <span className="text-[#9ca3af]">[Source {src.source_id}]</span>
              </div>

              <div className="flex items-center gap-2 mb-1">
                <span className="text-[#9ca3af]">相关度</span>
                <div className="flex-1 h-1.5 bg-[#1e293b] rounded-full overflow-hidden max-w-[120px]">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{
                      width: `${score}%`,
                      background: score > 70 ? '#22c55e' : score > 40 ? '#f59e0b' : '#ef4444',
                    }}
                  />
                </div>
                <span className="text-[#9ca3af]">{score}%</span>
              </div>

              {cited && (
                <div className="text-[#9ca3af] italic mt-1">引用理由: {cited.reason}</div>
              )}

              {src.excerpt && (
                <div className="text-[#6b7280] mt-1 line-clamp-2">{src.excerpt}</div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
