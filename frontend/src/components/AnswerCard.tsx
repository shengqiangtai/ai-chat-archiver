import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import 'highlight.js/styles/github-dark.css'
import type { SourceRef, Citation, RetrievalDebug } from '../types'
import CitationList from './CitationList'

interface Props {
  answer: string
  citations: Citation[]
  uncertainty: string | null
  sources: SourceRef[]
  debug?: RetrievalDebug
  isStreaming?: boolean
  onSourceClick?: (sourceId: string) => void
}

export default function AnswerCard({ answer, citations, uncertainty, sources, debug, isStreaming, onSourceClick }: Props) {
  return (
    <div className="bg-[#1f2937] border border-[#374151] rounded-xl p-4 space-y-3">
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
          {answer || ''}
        </ReactMarkdown>
      </div>

      {isStreaming && (
        <span className="inline-block w-2 h-4 bg-[#3b82f6] animate-pulse rounded-sm" />
      )}

      {uncertainty && (
        <div className="text-xs text-[#f59e0b] bg-[#f59e0b]/10 border border-[#f59e0b]/20 rounded-lg px-3 py-2">
          ⚠️ {uncertainty}
        </div>
      )}

      {debug?.grounding && (
        <div className="text-xs text-[#cbd5e1] bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 space-y-1">
          <div className="font-medium text-[#e2e8f0]">Grounding</div>
          <div>
            supported: <span className="text-[#93c5fd]">{debug.grounding.supported ? 'yes' : 'no'}</span>
            {' · '}
            support rate: <span className="text-[#93c5fd]">{Math.round(debug.grounding.support_rate * 100)}%</span>
            {' · '}
            message: <span className="text-[#93c5fd]">{debug.grounding.message}</span>
          </div>
        </div>
      )}

      {sources.length > 0 && (
        <CitationList
          citations={citations}
          sources={sources}
          onSourceClick={onSourceClick}
        />
      )}
    </div>
  )
}
