import { useState, type KeyboardEvent } from 'react'

interface Props {
  placeholder?: string
  onSearch: (query: string) => void
  loading?: boolean
  platformFilter?: string
  onPlatformChange?: (platform: string) => void
}

const PLATFORMS = ['ChatGPT', 'Claude', 'Gemini', 'DeepSeek', 'Poe']

export default function SearchBox({ placeholder, onSearch, loading, platformFilter, onPlatformChange }: Props) {
  const [query, setQuery] = useState('')

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (query.trim()) onSearch(query.trim())
    }
  }

  return (
    <div className="flex gap-2 flex-wrap">
      {onPlatformChange && (
        <select
          value={platformFilter || ''}
          onChange={(e) => onPlatformChange(e.target.value)}
          className="bg-[#1f2937] border border-[#374151] rounded-lg px-3 py-2 text-sm text-[#e5e7eb] max-w-[160px]"
        >
          <option value="">全部平台</option>
          {PLATFORMS.map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>
      )}
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder || '输入搜索内容...'}
        className="flex-1 min-w-[200px] bg-[#1f2937] border border-[#374151] rounded-lg px-4 py-2 text-sm text-[#e5e7eb] placeholder:text-[#6b7280] focus:outline-none focus:border-[#3b82f6] transition"
      />
      <button
        onClick={() => query.trim() && onSearch(query.trim())}
        disabled={loading || !query.trim()}
        className="px-4 py-2 bg-[#3b82f6] hover:bg-[#2563eb] disabled:opacity-50 text-white rounded-lg text-sm font-medium transition"
      >
        {loading ? '检索中...' : '搜索'}
      </button>
    </div>
  )
}
