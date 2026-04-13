import { useState, useEffect, useCallback } from 'react'
import { api } from '../api/client'
import type { KbStatus, IndexProgress } from '../types'

interface LLMBackendInfo {
  available: boolean
  models?: string[]
  current_model?: string
  base_url?: string
}

interface LLMStatus {
  current_backend: string
  lmstudio: LLMBackendInfo
  transformers: LLMBackendInfo
}

export default function ReindexPanel() {
  const [kbStatus, setKbStatus] = useState<KbStatus | null>(null)
  const [llmStatus, setLLMStatus] = useState<LLMStatus | null>(null)
  const [taskId, setTaskId] = useState<string | null>(null)
  const [progress, setProgress] = useState<IndexProgress | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [selectedBackend, setSelectedBackend] = useState('lmstudio')
  const [selectedModel, setSelectedModel] = useState('')

  const addLog = useCallback((msg: string) => {
    const now = new Date().toLocaleTimeString()
    setLogs((prev) => [...prev, `[${now}] ${msg}`])
  }, [])

  const refreshStatus = useCallback(async () => {
    try {
      const [kb, llm] = await Promise.all([api.getKbStatus(), api.getLLMStatus()])
      setKbStatus(kb)
      setLLMStatus(llm)
      setSelectedBackend(llm.current_backend || 'lmstudio')
      const current = llm[llm.current_backend as keyof LLMStatus] as LLMBackendInfo | undefined
      if (current?.current_model) setSelectedModel(current.current_model)
    } catch (e: any) {
      addLog(`状态刷新失败: ${e.message}`)
    }
  }, [addLog])

  useEffect(() => { refreshStatus() }, [refreshStatus])

  useEffect(() => {
    if (!taskId) return
    const timer = setInterval(async () => {
      try {
        const p = await api.getIndexProgress(taskId)
        setProgress(p)
        addLog(`状态=${p.status}, 进度=${p.processed_files}/${p.total_files}, chunks=${p.total_chunks}`)
        if (p.status === 'done' || p.status === 'error') {
          clearInterval(timer)
          if (p.error) addLog(`错误: ${p.error}`)
          refreshStatus()
        }
      } catch {
        clearInterval(timer)
      }
    }, 2000)
    return () => clearInterval(timer)
  }, [taskId, addLog, refreshStatus])

  const startIndex = async (incremental: boolean) => {
    try {
      const data = incremental ? await api.reindexIncremental() : await api.reindex()
      setTaskId(data.task_id)
      setProgress(null)
      addLog(data.message || '索引任务已启动')
    } catch (e: any) {
      addLog(`启动失败: ${e.message}`)
    }
  }

  const switchBackend = async () => {
    try {
      await api.switchLLMBackend(selectedBackend, selectedModel || undefined)
      addLog(`已切换后端: ${selectedBackend}${selectedModel ? ` (${selectedModel})` : ''}`)
      refreshStatus()
    } catch (e: any) {
      addLog(`切换失败: ${e.message}`)
    }
  }

  const ratio = progress && progress.total_files > 0
    ? Math.round((progress.processed_files / progress.total_files) * 100)
    : 0

  const currentBackendInfo = llmStatus
    ? (llmStatus[selectedBackend as keyof LLMStatus] as LLMBackendInfo | undefined)
    : null
  const availableModels = currentBackendInfo?.models || []

  const backendLabel: Record<string, string> = {
    lmstudio: 'LM Studio',
    transformers: 'Transformers (本地)',
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* 知识库状态 */}
      <div className="bg-[#111827] border border-[#374151] rounded-xl p-4 space-y-3">
        <h3 className="text-base font-semibold">知识库状态</h3>
        {kbStatus ? (
          <div className="space-y-1 text-sm text-[#9ca3af]">
            <div>总聊天数: <span className="text-[#e5e7eb] font-medium">{kbStatus.total_chats}</span></div>
            <div>总 Chunk 数: <span className="text-[#e5e7eb] font-medium">{kbStatus.total_chunks}</span></div>
            <div>向量库大小: <span className="text-[#e5e7eb] font-medium">{Math.round(kbStatus.vectorstore_size_bytes / 1024)} KB</span></div>
            <div>最后索引: <span className="text-[#e5e7eb]">{kbStatus.last_index_time || '未执行'}</span></div>
            <div>状态: <span className={kbStatus.is_indexing ? 'text-[#f59e0b]' : 'text-[#22c55e]'}>{kbStatus.is_indexing ? '索引进行中' : '空闲'}</span></div>
          </div>
        ) : (
          <div className="text-sm text-[#6b7280]">加载中...</div>
        )}

        <div className="flex gap-2 flex-wrap">
          <button
            onClick={() => startIndex(false)}
            className="px-3 py-1.5 bg-[#f59e0b] hover:bg-[#d97706] text-[#111827] rounded-lg text-xs font-medium transition"
          >
            🔄 全量重建索引
          </button>
          <button
            onClick={() => startIndex(true)}
            className="px-3 py-1.5 bg-[#1f2937] border border-[#374151] hover:border-[#3b82f6] rounded-lg text-xs transition"
          >
            ➕ 增量更新
          </button>
        </div>

        <div>
          <div className="h-2 bg-[#0f172a] rounded-full overflow-hidden border border-[#374151]">
            <div
              className="h-full bg-[#22c55e] rounded-full transition-all duration-300"
              style={{ width: `${ratio}%` }}
            />
          </div>
          <div className="text-xs text-[#6b7280] mt-1">
            {progress
              ? `${progress.processed_files}/${progress.total_files} files · ${progress.total_chunks} chunks · ${progress.status}`
              : '未开始'}
          </div>
        </div>

        <div className="h-40 overflow-y-auto bg-[#0f172a] border border-[#374151] rounded-lg p-2 font-mono text-xs text-[#6b7280] whitespace-pre-wrap">
          {logs.length === 0 ? '暂无日志' : logs.join('\n')}
        </div>
      </div>

      {/* LLM 后端管理 */}
      <div className="bg-[#111827] border border-[#374151] rounded-xl p-4 space-y-3">
        <h3 className="text-base font-semibold">LLM 生成后端</h3>

        {llmStatus ? (
          <div className="space-y-2">
            {/* 后端状态概览 */}
            <div className="grid grid-cols-2 gap-2 text-xs">
              {(['lmstudio', 'transformers'] as const).map((b) => {
                const info = llmStatus[b] as LLMBackendInfo
                const isCurrent = llmStatus.current_backend === b
                return (
                  <div
                    key={b}
                    className={`p-2 rounded-lg border ${
                      isCurrent ? 'border-[#3b82f6] bg-[#3b82f6]/10' : 'border-[#374151] bg-[#0f172a]'
                    }`}
                  >
                    <div className="font-medium text-[#e5e7eb]">{backendLabel[b]}</div>
                    <div className={info.available ? 'text-[#22c55e]' : 'text-[#ef4444]'}>
                      {info.available ? '● 在线' : '○ 离线'}
                    </div>
                    {isCurrent && <div className="text-[#3b82f6] mt-0.5">当前使用</div>}
                  </div>
                )
              })}
            </div>

            {/* 后端切换 */}
            <div className="space-y-2">
              <label className="text-xs text-[#9ca3af]">选择后端</label>
              <select
                value={selectedBackend}
                onChange={(e) => {
                  setSelectedBackend(e.target.value)
                  setSelectedModel('')
                }}
                className="w-full bg-[#1f2937] border border-[#374151] rounded-lg px-3 py-2 text-sm text-[#e5e7eb]"
              >
                <option value="lmstudio">LM Studio（推荐 Mac）</option>
                <option value="transformers">Transformers 本地推理</option>
              </select>
            </div>

            {/* 模型选择（LM Studio / Ollama） */}
            {selectedBackend !== 'transformers' && (
              <div className="space-y-2">
                <label className="text-xs text-[#9ca3af]">
                  选择模型
                  {availableModels.length === 0 && ' (未检测到模型)'}
                </label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full bg-[#1f2937] border border-[#374151] rounded-lg px-3 py-2 text-sm text-[#e5e7eb]"
                >
                  <option value="">自动（使用当前加载的模型）</option>
                  {availableModels.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </div>
            )}

            <button
              onClick={switchBackend}
              className="w-full px-3 py-2 bg-[#3b82f6] hover:bg-[#2563eb] text-white rounded-lg text-sm font-medium transition"
            >
              应用切换
            </button>
          </div>
        ) : (
          <div className="text-sm text-[#6b7280]">加载中...</div>
        )}

        {/* LM Studio 使用提示 */}
        {llmStatus && !llmStatus.lmstudio.available && selectedBackend === 'lmstudio' && (
          <div className="bg-[#3b82f6]/10 border border-[#3b82f6]/20 rounded-lg p-3 text-xs text-[#93c5fd]">
            <p className="font-medium mb-1">LM Studio 未运行</p>
            <p className="mb-2">请先启动 LM Studio 并加载模型：</p>
            <ol className="list-decimal list-inside space-y-1 text-[#e5e7eb]">
              <li>下载 LM Studio: <code className="bg-[#0f172a] px-1 rounded">lmstudio.ai</code></li>
              <li>搜索并下载 <code className="bg-[#0f172a] px-1 rounded">Qwen3.5-0.8B-GGUF</code></li>
              <li>加载模型并启动本地服务器（默认端口 1234）</li>
            </ol>
          </div>
        )}
      </div>
    </div>
  )
}
