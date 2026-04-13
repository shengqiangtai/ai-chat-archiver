# AI-Chats 存储目录

此目录存放所有归档的 AI 聊天记录，供知识库索引和 RAG 问答使用。

## 目录结构

```
AI-Chats/
├── ChatGPT/
│   └── 2026/
│       └── 2026-04-08_对话标题/
│           ├── chat.md       ← 聊天内容（Markdown 格式）
│           └── meta.json     ← 元数据（平台、标题、时间、模型等）
├── Claude/
├── Gemini/
├── DeepSeek/
└── Poe/
```

## meta.json 格式示例

```json
{
  "id": "唯一ID（可选）",
  "platform": "ChatGPT",
  "title": "对话标题",
  "created_at": "2026-04-08",
  "model": "gpt-4o",
  "url": "https://chatgpt.com/c/xxx（可选）",
  "tags": ["标签1", "标签2"]
}
```

## chat.md 格式示例

```markdown
**User:** 你好，请介绍一下自己

**Assistant:** 我是 ChatGPT，一个由 OpenAI 开发的 AI 助手...

**User:** 下一个问题...

**Assistant:** ...
```

## 注意

- 此目录已加入 `.gitignore`，聊天内容不会被提交到 Git
- 系统会递归扫描所有子目录中的 `chat.md` 文件
- 修改文件后，在前端点击「增量索引」即可更新知识库
