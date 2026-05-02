# API Reference

Base URL: `http://127.0.0.1:8765`

## Health

- `GET /health`

## Chats

- `POST /save`
- `GET /chats?platform=&limit=&offset=`
- `GET /chats/{chat_id}`
- `DELETE /chats/{chat_id}`
- `POST /search`
- `GET /stats`

## Cache / Realtime

- `POST /cache/append`

## Topics

- `POST /topic/merge`
- `GET /topic/{topic_id}`
- `GET /topics?limit=&offset=&query=`

## Export

- `POST /export` (`format`: `md|html|xlsx`)

## Dashboard

- `GET /dashboard`

## OpenAI-Compatible API

Base URL for compatible clients: `http://127.0.0.1:8765/v1`

### Models

- `GET /v1/models`

Returns:

```json
{
  "object": "list",
  "data": [
    {
      "id": "ai-chat-archiver-rag",
      "object": "model",
      "created": 1777651200,
      "owned_by": "ai-chat-archiver"
    }
  ]
}
```

### Chat Completions

- `POST /v1/chat/completions`

Example:

```bash
curl http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ai-chat-archiver-rag",
    "messages": [
      {"role": "user", "content": "总结最近关于 rerank 的记录"}
    ],
    "stream": false
  }'
```

Streaming:

```bash
curl http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ai-chat-archiver-rag",
    "messages": [
      {"role": "user", "content": "总结最近关于 rerank 的记录"}
    ],
    "stream": true
  }'
```

Unsupported OpenAI fields such as tools, vision, and strict JSON schema output are ignored in the first compatible version.
