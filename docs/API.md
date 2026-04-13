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
