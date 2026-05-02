# OpenAI Compatible API Design

## Goal

Add a small, complete OpenAI-compatible integration layer to AI Chat Archiver.

The feature has two directions:

- The project can call external OpenAI-compatible model providers such as DeepSeek, OpenRouter, SiliconFlow, LM Studio, vLLM, or a self-hosted compatible service.
- The project can expose an OpenAI-compatible API so clients such as Cherry Studio, Chatbox, Open WebUI, or other tools can use the local RAG knowledge base through `/v1`.

The first implementation targets the common chat completion path. It does not try to become a full OpenAI API gateway.

## Non-Goals

- No tools or function calling support.
- No vision or multimodal message support.
- No strict JSON schema `response_format` support.
- No multi-provider routing UI.
- No persistent API key storage.

Unsupported OpenAI request fields should be ignored unless they prevent basic chat completion behavior.

## Current Context

The backend already has:

- `GeneratorProvider` in `backend/app/services/llm/generator.py`.
- Existing generation backends for `lmstudio`, `ollama`, and `transformers`.
- Knowledge base QA entry points in `backend/app/api/routes_qa.py`.
- Runtime configuration helpers in `backend/app/core/config.py`.
- Existing public custom APIs under `/api/kb/qa` and `/api/kb/qa/stream`.

LM Studio already uses an OpenAI-compatible API internally, but it is specialized around the local LM Studio default. The new backend should be a provider-neutral compatible client.

## Approach

Use the smallest useful compatibility layer:

- Add an `openai_compatible` generation backend.
- Add `/v1/models`.
- Add `/v1/chat/completions` with non-streaming and streaming responses.
- Keep existing `/api/kb/*` APIs unchanged.
- Use the existing RAG pipeline behind the compatible server endpoint.

This keeps the feature scoped to the project's knowledge-base purpose while making it usable from standard LLM clients.

## Configuration

Add environment variables:

```text
OPENAI_COMPAT_BASE_URL=https://api.deepseek.com/v1
OPENAI_COMPAT_API_KEY=...
OPENAI_COMPAT_MODEL=deepseek-chat
OPENAI_COMPAT_PROVIDER_NAME=DeepSeek
```

Add runtime config fields:

```json
{
  "generator_backend": "openai_compatible",
  "openai_compatible_base_url": "https://api.deepseek.com/v1",
  "openai_compatible_model": "deepseek-chat",
  "openai_compatible_provider_name": "DeepSeek"
}
```

The API key must stay environment-only and must not be written to `config.json`.

## Internal External-Provider Flow

When the selected generation backend is `openai_compatible`:

1. `qa_answer` or `qa_answer_stream` builds the existing RAG prompt.
2. `GeneratorProvider` selects `OpenAICompatibleGenerator`.
3. The generator calls `POST {base_url}/chat/completions`.
4. Non-streaming mode reads `choices[0].message.content`.
5. Streaming mode reads SSE `choices[0].delta.content`.
6. The existing citation parsing, grounding, cache, and fallback behavior continue to run.

`Authorization: Bearer <OPENAI_COMPAT_API_KEY>` should be sent only when an API key is configured.

## Public OpenAI-Compatible Server Flow

Expose:

- `GET /v1/models`
- `POST /v1/chat/completions`

For `/v1/chat/completions`:

1. Accept common OpenAI chat completion fields: `model`, `messages`, `stream`, `temperature`, `max_tokens`.
2. Collect `system` and `developer` messages as optional instruction context.
3. Use the last non-empty `user` message as the RAG query.
4. If `stream` is false, call `qa_answer`.
5. If `stream` is true, call `qa_answer_stream`.
6. Wrap the result in OpenAI-compatible response objects.

The server endpoint is a RAG compatibility surface, not a raw model pass-through endpoint. The model id can default to `ai-chat-archiver-rag`.

## Response Shapes

Non-streaming response:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1777651200,
  "model": "ai-chat-archiver-rag",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

Streaming response:

```text
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1777651200,"model":"ai-chat-archiver-rag","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1777651200,"model":"ai-chat-archiver-rag","choices":[{"index":0,"delta":{"content":"..."},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1777651200,"model":"ai-chat-archiver-rag","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

Usage token counts can be zero in the first version because the current pipeline does not track provider token accounting consistently.

## Error Handling

- Missing or empty `messages` returns HTTP 400 with an OpenAI-style error object.
- No user message returns HTTP 400 with an OpenAI-style error object.
- External provider failures should be logged and should follow the current generator fallback chain.
- Public streaming errors should emit an error-shaped SSE chunk and then `[DONE]`.
- Unsupported request fields are ignored.

## Files To Change

Likely backend changes:

- `backend/app/core/config.py`
- `backend/app/models/schemas.py`
- `backend/app/services/llm/generator.py`
- `backend/app/api/routes_openai_compatible.py`
- `backend/app/main.py`
- `backend/tests/test_openai_compatible_api.py`
- `backend/tests/test_openai_compatible_generator.py`

Likely documentation changes:

- `README.md`
- `docs/API.md`

Frontend changes are optional for the first implementation. The existing LLM settings UI can continue to work for LM Studio and Ollama; `openai_compatible` can be configured by environment variables first.

## Testing

Add focused backend tests:

- `OpenAICompatibleGenerator` sends bearer auth when an API key exists.
- Non-streaming external provider response is parsed from `choices[0].message.content`.
- Streaming external provider response is parsed from `choices[0].delta.content`.
- `GET /v1/models` returns an OpenAI-style model list.
- `POST /v1/chat/completions` returns an OpenAI-style non-streaming response.
- `POST /v1/chat/completions` returns OpenAI-style SSE chunks when `stream=true`.
- Invalid requests without a user message return 400.

Tests should mock the RAG pipeline and external HTTP calls so they do not require local models, Chroma, or network access.

## Acceptance Criteria

- Users can configure an external OpenAI-compatible provider with environment variables.
- Setting `generator_backend` to `openai_compatible` makes project QA use that provider.
- `GET /v1/models` works.
- `POST /v1/chat/completions` works in non-streaming mode from standard clients.
- `POST /v1/chat/completions` works in streaming mode from standard clients.
- Existing `/api/kb/qa` and `/api/kb/qa/stream` continue to work.
- No API key is persisted to disk.
