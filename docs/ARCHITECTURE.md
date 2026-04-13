# Architecture

AI Chat Archiver is a local-first system with 4 layers:

1. Browser extension (collection)
2. FastAPI server (processing)
3. Local filesystem + SQLite (storage/index)
4. Dashboard (read/search/view)

## Data Flow

- Content scripts extract chat messages from supported AI sites.
- Extension sends data to local FastAPI service.
- Server writes markdown/html/meta files and indexes text into SQLite FTS.
- Dashboard queries server APIs for list/search/detail/topic views.

## Key Modules

- `extension/content_scripts/*`: site adapters
- `extension/background.js`: API forwarding and retry queue
- `server/main.py`: FastAPI routes
- `server/storage.py`: persistence, topic merge, export
- `dashboard/index.html`: dashboard UI (chat + topic views)
