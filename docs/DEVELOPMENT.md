# Development Guide

## Run Locally

```bash
cd server
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python main.py
```

Then load extension from `extension/` in Chrome developer mode.

## Storage Path

Default storage root:

- `~/Documents/AI-Chats`

Config file:

- `server/config.py`

## Debug Tips

- Extension logs: Chrome extension service worker console.
- Site extraction logs: target page DevTools console.
- Server logs: terminal running `python main.py`.
