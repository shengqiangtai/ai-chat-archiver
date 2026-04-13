# Contributing

Thanks for contributing to AI Chat Archiver.

## Development Setup

```bash
cd server
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python main.py
```

Open Chrome extension page:

- `chrome://extensions/`
- enable developer mode
- load unpacked `extension/`

## Pull Request Checklist

- Keep changes scoped and explain why.
- Avoid unrelated formatting-only changes.
- Verify server starts and extension popup can save a chat.
- Update docs when behavior or API changes.

## Commit Message

Use concise, descriptive messages, for example:

- `feat(server): add topic list endpoint`
- `fix(extension): improve poe message extraction`
- `docs: update quick start`
