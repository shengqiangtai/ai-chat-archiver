"""SQLite 存储层 — 聊天记录索引 + 文件状态追踪 + 全文搜索。"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.core.config import DB_PATH, STORAGE_ROOT
from app.models.schemas import SaveRequest

logger = logging.getLogger("archiver.db")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_name(name: str, max_len: int = 30) -> str:
    blocked = '<>:"/\\|?*\r\n\t'
    cleaned = "".join(ch for ch in name if ch not in blocked).strip()
    cleaned = cleaned.replace("  ", " ")
    return cleaned[:max_len] if cleaned else "Untitled"


def _role_title(role: str) -> str:
    r = (role or "").strip().lower()
    if r == "user":
        return "👤 User"
    if r == "assistant":
        return "🤖 Assistant"
    return f"🧩 {r.capitalize() or 'Unknown'}"


def _build_markdown(
    *, title: str, platform: str, model: str | None,
    created_at: str, url: str | None, tags: list[str], messages: list[dict],
) -> str:
    lines = [
        f"# {title}", "",
        f"- **平台**: {platform}",
        f"- **模型**: {model or 'N/A'}",
        f"- **时间**: {created_at}",
        f"- **URL**: {url or 'N/A'}",
        f"- **标签**: {', '.join(tags)}",
        "", "---",
    ]
    for msg in messages:
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        lines.extend(["", f"## {_role_title(msg.get('role') or '')}", content, "", "---"])
    return "\n".join(lines).rstrip() + "\n"


def _messages_text(messages: list[dict]) -> str:
    return "\n\n".join(
        f"[{(m.get('role') or '').lower()}]\n{(m.get('content') or '').strip()}"
        for m in messages if (m.get("content") or "").strip()
    )


class Database:
    """封装 SQLite 操作，支持聊天记录 CRUD + 文件状态追踪。"""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    platform TEXT NOT NULL,
                    model TEXT,
                    title TEXT NOT NULL,
                    url TEXT,
                    tags TEXT,
                    created_at TEXT NOT NULL,
                    saved_at TEXT NOT NULL,
                    message_count INTEGER NOT NULL,
                    file_path TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chats_fts USING fts5(
                    id, platform, model, title, content, tags, url,
                    created_at, saved_at, file_path
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_index (
                    file_path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    modified_time REAL NOT NULL,
                    doc_id TEXT,
                    chunk_count INTEGER DEFAULT 0,
                    indexed_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunk_hashes (
                    text_hash TEXT PRIMARY KEY,
                    chunk_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL
                )
            """)
            conn.commit()

    def save_chat(self, data: SaveRequest) -> dict:
        if not data.messages:
            raise ValueError("messages 不能为空")

        now = _now_iso()
        title = (data.title or "Untitled Chat").strip() or "Untitled Chat"
        tags = [t.strip() for t in data.tags if t and t.strip()]
        messages = [m.model_dump() for m in data.messages]

        with self._conn() as conn:
            existing = None
            if data.url:
                existing = conn.execute(
                    "SELECT * FROM chats WHERE platform = ? AND url = ?",
                    (data.platform, data.url),
                ).fetchone()

            if existing:
                chat_id = str(existing["id"])
                created_at = str(existing["created_at"])
                chat_dir = Path(str(existing["file_path"]))
            else:
                chat_id = uuid.uuid4().hex
                created_at = messages[0].get("time") or now
                chat_dir = self._generate_chat_dir(data.platform, created_at, title, chat_id)

            chat_dir.mkdir(parents=True, exist_ok=True)
            md_content = _build_markdown(
                title=title, platform=data.platform, model=data.model,
                created_at=created_at, url=data.url, tags=tags, messages=messages,
            )
            content_text = _messages_text(messages)

            meta = {
                "id": chat_id, "platform": data.platform, "model": data.model,
                "title": title, "url": data.url, "tags": tags,
                "message_count": len(messages), "created_at": created_at,
                "saved_at": now, "file_path": str(chat_dir),
            }
            (chat_dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            md_path = chat_dir / "chat.md"
            md_path.write_text(md_content, encoding="utf-8")

            if md_path.stat().st_size > 1024 * 1024:
                logger.warning("chat.md 超过 1MB: %s", md_path)

            tags_csv = ",".join(tags)
            if existing:
                conn.execute(
                    """UPDATE chats SET model=?, title=?, url=?, tags=?,
                       saved_at=?, message_count=?, file_path=? WHERE id=?""",
                    (data.model, title, data.url, tags_csv, now, len(messages), str(chat_dir), chat_id),
                )
            else:
                conn.execute(
                    """INSERT INTO chats(id, platform, model, title, url, tags,
                       created_at, saved_at, message_count, file_path) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (chat_id, data.platform, data.model, title, data.url, tags_csv,
                     created_at, now, len(messages), str(chat_dir)),
                )

            conn.execute("DELETE FROM chats_fts WHERE id = ?", (chat_id,))
            conn.execute(
                """INSERT INTO chats_fts(id, platform, model, title, content, tags, url,
                   created_at, saved_at, file_path) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (chat_id, data.platform, data.model or "", title, content_text,
                 tags_csv, data.url or "", created_at, now, str(chat_dir)),
            )
            conn.commit()

        return {"ok": True, "path": str(chat_dir), "id": chat_id}

    def _generate_chat_dir(self, platform: str, created_at: str, title: str, chat_id: str) -> Path:
        try:
            dt = datetime.fromisoformat(created_at)
        except ValueError:
            dt = datetime.now()
        year = dt.strftime("%Y")
        date = dt.strftime("%Y-%m-%d")
        base_name = f"{date}_{_safe_name(title)}"
        chat_dir = STORAGE_ROOT / platform / year / base_name
        if chat_dir.exists():
            chat_dir = STORAGE_ROOT / platform / year / f"{base_name}_{chat_id[:8]}"
        return chat_dir

    def get_chat_list(self, platform: str | None = None, limit: int = 50, offset: int = 0) -> list:
        with self._conn() as conn:
            if platform:
                rows = conn.execute(
                    """SELECT id, platform, model, title, url, tags, message_count,
                       created_at, saved_at, file_path FROM chats
                       WHERE platform = ? ORDER BY saved_at DESC LIMIT ? OFFSET ?""",
                    (platform, max(1, limit), max(0, offset)),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT id, platform, model, title, url, tags, message_count,
                       created_at, saved_at, file_path FROM chats
                       ORDER BY saved_at DESC LIMIT ? OFFSET ?""",
                    (max(1, limit), max(0, offset)),
                ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def search_chats(self, query: str, platform: str | None = None, limit: int = 20) -> list:
        q = (query or "").strip()
        if not q:
            return []
        with self._conn() as conn:
            try:
                if platform:
                    rows = conn.execute(
                        """SELECT id, platform, model, title, tags, url, created_at, saved_at, file_path,
                           snippet(chats_fts, 4, '<mark>', '</mark>', ' ... ', 24) AS snippet
                           FROM chats_fts WHERE chats_fts MATCH ? AND platform = ?
                           ORDER BY bm25(chats_fts) LIMIT ?""",
                        (q, platform, max(1, limit)),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT id, platform, model, title, tags, url, created_at, saved_at, file_path,
                           snippet(chats_fts, 4, '<mark>', '</mark>', ' ... ', 24) AS snippet
                           FROM chats_fts WHERE chats_fts MATCH ?
                           ORDER BY bm25(chats_fts) LIMIT ?""",
                        (q, max(1, limit)),
                    ).fetchall()
            except sqlite3.OperationalError:
                like = f"%{q}%"
                if platform:
                    rows = conn.execute(
                        """SELECT id, platform, model, title, tags, url, created_at, saved_at, file_path,
                           substr(content, 1, 160) AS snippet FROM chats_fts
                           WHERE content LIKE ? AND platform = ? ORDER BY saved_at DESC LIMIT ?""",
                        (like, platform, max(1, limit)),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT id, platform, model, title, tags, url, created_at, saved_at, file_path,
                           substr(content, 1, 160) AS snippet FROM chats_fts
                           WHERE content LIKE ? ORDER BY saved_at DESC LIMIT ?""",
                        (like, max(1, limit)),
                    ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_chat_by_id(self, chat_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT id, platform, model, title, url, tags, message_count,
                   created_at, saved_at, file_path FROM chats WHERE id = ?""",
                (chat_id,),
            ).fetchone()
        if not row:
            return None
        item = self._row_to_dict(row)
        chat_dir = Path(item["file_path"])
        meta = item
        meta_path = chat_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        content = ""
        md_path = chat_dir / "chat.md"
        if md_path.exists():
            content = md_path.read_text(encoding="utf-8")
        return {"id": item["id"], "meta": meta, "content": content}

    def delete_chat(self, chat_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute("SELECT file_path FROM chats WHERE id = ?", (chat_id,)).fetchone()
            if not row:
                return False
            file_path = row["file_path"]
            conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
            conn.execute("DELETE FROM chats_fts WHERE id = ?", (chat_id,))
            conn.commit()
        if file_path:
            shutil.rmtree(file_path, ignore_errors=True)
        return True

    def get_stats(self) -> dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) AS c FROM chats").fetchone()["c"]
            rows = conn.execute(
                "SELECT platform, COUNT(*) AS count FROM chats GROUP BY platform ORDER BY count DESC"
            ).fetchall()
        by_platform = {row["platform"]: row["count"] for row in rows}
        platforms = [{"platform": row["platform"], "count": row["count"]} for row in rows]
        return {"total": total, "by_platform": by_platform, "platforms": platforms}

    # ── 文件索引追踪（增量索引用） ──────────────────────────────────────
    def get_file_record(self, file_path: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM file_index WHERE file_path = ?", (file_path,)).fetchone()
        return dict(row) if row else None

    def upsert_file_record(self, file_path: str, file_hash: str, modified_time: float,
                           doc_id: str, chunk_count: int) -> None:
        now = _now_iso()
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO file_index(file_path, file_hash, modified_time,
                   doc_id, chunk_count, indexed_at) VALUES (?,?,?,?,?,?)""",
                (file_path, file_hash, modified_time, doc_id, chunk_count, now),
            )
            conn.commit()

    def delete_file_record(self, file_path: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM file_index WHERE file_path = ?", (file_path,))
            conn.commit()

    def get_all_file_records(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM file_index").fetchall()
        return [dict(r) for r in rows]

    # ── chunk 去重 ─────────────────────────────────────────────────────
    def has_chunk_hash(self, text_hash: str) -> bool:
        with self._conn() as conn:
            row = conn.execute("SELECT 1 FROM chunk_hashes WHERE text_hash = ?", (text_hash,)).fetchone()
        return row is not None

    def add_chunk_hash(self, text_hash: str, chunk_id: str, doc_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO chunk_hashes(text_hash, chunk_id, doc_id) VALUES (?,?,?)",
                (text_hash, chunk_id, doc_id),
            )
            conn.commit()

    def delete_chunk_hashes_by_doc(self, doc_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM chunk_hashes WHERE doc_id = ?", (doc_id,))
            conn.commit()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        item = dict(row)
        tags = item.get("tags") or ""
        if isinstance(tags, str):
            item["tags"] = [t for t in tags.split(",") if t]
        return item


_db_instance: Database | None = None


def get_db() -> Database:
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance
