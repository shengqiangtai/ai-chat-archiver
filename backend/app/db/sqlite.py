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
from app.models.schemas import Chunk, EntityMention, SaveRequest

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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kb_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    title TEXT NOT NULL,
                    role_summary TEXT,
                    message_range TEXT,
                    created_at TEXT,
                    url TEXT,
                    source_path TEXT NOT NULL,
                    tags TEXT,
                    model_name TEXT,
                    turn_index INTEGER DEFAULT 0,
                    chunk_index INTEGER DEFAULT 0,
                    content TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS kb_chunks_fts USING fts5(
                    chunk_id, doc_id, platform, title, content, tags, role_summary,
                    created_at, source_path, url, model_name
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kb_entities (
                    entity_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    norm_name TEXT NOT NULL UNIQUE,
                    entity_type TEXT NOT NULL,
                    mention_count INTEGER DEFAULT 0,
                    last_seen_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kb_entity_mentions (
                    entity_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    turn_index INTEGER DEFAULT 0,
                    mention_text TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    PRIMARY KEY (entity_id, chunk_id, mention_text)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kb_entity_edges (
                    src_entity_id TEXT NOT NULL,
                    dst_entity_id TEXT NOT NULL,
                    cooccur_scope TEXT NOT NULL,
                    weight INTEGER DEFAULT 1,
                    PRIMARY KEY (src_entity_id, dst_entity_id, cooccur_scope)
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

    # ── 知识库 chunk 全文索引 ───────────────────────────────────────────
    def clear_kb_chunks(self) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM kb_chunks")
            conn.execute("DELETE FROM kb_chunks_fts")
            conn.execute("DELETE FROM kb_entities")
            conn.execute("DELETE FROM kb_entity_mentions")
            conn.execute("DELETE FROM kb_entity_edges")
            conn.commit()

    def upsert_kb_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        with self._conn() as conn:
            for chunk in chunks:
                tags_csv = ",".join(chunk.tags)
                conn.execute(
                    """INSERT OR REPLACE INTO kb_chunks(
                       chunk_id, doc_id, platform, title, role_summary, message_range,
                       created_at, url, source_path, tags, model_name, turn_index,
                       chunk_index, content
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.platform,
                        chunk.title,
                        chunk.role_summary,
                        chunk.message_range,
                        chunk.created_at,
                        chunk.url or "",
                        chunk.source_path,
                        tags_csv,
                        chunk.model_name or "",
                        chunk.turn_index,
                        chunk.chunk_index,
                        chunk.text,
                    ),
                )
                conn.execute("DELETE FROM kb_chunks_fts WHERE chunk_id = ?", (chunk.chunk_id,))
                conn.execute(
                    """INSERT INTO kb_chunks_fts(
                       chunk_id, doc_id, platform, title, content, tags, role_summary,
                       created_at, source_path, url, model_name
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.platform,
                        chunk.title,
                        chunk.text,
                        tags_csv,
                        chunk.role_summary,
                        chunk.created_at,
                        chunk.source_path,
                        chunk.url or "",
                        chunk.model_name or "",
                    ),
                )
            conn.commit()

        return len(chunks)

    def delete_kb_chunks_by_doc(self, doc_id: str) -> int:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT chunk_id FROM kb_chunks WHERE doc_id = ?",
                (doc_id,),
            ).fetchall()
            if not rows:
                return 0
            chunk_ids = [str(r["chunk_id"]) for r in rows]
            conn.execute("DELETE FROM kb_chunks WHERE doc_id = ?", (doc_id,))
            for chunk_id in chunk_ids:
                conn.execute("DELETE FROM kb_chunks_fts WHERE chunk_id = ?", (chunk_id,))
            conn.execute("DELETE FROM kb_entity_mentions WHERE doc_id = ?", (doc_id,))
            conn.execute(
                """DELETE FROM kb_entity_edges
                   WHERE src_entity_id NOT IN (SELECT DISTINCT entity_id FROM kb_entity_mentions)
                      OR dst_entity_id NOT IN (SELECT DISTINCT entity_id FROM kb_entity_mentions)"""
            )
            conn.execute(
                """DELETE FROM kb_entities
                   WHERE entity_id NOT IN (SELECT DISTINCT entity_id FROM kb_entity_mentions)"""
            )
            conn.execute(
                """UPDATE kb_entities
                   SET mention_count = (
                       SELECT COUNT(*) FROM kb_entity_mentions m
                       WHERE m.entity_id = kb_entities.entity_id
                   )"""
            )
            self._rebuild_entity_edges(conn)
            conn.commit()
        return len(chunk_ids)

    def upsert_entity_mentions(self, mentions: list[EntityMention], created_at: str = "") -> int:
        if not mentions:
            return 0

        with self._conn() as conn:
            for mention in mentions:
                conn.execute(
                    """INSERT INTO kb_entities(entity_id, name, norm_name, entity_type, mention_count, last_seen_at)
                       VALUES (?,?,?,?,1,?)
                       ON CONFLICT(norm_name) DO UPDATE SET
                         name=excluded.name,
                         entity_type=excluded.entity_type,
                         mention_count=mention_count + 1,
                         last_seen_at=excluded.last_seen_at""",
                    (
                        mention.entity_id,
                        mention.name,
                        mention.norm_name,
                        mention.entity_type,
                        created_at,
                    ),
                )
                entity_row = conn.execute(
                    "SELECT entity_id FROM kb_entities WHERE norm_name = ?",
                    (mention.norm_name,),
                ).fetchone()
                entity_id = str(entity_row["entity_id"]) if entity_row else mention.entity_id
                conn.execute(
                    """INSERT OR REPLACE INTO kb_entity_mentions(
                       entity_id, chunk_id, doc_id, turn_index, mention_text, entity_type
                    ) VALUES (?,?,?,?,?,?)""",
                    (
                        entity_id,
                        mention.chunk_id,
                        mention.doc_id,
                        mention.turn_index,
                        mention.mention_text,
                        mention.entity_type,
                    ),
                )
            conn.execute(
                """UPDATE kb_entities
                   SET mention_count = (
                       SELECT COUNT(*) FROM kb_entity_mentions m
                       WHERE m.entity_id = kb_entities.entity_id
                   )"""
            )
            self._rebuild_entity_edges(conn)
            conn.commit()

        return len(mentions)

    def search_entities(self, names: list[str], limit: int = 8) -> list[dict]:
        if not names:
            return []
        seen: set[str] = set()
        rows_out: list[dict] = []
        with self._conn() as conn:
            for name in names:
                rows = conn.execute(
                    """SELECT entity_id, name, norm_name, entity_type, mention_count, last_seen_at
                       FROM kb_entities
                       WHERE norm_name = ? OR norm_name LIKE ?
                       ORDER BY mention_count DESC
                       LIMIT ?""",
                    (name, f"%{name}%", max(1, limit)),
                ).fetchall()
                for row in rows:
                    entity_id = str(row["entity_id"])
                    if entity_id in seen:
                        continue
                    seen.add(entity_id)
                    rows_out.append(dict(row))
                    if len(rows_out) >= limit:
                        return rows_out
        return rows_out

    def search_entity_chunks(
        self,
        entity_names: list[str],
        platform: str | None = None,
        model_name: str | None = None,
        tag: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 15,
    ) -> list[dict]:
        entities = self.search_entities(entity_names, limit=max(1, limit))
        if not entities:
            return []

        entity_ids = [str(item["entity_id"]) for item in entities]
        entity_name_map = {str(item["entity_id"]): str(item["name"]) for item in entities}
        placeholders = ",".join("?" for _ in entity_ids)
        filters = []
        params: list[object] = [*entity_ids]

        if platform:
            filters.append("c.platform = ?")
            params.append(platform)
        if model_name:
            filters.append("c.model_name = ?")
            params.append(model_name)
        if tag:
            filters.append("c.tags LIKE ?")
            params.append(f"%{tag}%")
        if date_from:
            filters.append("c.created_at >= ?")
            params.append(date_from)
        if date_to:
            filters.append("c.created_at <= ?")
            params.append(date_to)

        where_sql = f" AND {' AND '.join(filters)}" if filters else ""

        with self._conn() as conn:
            rows = conn.execute(
                f"""SELECT c.*, COUNT(DISTINCT m.entity_id) AS entity_match_count,
                           GROUP_CONCAT(DISTINCT m.entity_id) AS matched_entity_ids
                    FROM kb_entity_mentions m
                    JOIN kb_chunks c ON c.chunk_id = m.chunk_id
                    WHERE m.entity_id IN ({placeholders}){where_sql}
                    GROUP BY c.chunk_id
                    ORDER BY entity_match_count DESC, c.created_at DESC
                    LIMIT ?""",
                [*params, max(1, limit)],
            ).fetchall()

        result: list[dict] = []
        for row in rows:
            item = dict(row)
            matched_ids = [eid for eid in str(item.get("matched_entity_ids") or "").split(",") if eid]
            item["entity_names"] = [entity_name_map[eid] for eid in matched_ids if eid in entity_name_map]
            item["entity_score"] = float(item.get("entity_match_count") or 0.0)
            result.append(item)
        return result

    def get_related_entities(self, entity_ids: list[str], limit: int = 8) -> list[dict]:
        if not entity_ids:
            return []
        placeholders = ",".join("?" for _ in entity_ids)
        with self._conn() as conn:
            rows = conn.execute(
                f"""SELECT e.entity_id, e.name, e.norm_name, e.entity_type, SUM(edges.weight) AS edge_weight
                    FROM kb_entity_edges edges
                    JOIN kb_entities e
                      ON e.entity_id = CASE
                        WHEN edges.src_entity_id IN ({placeholders}) THEN edges.dst_entity_id
                        ELSE edges.src_entity_id
                      END
                    WHERE edges.src_entity_id IN ({placeholders}) OR edges.dst_entity_id IN ({placeholders})
                    GROUP BY e.entity_id, e.name, e.norm_name, e.entity_type
                    ORDER BY edge_weight DESC, e.mention_count DESC
                    LIMIT ?""",
                [*entity_ids, *entity_ids, *entity_ids, max(1, limit)],
            ).fetchall()
        return [dict(row) for row in rows]

    def get_entity_stats(self) -> dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) AS c FROM kb_entities").fetchone()["c"]
            top_rows = conn.execute(
                """SELECT name, entity_type, mention_count
                   FROM kb_entities
                   ORDER BY mention_count DESC, name ASC
                   LIMIT 8"""
            ).fetchall()
        return {
            "total_entities": int(total),
            "top_entities": [dict(row) for row in top_rows],
        }

    @staticmethod
    def _rebuild_entity_edges(conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM kb_entity_edges")
        groups = conn.execute(
            """SELECT doc_id, turn_index, GROUP_CONCAT(DISTINCT entity_id) AS entity_ids
               FROM kb_entity_mentions
               GROUP BY doc_id, turn_index"""
        ).fetchall()
        for row in groups:
            entity_ids = sorted(
                eid for eid in str(row["entity_ids"] or "").split(",") if eid
            )
            for i, src in enumerate(entity_ids):
                for dst in entity_ids[i + 1:]:
                    conn.execute(
                        """INSERT INTO kb_entity_edges(src_entity_id, dst_entity_id, cooccur_scope, weight)
                           VALUES (?, ?, 'turn', 1)
                           ON CONFLICT(src_entity_id, dst_entity_id, cooccur_scope)
                           DO UPDATE SET weight = weight + 1""",
                        (src, dst),
                    )

    def search_kb_chunks(
        self,
        query: str,
        platform: str | None = None,
        model_name: str | None = None,
        tag: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 15,
    ) -> list[dict]:
        q = (query or "").strip()
        if not q:
            return []

        filters = []
        params: list[object] = []

        if platform:
            filters.append("c.platform = ?")
            params.append(platform)
        if model_name:
            filters.append("c.model_name = ?")
            params.append(model_name)
        if tag:
            filters.append("c.tags LIKE ?")
            params.append(f"%{tag}%")
        if date_from:
            filters.append("c.created_at >= ?")
            params.append(date_from)
        if date_to:
            filters.append("c.created_at <= ?")
            params.append(date_to)

        where_sql = f" AND {' AND '.join(filters)}" if filters else ""

        with self._conn() as conn:
            try:
                rows = conn.execute(
                    f"""SELECT c.*, snippet(kb_chunks_fts, 4, '<mark>', '</mark>', ' ... ', 24) AS snippet,
                           bm25(kb_chunks_fts) AS rank_score
                        FROM kb_chunks_fts
                        JOIN kb_chunks c ON c.chunk_id = kb_chunks_fts.chunk_id
                        WHERE kb_chunks_fts MATCH ?{where_sql}
                        ORDER BY rank_score
                        LIMIT ?""",
                    [q, *params, max(1, limit)],
                ).fetchall()
            except sqlite3.OperationalError:
                like = f"%{q}%"
                rows = conn.execute(
                    f"""SELECT c.*, substr(c.content, 1, 220) AS snippet, 0.0 AS rank_score
                        FROM kb_chunks c
                        WHERE c.content LIKE ?{where_sql}
                        ORDER BY c.created_at DESC
                        LIMIT ?""",
                    [like, *params, max(1, limit)],
                ).fetchall()

        return [dict(row) for row in rows]

    def get_chunks_in_turn_window(
        self,
        doc_id: str,
        center_turn_index: int,
        window: int = 1,
    ) -> list[dict]:
        start_turn = max(0, center_turn_index - max(0, window))
        end_turn = center_turn_index + max(0, window)
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM kb_chunks
                   WHERE doc_id = ? AND turn_index BETWEEN ? AND ?
                   ORDER BY chunk_index ASC""",
                (doc_id, start_turn, end_turn),
            ).fetchall()
        return [dict(row) for row in rows]

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
