"""文档加载器 — 递归扫描归档目录中的 chat.md + meta.json。"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.core.config import STORAGE_ROOT
from app.models.schemas import Document
from app.utils.hashing import file_hash, short_id

logger = logging.getLogger("archiver.ingest.loader")


def scan_chat_files(storage_root: Path | None = None) -> list[tuple[Path, Path]]:
    """
    递归扫描 storage_root 下所有 chat.md 文件，
    返回 (md_path, meta_path) 列表，按路径排序。
    """
    root = storage_root or STORAGE_ROOT
    if not root.exists():
        logger.warning("存储根目录不存在: %s", root)
        return []

    files: list[tuple[Path, Path]] = []
    for md_path in root.rglob("chat.md"):
        if ".chroma" in md_path.parts or "node_modules" in md_path.parts:
            continue
        meta_path = md_path.parent / "meta.json"
        files.append((md_path, meta_path))

    files.sort(key=lambda item: str(item[0]))
    logger.info("扫描到 %d 个 chat.md 文件", len(files))
    return files


def load_meta(meta_path: Path) -> dict:
    """安全加载 meta.json，失败返回空字典。"""
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("meta.json 解析失败: %s — %s", meta_path, e)
        return {}


def load_document(md_path: Path, meta_path: Path) -> Document:
    """
    加载一个聊天文件为统一的 Document 结构。
    doc_id 优先取 meta.json 中的 id，否则根据路径生成。
    """
    meta = load_meta(meta_path)
    raw = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    stat = md_path.stat() if md_path.exists() else None

    doc_id = str(meta.get("id") or "").strip()
    if not doc_id:
        doc_id = short_id(str(md_path))

    return Document(
        doc_id=doc_id,
        platform=str(meta.get("platform") or _guess_platform(md_path)),
        title=str(meta.get("title") or md_path.parent.name),
        created_at=str(meta.get("created_at") or ""),
        updated_at=str(meta.get("saved_at") or meta.get("updated_at") or ""),
        path=str(md_path),
        url=meta.get("url"),
        tags=meta.get("tags") or [],
        raw_markdown=raw,
        model_name=meta.get("model"),
        file_hash=file_hash(md_path) if md_path.exists() else "",
        modified_time=stat.st_mtime if stat else 0.0,
    )


def load_all_documents(storage_root: Path | None = None) -> list[Document]:
    """加载存储根目录下所有文档。"""
    files = scan_chat_files(storage_root)
    docs = []
    for md_path, meta_path in files:
        try:
            doc = load_document(md_path, meta_path)
            if doc.raw_markdown.strip():
                docs.append(doc)
        except Exception as e:
            logger.error("加载文档失败: %s — %s", md_path, e)
    logger.info("成功加载 %d 个文档", len(docs))
    return docs


def _guess_platform(md_path: Path) -> str:
    """从路径猜测平台名称。"""
    parts = md_path.parts
    known = {"ChatGPT", "Claude", "Gemini", "DeepSeek", "Poe"}
    for part in parts:
        if part in known:
            return part
    return "Unknown"
