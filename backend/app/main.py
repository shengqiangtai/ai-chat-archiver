"""AI Chat Archiver — 主入口。

启动方式：
    cd backend
    python -m app.main
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import ALLOWED_ORIGINS, PORT, REPO_ROOT
from app.core.logger import setup_logger
from app.api.routes_docs import router as docs_router
from app.api.routes_ingest import router as ingest_router
from app.api.routes_search import router as search_router
from app.api.routes_qa import router as qa_router

logger = setup_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AI Chat Archiver v3.0.0 starting on http://localhost:%s", PORT)
    if (REPO_ROOT / "frontend" / "dist").exists():
        logger.info("前端 SPA: http://localhost:%s/", PORT)
    else:
        logger.info("前端未构建，使用旧版 Dashboard: http://localhost:%s/dashboard", PORT)
    yield


app = FastAPI(
    title="AI Chat Archiver",
    version="3.0.0",
    description="本地 AI 聊天归档 + 知识库问答系统",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API 路由（必须在静态文件之前注册）──────────────────────────────────
app.include_router(docs_router)
app.include_router(ingest_router)
app.include_router(search_router)
app.include_router(qa_router)

# ── 前端静态文件服务 ──────────────────────────────────────────────────
DASHBOARD_FILE = REPO_ROOT / "dashboard" / "index.html"
FRONTEND_DIR = REPO_ROOT / "frontend" / "dist"

if FRONTEND_DIR.exists() and (FRONTEND_DIR / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="frontend-assets")


@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    """旧版单文件 Dashboard。"""
    if DASHBOARD_FILE.exists():
        return HTMLResponse(DASHBOARD_FILE.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Dashboard not found</h1>")


@app.get("/{full_path:path}")
async def serve_spa(request: Request, full_path: str):
    """SPA 回退：未匹配的 GET 请求都返回 index.html（支持前端路由）。"""
    if FRONTEND_DIR.exists():
        # 尝试返回具体静态文件（如 favicon.svg、icons.svg）
        file_path = FRONTEND_DIR / full_path
        if full_path and file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        # 其他所有路径返回 index.html（SPA 路由）
        index = FRONTEND_DIR / "index.html"
        if index.exists():
            return HTMLResponse(index.read_text(encoding="utf-8"))

    if DASHBOARD_FILE.exists():
        return HTMLResponse(DASHBOARD_FILE.read_text(encoding="utf-8"))

    return HTMLResponse(
        "<h1>前端未构建</h1><p>请运行: <code>cd frontend && npm run build</code></p>",
        status_code=200,
    )


def main():
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
