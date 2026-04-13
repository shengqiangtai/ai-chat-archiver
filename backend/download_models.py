"""从 HuggingFace 镜像站下载三个模型到项目本地 models/ 目录。

使用镜像站加速下载（国内网络友好）：
  - hf-mirror.com（默认）
  - 或通过 HF_ENDPOINT 环境变量自定义

用法：
    cd backend
    python download_models.py

    # 指定镜像站
    HF_ENDPOINT=https://hf-mirror.com python download_models.py

    # 只下载某个模型
    python download_models.py --model embedding
    python download_models.py --model reranker
    python download_models.py --model generator
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────────────────────
# 项目根目录（此脚本所在的 backend/ 的上级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# HuggingFace 镜像站（默认使用 hf-mirror.com）
HF_MIRROR = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

# 三个模型配置
MODELS = {
    "embedding": {
        "repo_id": "Qwen/Qwen3-Embedding-0.6B",
        "local_dir": MODELS_DIR / "Qwen3-Embedding-0.6B",
        "desc": "Embedding 模型（1024 维，约 1.2 GB）",
        "type": "sentence_transformers",
    },
    "reranker": {
        "repo_id": "Qwen/Qwen3-Reranker-0.6B",
        "local_dir": MODELS_DIR / "Qwen3-Reranker-0.6B",
        "desc": "Reranker 模型（约 1.2 GB）",
        "type": "transformers",
    },
    "generator": {
        "repo_id": "Qwen/Qwen3.5-0.8B",
        "local_dir": MODELS_DIR / "Qwen3.5-0.8B",
        "desc": "生成模型 Qwen3.5-0.8B（约 1.8 GB）",
        "type": "transformers",
    },
}


def check_huggingface_hub() -> bool:
    try:
        import huggingface_hub  # noqa: F401
        return True
    except ImportError:
        return False


def install_huggingface_hub() -> None:
    import subprocess
    print("  安装 huggingface_hub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"])


def download_model(key: str, cfg: dict, mirror: str) -> bool:
    """
    使用 huggingface_hub.snapshot_download 下载模型到本地目录。
    通过设置 HF_ENDPOINT 环境变量使用镜像站。
    """
    from huggingface_hub import snapshot_download

    local_dir: Path = cfg["local_dir"]
    repo_id: str = cfg["repo_id"]
    desc: str = cfg["desc"]

    # 判断是否已下载（检查关键文件）
    if _is_model_downloaded(local_dir):
        print(f"  ✓ 已存在，跳过: {local_dir.relative_to(PROJECT_ROOT)}")
        return True

    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"  下载中: {repo_id}")
    print(f"  镜像站: {mirror}")
    print(f"  保存到: {local_dir.relative_to(PROJECT_ROOT)}")

    os.environ["HF_ENDPOINT"] = mirror

    t0 = time.time()
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            ignore_patterns=["*.bin.index.json", "flax_model*", "tf_model*", "rust_model*"],
        )
        elapsed = time.time() - t0
        print(f"  ✓ 下载完成，耗时 {elapsed:.0f}s，路径: {local_dir.relative_to(PROJECT_ROOT)}")
        return True
    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        print(f"    可以手动运行:")
        print(f"    HF_ENDPOINT={mirror} huggingface-cli download {repo_id} --local-dir {local_dir}")
        return False


def _is_model_downloaded(local_dir: Path) -> bool:
    """检查模型目录中是否已有完整的模型文件。"""
    if not local_dir.exists():
        return False
    # 检查是否有 safetensors 或 .bin 文件
    has_weights = any(
        local_dir.rglob(pat)
        for pat in ["*.safetensors", "model.safetensors", "*.bin"]
        if list(local_dir.rglob(pat))
    )
    has_config = (local_dir / "config.json").exists()
    return has_weights and has_config


def generate_env_config() -> None:
    """在 backend/ 下生成 .env.local 配置文件，让应用使用本地模型路径。"""
    env_path = Path(__file__).parent / ".env.local"
    lines = [
        "# 本地模型路径配置（由 download_models.py 自动生成）",
        "# 将此文件内容复制到 .env 或在启动时 source 它",
        "",
    ]
    for key, cfg in MODELS.items():
        local_dir: Path = cfg["local_dir"]
        if _is_model_downloaded(local_dir):
            env_var = f"EMBEDDING_MODEL" if key == "embedding" else \
                      f"RERANKER_MODEL" if key == "reranker" else \
                      f"GENERATOR_MODEL"
            lines.append(f'{env_var}="{local_dir}"')

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  已生成本地路径配置文件: {env_path.name}")
    print("  使用方式: source backend/.env.local  或手动设置这些环境变量")


def print_summary(results: dict[str, bool]) -> None:
    print("\n" + "=" * 60)
    print("下载结果汇总")
    print("=" * 60)
    for key, ok in results.items():
        cfg = MODELS[key]
        status = "✓ 就绪" if ok else "✗ 失败"
        print(f"  {status}  {cfg['desc']}")
        if ok:
            print(f"         路径: {cfg['local_dir'].relative_to(PROJECT_ROOT)}")

    all_ok = all(results.values())
    if all_ok:
        print("\n✅ 所有模型已就绪，可以启动服务:")
        print("   cd backend && python -m app.main")
    else:
        print("\n⚠️  部分模型下载失败，请检查网络或手动下载")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 AI Chat Archiver 所需模型")
    parser.add_argument(
        "--model",
        choices=["embedding", "reranker", "generator", "all"],
        default="all",
        help="要下载的模型（默认: all）",
    )
    parser.add_argument(
        "--mirror",
        default=HF_MIRROR,
        help=f"HuggingFace 镜像站地址（默认: {HF_MIRROR}）",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("AI Chat Archiver — 模型下载工具")
    print("=" * 60)
    print(f"\n镜像站: {args.mirror}")
    print(f"模型目录: models/\n")

    # 检查 huggingface_hub
    if not check_huggingface_hub():
        print("安装必要依赖...")
        install_huggingface_hub()

    # 确定要下载的模型
    if args.model == "all":
        targets = list(MODELS.keys())
    else:
        targets = [args.model]

    results: dict[str, bool] = {}
    for key in targets:
        cfg = MODELS[key]
        print(f"\n[{cfg['desc']}]")
        results[key] = download_model(key, cfg, args.mirror)

    # 生成环境配置文件
    generate_env_config()

    print_summary(results)


if __name__ == "__main__":
    main()
