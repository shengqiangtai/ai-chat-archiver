# 本地模型目录

此目录存放通过 `download_models.py` 下载到本地的三个模型。

下载完成后目录结构如下：

```
models/
├── Qwen3-Embedding-0.6B/     # Embedding 模型（约 1.2 GB）
│   ├── config.json
│   ├── tokenizer.json
│   └── model.safetensors
├── Qwen3-Reranker-0.6B/      # Reranker 模型（约 1.2 GB）
│   ├── config.json
│   └── model.safetensors
└── Qwen3.5-0.8B/             # 生成模型（约 1.8 GB）
    ├── config.json
    └── model.safetensors
```

## 下载方式

```bash
cd backend

# 下载全部三个模型（通过 hf-mirror.com 镜像站）
python download_models.py

# 指定镜像站
HF_ENDPOINT=https://hf-mirror.com python download_models.py

# 只下载某一个
python download_models.py --model embedding
python download_models.py --model reranker
python download_models.py --model generator
```

## 说明

- 应用启动时会自动检测此目录是否有完整的模型文件
- 如果本地存在，优先使用本地；否则从 HuggingFace Hub 在线下载
- 本目录下的模型文件**不应提交到 Git**（体积太大）
