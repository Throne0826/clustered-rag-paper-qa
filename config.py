"""Clustered RAG 配置 - DashScope版本"""
import os

# ======== DashScope API配置 ========
# 请在系统环境变量中设置 DASHSCOPE_API_KEY，避免在代码中明文保存密钥
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# ======== 模型选择 ========
MODEL_CONFIG = {
    "drafter": {
        "model": "qwen3.5-flash",        # 超快，适合并行生成drafts
        "temperature": 0.3,
        "max_tokens": 800,
        "top_p": 0.8
    },
    "evaluator": {
        "model": "qwen3.5-plus",         # 高质量，平衡速度
        # "model": "qwen3.5-122b-a10b",  # 如需最高质量，取消注释这行（速度较慢）
        "temperature": 0.2,
        "max_tokens": 1500,
        "top_p": 0.8
    }
}

# ======== 检索配置 ========
EMBEDDING_MODEL = "BAAI/bge-large-en"
FAISS_INDEX = "faiss_index.bin"
ID_MAPPING = "id_to_chunk.pkl"
CLUSTER_LABELS = "cluster_labels.json"

TOP_K = 20                      # 初始检索数量
USE_RERANKER = True
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # 轻量版reranker，速度快
RERANK_TOP_K = 10               # rerank后保留数量

# ======== Cluster策略 ========
MAX_CLUSTERS = 3                # 最多并行生成3个draft
MIN_CLUSTER_SIZE = 2            # 过滤单文档cluster
CLUSTER_SCORE_THRESHOLD = 0.1   # cluster相关性阈值（基于平均检索分）

# ======== 系统配置 ========
MAX_WORKERS = 3                 # 并发drafter线程数（防止API限流）
REQUEST_TIMEOUT = 60            # API调用超时