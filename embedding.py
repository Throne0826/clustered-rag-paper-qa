import os
import json
import torch
import re
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ======== 配置 ========
input_file = "chunks.jsonl"
output_file = "chunks_with_emb.jsonl"

# 建议：中英混合场景优先用多语模型
# 可选示例："BAAI/bge-m3" / "BAAI/bge-large-zh-v1.5" / "BAAI/bge-large-en-v1.5"
model_name = "BAAI/bge-m3"

batch_size = 64
device = "cpu"

if torch.cuda.is_available():
    device = "cuda"
    batch_size = 128

print(f"[INFO] Device: {device}")
print(f"[INFO] Model: {model_name}")

# ======== 加载模型 ========
model = SentenceTransformer(model_name, device=device)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_passage_text(text: str) -> str:
    """给嵌入模型添加检索前缀。"""
    lower_name = model_name.lower()
    # BGE 系列通常建议使用 instruction 前缀
    if "bge" in lower_name:
        return "passage: " + text
    return text


# ======== 已处理chunk检查（增量写入） ========
processed_ids = set()

if os.path.exists(output_file):
    print("Checking existing output...")
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                processed_ids.add(data["chunk_id"])
            except Exception:
                continue

print("Processed:", len(processed_ids))

# ======== 读取待处理数据 ========
chunks_data = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        if data["chunk_id"] in processed_ids:
            continue

        text = clean_text(data.get("text", ""))
        if len(text) < 30:
            continue

        data["text"] = text
        chunks_data.append(data)

print("To process:", len(chunks_data))

if not chunks_data:
    print("No new chunks to process.")
    raise SystemExit(0)

# ======== 生成 embedding ========
total_batches = (len(chunks_data) + batch_size - 1) // batch_size

with open(output_file, "a", encoding="utf-8") as out_f:
    for i in tqdm(range(total_batches), desc="Embedding"):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(chunks_data))
        batch = chunks_data[start:end]

        texts = [build_passage_text(item["text"]) for item in batch]

        embeddings = model.encode(
            texts,
            batch_size=len(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        for j, chunk in enumerate(batch):
            result = {
                "chunk_id": chunk["chunk_id"],
                "paper_title": chunk.get("paper_title", ""),
                "category": chunk.get("category", ""),
                "page_number": chunk.get("page_number", 0),
                "text": chunk["text"],
                "embedding_model": model_name,
                "embedding": embeddings[j].tolist(),
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

print("[DONE] Embedding generation finished")
print("[WARN] 如果更换了 embedding 模型，请务必重建 FAISS 索引并替换线上索引文件。")
