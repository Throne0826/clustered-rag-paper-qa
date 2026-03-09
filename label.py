import os
import json
import random
import time
from collections import defaultdict

from tqdm import tqdm
from openai import OpenAI

# ======== 配置 ========
input_file = "chunks_clustered.jsonl"
output_file = "cluster_labels.json"

# 统一通过环境变量配置，避免明文密钥
# - OPENAI_API_KEY: 必填
# - OPENAI_BASE_URL: 可选（默认 DashScope 兼容地址）
# - OPENAI_MODEL: 可选（默认 qwen3-max）
API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.getenv("OPENAI_MODEL", "qwen3-max")

samples_per_cluster = 3  # 每个cluster抽样数
max_text_len = 200  # 每段文本限制字符数

if not API_KEY:
    raise ValueError("未检测到 OPENAI_API_KEY，请先设置环境变量后再运行。")

# 初始化客户端（增加超时）
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=60.0,
    max_retries=3,
)


def generate_label(cluster_id, texts, max_retries=3):
    """调用LLM生成cluster标签（带重试）"""
    prompt = f"""These text chunks belong to the same topic:

{chr(10).join([f"{i + 1}. {text}" for i, text in enumerate(texts)])}

Summarize the topic in 2-5 words. Return ONLY the label:"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You summarize academic topics concisely."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=15,
            )
            label = (response.choices[0].message.content or "").strip()
            label = label.strip('"\'').split(".")[0].split("\n")[0]
            if len(label) > 50:
                label = label[:50]
            return label or f"Cluster_{cluster_id}"

        except Exception as e:
            print(f"   Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            else:
                return f"Cluster_{cluster_id}"


print("读取聚类数据...")
clusters = defaultdict(list)

with open(input_file, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="加载"):
        try:
            data = json.loads(line)
            if "cluster_id" in data and "text" in data:
                clusters[data["cluster_id"]].append(data["text"])
        except json.JSONDecodeError:
            continue

print(f"发现 {len(clusters)} 个clusters")

existing_labels = {}
if os.path.exists(output_file):
    print("发现已有标签文件，加载已完成内容...")
    with open(output_file, "r", encoding="utf-8") as f:
        existing_labels = json.load(f)
    print(f"已完成: {len(existing_labels)}/{len(clusters)}")

cluster_labels = existing_labels.copy()
remaining = [
    cid
    for cid in sorted(clusters.keys())
    if str(cid) not in cluster_labels and cid not in cluster_labels
]

print(f"\n待生成: {len(remaining)} 个clusters (每cluster {samples_per_cluster}个样本)")

for cluster_id in tqdm(remaining, desc="生成标签"):
    texts = clusters[cluster_id]

    sample_texts = random.sample(texts, min(samples_per_cluster, len(texts)))
    sample_texts = [t[:max_text_len].replace("\n", " ") for t in sample_texts]

    label = generate_label(cluster_id, sample_texts)
    cluster_labels[cluster_id] = label

    print(f"   Cluster {cluster_id}: {label}")

    if len(cluster_labels) % 5 == 0:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(cluster_labels, f, ensure_ascii=False, indent=2)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(cluster_labels, f, ensure_ascii=False, indent=2)

print(f"\n{'=' * 50}")
print(f"完成! 共 {len(cluster_labels)} 个标签")
print(f"保存到: {output_file}")
print("\n标签列表示例:")

def _sort_key(cid):
    try:
        return (0, int(cid))
    except (TypeError, ValueError):
        return (1, str(cid))

for cid in sorted(cluster_labels.keys(), key=_sort_key)[:10]:
    print(f"  Cluster {cid}: {cluster_labels[cid]}")
