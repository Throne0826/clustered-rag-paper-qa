import os
import json
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm

# =============================
# 配置
# =============================

input_file = "chunks_with_emb.jsonl"
output_file = "chunks_clustered.jsonl"
cluster_meta_file = "cluster_metadata.json"

# 自动计算更合理的cluster数量
AUTO_CLUSTER = True
DEFAULT_CLUSTER = 120

batch_size = 1000
use_mini_batch = False


print("🔍 检查输入文件...")
if not os.path.exists(input_file):
    print(f"❌ 找不到 {input_file}")
    exit(1)

# =============================
# Step1 读取数据
# =============================

print("\n📥 加载数据...")

all_embeddings = []
all_metadata = []
line_count = 0

with open(input_file, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="读取进度"):

        try:
            data = json.loads(line.strip())

            embedding = np.array(data["embedding"], dtype=np.float32)

            # L2 normalize（重要）
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            all_embeddings.append(embedding)

            meta = {
                "chunk_id": data["chunk_id"],
                "paper_title": data.get("paper_title", ""),
                "category": data.get("category", ""),
                "page_number": data.get("page_number", 0),
                "text": data["text"],
                "embedding": embedding.tolist()
            }

            all_metadata.append(meta)
            line_count += 1

        except:
            continue

print(f"✅ 共加载 {line_count} chunks")

if line_count == 0:
    print("❌ 没有数据")
    exit(1)

X = np.array(all_embeddings)

print("🧮 数据矩阵:", X.shape)

# =============================
# 自动选择cluster数量
# =============================

if AUTO_CLUSTER:

    n_clusters = int(np.sqrt(line_count) * 2)

    n_clusters = max(50, min(n_clusters, 200))

else:

    n_clusters = DEFAULT_CLUSTER

print(f"\n🎯 使用 cluster 数: {n_clusters}")

# =============================
# Step2 聚类
# =============================

print("\n🔨 开始聚类...")

if use_mini_batch and line_count > 20000:

    print("使用 MiniBatchKMeans")

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=2048,
        random_state=42,
        n_init=5,
        max_iter=100
    )

else:

    print("使用 KMeans")

    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=42
    )

cluster_ids = kmeans.fit_predict(X)

print("✅ 聚类完成")

# =============================
# cluster统计
# =============================

print("\n📊 聚类分布:")

unique, counts = np.unique(cluster_ids, return_counts=True)

for cid, count in zip(unique, counts):

    print(
        f"Cluster {cid:3d}: "
        f"{count:4d} chunks "
        f"({count/line_count*100:.1f}%)"
    )

# =============================
# Step3 保存
# =============================

print("\n💾 保存结果...")

with open(output_file, "w", encoding="utf-8") as f:

    for i, meta in enumerate(tqdm(all_metadata)):

        result = {

            "chunk_id": meta["chunk_id"],
            "cluster_id": int(cluster_ids[i]),
            "paper_title": meta["paper_title"],
            "category": meta["category"],
            "page_number": meta["page_number"],
            "text": meta["text"],
            "embedding": meta["embedding"]

        }

        f.write(json.dumps(result, ensure_ascii=False) + "\n")

# =============================
# 保存cluster meta
# =============================

cluster_centers = kmeans.cluster_centers_.tolist()

cluster_metadata = {

    "n_clusters": n_clusters,
    "n_samples": line_count,
    "cluster_centers": cluster_centers,
    "cluster_distribution": {

        int(cid): int(count)
        for cid, count in zip(unique, counts)

    }

}

with open(cluster_meta_file, "w", encoding="utf-8") as f:

    json.dump(cluster_metadata, f, indent=2)

print("\n" + "="*50)
print("✅ 完成")
print("📁 输出:", output_file)
print("📊 cluster meta:", cluster_meta_file)