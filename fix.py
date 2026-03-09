"""数据分布诊断"""
import json
from collections import Counter

# 1. 检查 Cluster 分布
with open("chunks_clustered.jsonl", "r", encoding="utf-8") as f:
    clusters = []
    papers = set()
    for line in f:
        data = json.loads(line)
        clusters.append(data.get("cluster_id"))
        papers.add(data.get("paper_title"))

print(f"📊 数据统计:")
print(f"  总 Chunks: {len(clusters)}")
print(f"  总论文数: {len(papers)}")
print(f"\n📁 Cluster 分布 (Top 10):")

cluster_counts = Counter(clusters)
for cid, count in cluster_counts.most_common(10):
    # 尝试从文件中找到这个 cluster 的样本文本
    sample_text = ""
    with open("chunks_clustered.jsonl", "r", encoding="utf-8") as f2:
        for line in f2:
            d = json.loads(line)
            if d.get("cluster_id") == cid:
                sample_text = d.get("text", "")[:100]
                break
    print(f"  Cluster {cid}: {count} chunks - {sample_text}...")

# 2. 检查关键词覆盖
keywords = ["GraphRAG", "Self-RAG", "SpecialRAG", "retrieval", "embedding",
           "knowledge graph", "vector", "semantic", "multi-hop", "fusion"]

print(f"\n🔍 关键词覆盖检查:")
for kw in keywords:
    count = 0
    with open("chunks_clustered.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if kw.lower() in line.lower():
                count += 1
                if count >= 5:  # 找到5个就停
                    break
    status = "✅" if count >= 5 else "❌"
    print(f"  {status} {kw}: {count}+ chunks")