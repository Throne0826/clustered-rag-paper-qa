"""Clustered RAG - Streamlit Web界面（修复路径版）"""
import streamlit as st
import os
import sys
import json
import pickle
import numpy as np
import faiss
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import time

# 获取当前脚本所在目录（用于找数据文件）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 页面配置
st.set_page_config(
    page_title="Clustered RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS样式
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button { 
        width: 100%; 
        height: 50px;
        font-size: 18px;
        font-weight: bold;
    }
    .draft-box { 
        background-color: #f0f2f6; 
        padding: 15px; 
        border-radius: 10px; 
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    .draft-external {
        border-left-color: #FF9800 !important;
        background-color: #fff3e0 !important;
    }
    .final-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .source-tag {
        font-size: 12px;
        padding: 2px 8px;
        border-radius: 12px;
        background-color: #e3f2fd;
        color: #1976d2;
    }
    .source-external {
        background-color: #fff3e0 !important;
        color: #f57c00 !important;
    }
</style>
""", unsafe_allow_html=True)

# 初始化（缓存）
@st.cache_resource(show_spinner=False)
def load_models():
    """加载模型（只执行一次）"""
    # 使用BASE_DIR构建绝对路径
    index_path = os.path.join(BASE_DIR, "faiss_index.bin")
    mapping_path = os.path.join(BASE_DIR, "id_to_chunk.pkl")
    labels_path = os.path.join(BASE_DIR, "cluster_labels.json")

    # 检查文件是否存在
    for path in [index_path, mapping_path, labels_path]:
        if not os.path.exists(path):
            st.error(f"❌ 找不到文件: {path}")
            st.stop()

    embed_model = SentenceTransformer("BAAI/bge-large-en", device="cpu", local_files_only=True)
    index = faiss.read_index(index_path)

    with open(mapping_path, "rb") as f:
        id_to_meta = pickle.load(f)["index_to_metadata"]

    with open(labels_path, "r", encoding="utf-8") as f:
        cluster_labels = {int(k): v for k, v in json.load(f).items()}

    client = OpenAI(
        api_key="sk-91c002d7203543c98c79463e20d5e5ea",  # 你的API Key
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=60
    )

    return embed_model, index, id_to_meta, cluster_labels, client

class StreamlitRAG:
    def __init__(self, embed_model, index, id_to_meta, cluster_labels, client):
        self.embed_model = embed_model
        self.index = index
        self.id_to_meta = id_to_meta
        self.cluster_labels = cluster_labels
        self.client = client

    def retrieve(self, query: str, k: int = 20):
        """向量检索"""
        query_emb = self.embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(query_emb, k)

        chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: continue
            meta = self.id_to_meta[idx]
            chunks.append({
                "score": float(score),
                "text": meta["text"],
                "cluster_id": meta.get("cluster_id", 0),
                "paper_title": meta.get("paper_title", "Unknown"),
                "page_number": meta.get("page_number", 0),
            })
        return chunks

    def cluster_grouping(self, chunks):
        """Cluster分组"""
        clusters = defaultdict(list)
        for chunk in chunks:
            clusters[chunk["cluster_id"]].append(chunk)

        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

        # 如果只有一个cluster，补充其他
        if len(sorted_clusters) == 1:
            additional = self._get_diverse_clusters(sorted_clusters[0][0])
            sorted_clusters.extend(additional[:2])

        return sorted_clusters[:3]

    def _get_diverse_clusters(self, exclude_cid):
        """获取其他clusters"""
        additional = []
        chunks_path = os.path.join(BASE_DIR, "chunks_clustered.jsonl")
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                other_clusters = defaultdict(list)
                for i, line in enumerate(f):
                    if i > 500: break
                    try:
                        data = json.loads(line)
                        cid = data.get("cluster_id", 0)
                        if cid != exclude_cid and len(other_clusters[cid]) < 2:
                            other_clusters[cid].append({
                                "text": data["text"],
                                "cluster_id": cid,
                                "paper_title": data.get("paper_title", "Unknown"),
                                "page_number": data.get("page_number", 0),
                                "score": 0.3
                            })
                    except: continue

                for cid, docs in list(other_clusters.items())[:2]:
                    if docs:
                        additional.append((cid, docs))
        except: pass
        return additional

    def generate_draft(self, query, cid, docs, label):
        """生成Draft"""
        docs_text = "\n".join([f"Doc {i+1}: {d['text'][:180]}" for i, d in enumerate(docs[:2])])

        prompt = f"""You are a research assistant. Answer based on available info.
Question: {query}
Topic: {label}
Documents: {docs_text}
Instructions:
1. Check if documents are relevant to "{query}"
2. If relevant: answer using docs, cite as [Source: Title]
3. If NOT relevant: state "[Note: Docs not directly relevant]" then use your knowledge about "{query}"
Answer (100-200 words):"""

        try:
            response = self.client.chat.completions.create(
                model="qwen3.5-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )
            draft = response.choices[0].message.content.strip()
            used_external = "not directly relevant" in draft.lower() or "not relevant" in draft.lower()
            return {
                "label": label,
                "draft": draft,
                "sources": [f"{d['paper_title']} (P{d['page_number']})" for d in docs[:2]],
                "used_external": used_external
            }
        except Exception as e:
            return {"label": label, "draft": f"Error: {str(e)[:100]}", "sources": [], "used_external": True}

    def evaluate(self, query, drafts):
        """Evaluator综合"""
        if len(drafts) == 1:
            return drafts[0]["draft"]

        candidates = "\n\n".join([
            f"Draft {i+1} ({d['label']}): {d['draft'][:150]}..."
            for i, d in enumerate(drafts)
        ])

        prompt = f"""Question: {query}
Sources:
{candidates}
Synthesize one comprehensive final answer (150-250 words)."""

        try:
            response = self.client.chat.completions.create(
                model="qwen3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except:
            return drafts[0]["draft"] if drafts else "Error"

def main():
    st.title("🧠 Clustered Multi-Document RAG")
    st.markdown("""
    **基于多文档聚类的智能问答系统**  
    *流程：查询 → 向量检索 → Cluster分组 → 小模型并行Drafts → 大模型综合*
    """)

    # 加载模型
    with st.spinner("🚀 正在加载模型（首次较慢）..."):
        try:
            embed_model, index, id_to_meta, cluster_labels, client = load_models()
            rag = StreamlitRAG(embed_model, index, id_to_meta, cluster_labels, client)
            st.success("✅ 模型加载完成！")
        except Exception as e:
            st.error(f"加载失败: {e}")
            return

    # 输入区域
    st.divider()
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "请输入您的问题",
            placeholder="例如：How does GraphRAG improve retrieval quality?",
            label_visibility="collapsed"
        )
    with col2:
        submit = st.button("🔍 开始查询", use_container_width=True, type="primary")

    if submit and query:
        process_query(rag, query)

    # 示例问题
    with st.expander("💡 点击查看示例问题"):
        examples = [
            "How does GraphRAG improve retrieval quality?",
            "对比一下selfrag和specialrag的区别",
            "What are the challenges in multi-hop reasoning?",
            "Explain the clustering mechanism in RAG"
        ]
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            with cols[i % 2]:
                if st.button(ex, key=f"ex_{i}", use_container_width=True):
                    st.session_state.query = ex
                    st.rerun()

def process_query(rag, query):
    """处理查询并展示结果"""
    start_time = time.time()

    # 创建占位符用于动态更新
    progress_area = st.empty()

    with progress_area.container():
        # Step 1: 检索
        with st.status("🔍 Step 1: 向量检索中...", expanded=True) as status:
            chunks = rag.retrieve(query)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("检索到Chunks", len(chunks))
            with col2:
                avg_score = sum(c["score"] for c in chunks[:5]) / 5 if chunks else 0
                st.metric("平均相似度", f"{avg_score:.3f}")
            with col3:
                unique_clusters = len(set(c["cluster_id"] for c in chunks))
                st.metric("涉及Clusters", unique_clusters)

            # 显示前3个chunks详情
            with st.expander("查看检索详情（Top 3）"):
                for i, chunk in enumerate(chunks[:3], 1):
                    st.markdown(f"""
                    **[{i}]** Score: `{chunk['score']:.3f}` | Cluster: `{chunk['cluster_id']}`  
                    **来源**: {chunk['paper_title'][:50]}... (Page {chunk['page_number']})  
                    **内容**: {chunk['text'][:200]}...
                    """)
                    st.divider()

            status.update(label=f"✅ 检索完成 ({len(chunks)} chunks, {unique_clusters} clusters)", state="complete")

        # Step 2: Cluster分组
        with st.status("📦 Step 2: Cluster分组...", expanded=True) as status:
            clusters = rag.cluster_grouping(chunks)

            st.write(f"选中 **{len(clusters)}** 个主题Clusters进行并行分析：")

            cluster_cols = st.columns(len(clusters))
            for i, (cid, docs) in enumerate(clusters):
                label = rag.cluster_labels.get(cid, f"Cluster_{cid}")
                with cluster_cols[i]:
                    st.metric(
                        label=f"📁 {label[:15]}",
                        value=f"{len(docs)} docs",
                        delta=f"ID:{cid}"
                    )

            status.update(label=f"✅ 分组完成", state="complete")

        # Step 3: 并行生成Drafts
        st.subheader("📝 Step 3: Drafter并行生成")

        draft_cols = st.columns(len(clusters))
        drafts = []

        for i, (cid, docs) in enumerate(clusters):
            label = rag.cluster_labels.get(cid, f"Cluster_{cid}")

            with draft_cols[i]:
                with st.spinner(f"生成 [{label[:12]}]..."):
                    draft_data = rag.generate_draft(query, cid, docs, label)
                    drafts.append(draft_data)

                    # 根据来源类型显示不同样式
                    is_external = draft_data.get("used_external", False)
                    css_class = "draft-external" if is_external else ""
                    source_tag = "🌐 模型知识" if is_external else "📚 本地文档"
                    source_class = "source-external" if is_external else ""

                    st.markdown(f"""
                    <div class="draft-box {css_class}">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                            <strong>{label}</strong>
                            <span class="source-tag {source_class}">{source_tag}</span>
                        </div>
                        <div style="font-size:12px; color:#666; margin-bottom:8px;">
                            来源: {', '.join(draft_data['sources'][:1])}
                        </div>
                        <div style="font-size:14px; line-height:1.5;">
                            {draft_data['draft']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Step 4: Evaluator综合
        with st.spinner("🎯 Step 4: Evaluator综合评估中..."):
            final_answer = rag.evaluate(query, drafts)

            st.subheader("🎯 Step 4: 最终答案（大模型综合）")
            st.markdown(f"""
            <div class="final-box">
                <div style="font-size:16px; line-height:1.8;">
                    {final_answer.replace(chr(10), '<br>')}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 统计信息
        elapsed = time.time() - start_time
        external_count = sum(1 for d in drafts if d.get("used_external"))

        st.divider()
        cols = st.columns(4)
        with cols[0]:
            st.metric("⏱️ 总耗时", f"{elapsed:.1f}s")
        with cols[1]:
            st.metric("📝 Drafts数", len(drafts))
        with cols[2]:
            st.metric("📚 本地文档", len(drafts) - external_count)
        with cols[3]:
            st.metric("🌐 模型知识", external_count)

        # 详细来源展开
        with st.expander("📋 查看详细来源"):
            for i, draft in enumerate(drafts, 1):
                source_type = "模型知识补充" if draft.get("used_external") else "本地知识库"
                st.markdown(f"""
                **Draft {i}: {draft['label']}** ({source_type})  
                {chr(10).join(['• ' + s for s in draft['sources']])}
                """)
                st.divider()

if __name__ == "__main__":
    main()