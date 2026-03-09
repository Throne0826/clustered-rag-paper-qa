"""Clustered RAG - Streamlit Web界面（Query Decomposition + Subquery Routing 版）"""
import os
import json
import pickle
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import faiss
import numpy as np
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Clustered RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; height: 50px; font-size: 18px; font-weight: bold; }
    .subq-box { background:#f0f2f6; border-left:4px solid #4CAF50; padding:12px; border-radius:8px; margin:8px 0; }
    .answer-box { background:#eef7ff; border-left:4px solid #1976d2; padding:12px; border-radius:8px; margin:8px 0; }
    .final-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_models():
    index_path = os.path.join(BASE_DIR, "faiss_index.bin")
    mapping_path = os.path.join(BASE_DIR, "id_to_chunk.pkl")
    labels_path = os.path.join(BASE_DIR, "cluster_labels.json")

    for path in [index_path, mapping_path, labels_path]:
        if not os.path.exists(path):
            st.error(f"❌ 找不到文件: {path}")
            st.stop()

    embed_model = SentenceTransformer("BAAI/bge-m3", device="cpu", local_files_only=True)
    index = faiss.read_index(index_path)

    with open(mapping_path, "rb") as f:
        id_to_meta = pickle.load(f)["index_to_metadata"]

    with open(labels_path, "r", encoding="utf-8") as f:
        cluster_labels = {int(k): v for k, v in json.load(f).items()}

    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    if not api_key:
        st.error("❌ 未检测到 OPENAI_API_KEY 环境变量")
        st.stop()

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
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

    @staticmethod
    def _tokenize(text: str):
        return [t for t in re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", text.lower()) if len(t) >= 2]

    def _lexical_overlap(self, query: str, text: str):
        q = set(self._tokenize(query))
        t = set(self._tokenize(text[:900]))
        if not q or not t:
            return 0.0
        return len(q & t) / len(q)

    def retrieve(self, query: str, k: int = 40, candidate_k: int = 220, max_per_cluster: int = 8):
        q_emb = self.embed_model.encode(["query: " + query], normalize_embeddings=True).astype(np.float32)
        candidate_k = max(k, candidate_k)
        scores, indices = self.index.search(q_emb, candidate_k)

        candidates, seen = [], set()
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.id_to_meta[idx]
            text = meta.get("text", "")
            key = text[:140]
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "score": float(score),
                "text": text,
                "cluster_id": int(meta.get("cluster_id", 0)),
                "paper_title": meta.get("paper_title", "Unknown"),
                "page_number": meta.get("page_number", 0),
            })

        if not candidates:
            return []

        sems = [c["score"] for c in candidates]
        smin, smax = min(sems), max(sems)
        denom = (smax - smin) if (smax - smin) > 1e-6 else 1.0

        for c in candidates:
            sem_norm = (c["score"] - smin) / denom
            lex = self._lexical_overlap(query, c["text"])
            c["hybrid_score"] = 0.78 * sem_norm + 0.22 * lex

        candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)

        buckets = defaultdict(list)
        for c in candidates:
            buckets[c["cluster_id"]].append(c)

        selected, take_count = [], defaultdict(int)
        cids = sorted(buckets.keys(), key=lambda cid: buckets[cid][0]["hybrid_score"], reverse=True)

        while len(selected) < k and cids:
            progressed = False
            for cid in cids:
                if len(selected) >= k:
                    break
                if take_count[cid] >= max_per_cluster:
                    continue
                pos = take_count[cid]
                if pos < len(buckets[cid]):
                    selected.append(buckets[cid][pos])
                    take_count[cid] += 1
                    progressed = True
            if not progressed:
                break

        return selected if selected else candidates[:k]

    def decompose_query(self, query: str, min_subqs: int = 2, max_subqs: int = 4):
        prompt = f"""You are a query planner for academic QA.
Question: {query}

Decompose into {min_subqs}-{max_subqs} focused sub-questions that can be answered by paper evidence.
Return strict JSON:
{{"sub_questions": ["...", "..."]}}
"""
        try:
            resp = self.client.chat.completions.create(
                model="qwen3.5-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
            )
            text = resp.choices[0].message.content.strip()
            m = re.search(r"\{[\s\S]*\}", text)
            if not m:
                return [query]
            data = json.loads(m.group(0))
            subqs = [s.strip() for s in data.get("sub_questions", []) if isinstance(s, str) and s.strip()]
            if not subqs:
                return [query]
            subqs = subqs[:max_subqs]
            if len(subqs) < min_subqs:
                subqs = [query] if min_subqs == 1 else (subqs + [query])[:min_subqs]
            return subqs
        except Exception:
            return [query]

    def route_evidence_for_subquery(
        self,
        subq: str,
        per_cluster_docs: int = 2,
        max_docs: int = 8,
        retrieve_k: int = 40,
        retrieve_candidate_k: int = 220,
        retrieve_max_per_cluster: int = 8,
        top_cluster_limit: int = 4,
    ):
        chunks = self.retrieve(
            subq,
            k=retrieve_k,
            candidate_k=retrieve_candidate_k,
            max_per_cluster=retrieve_max_per_cluster,
        )
        if not chunks:
            return [], 0

        clusters = defaultdict(list)
        for c in chunks:
            clusters[c["cluster_id"]].append(c)

        scored = []
        for cid, docs in clusters.items():
            docs = sorted(docs, key=lambda x: x["hybrid_score"], reverse=True)
            score = 0.7 * docs[0]["hybrid_score"] + 0.3 * np.mean([d["hybrid_score"] for d in docs[:4]])
            scored.append((cid, docs, float(score)))

        scored.sort(key=lambda x: x[2], reverse=True)

        evidence = []
        for cid, docs, _ in scored[:top_cluster_limit]:
            for d in docs[:per_cluster_docs]:
                d["cluster_label"] = self.cluster_labels.get(cid, f"Cluster_{cid}")
                evidence.append(d)
                if len(evidence) >= max_docs:
                    break
            if len(evidence) >= max_docs:
                break

        return evidence, len(scored)

    def answer_subquery(self, subq: str, evidence_docs):
        if not evidence_docs:
            return {
                "sub_question": subq,
                "grounded_level": "LOW",
                "answer": "证据不足，无法可靠回答该子问题。",
                "citations": [],
                "missing": "未检索到相关证据。"
            }

        docs_text = "\n\n".join([
            f"Doc{i+1} | {d['paper_title']} (P{d['page_number']}) | Cluster:{d.get('cluster_label', 'N/A')}\n{d['text'][:320]}"
            for i, d in enumerate(evidence_docs)
        ])

        prompt = f"""You are an evidence-grounded academic QA drafter.
Sub-question: {subq}

Evidence:
{docs_text}

Rules:
1) Use only evidence above.
2) If partially supported, mark PARTIAL and explain missing evidence.
3) Output strict JSON:
{{
  "grounded_level":"HIGH|PARTIAL|LOW",
  "answer":"...",
  "citations":["Title (Pxx)","..."],
  "missing":"..."
}}
"""
        try:
            resp = self.client.chat.completions.create(
                model="qwen3.5-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            text = resp.choices[0].message.content.strip()
            m = re.search(r"\{[\s\S]*\}", text)
            if not m:
                raise ValueError("no json")
            data = json.loads(m.group(0))
            level = str(data.get("grounded_level", "LOW")).upper()
            if level not in {"HIGH", "PARTIAL", "LOW"}:
                level = "LOW"
            return {
                "sub_question": subq,
                "grounded_level": level,
                "answer": data.get("answer", ""),
                "citations": data.get("citations", []),
                "missing": data.get("missing", "")
            }
        except Exception:
            return {
                "sub_question": subq,
                "grounded_level": "LOW",
                "answer": "证据格式化失败，无法可靠回答。",
                "citations": [f"{d['paper_title']} (P{d['page_number']})" for d in evidence_docs[:3]],
                "missing": "解析LLM输出失败"
            }

    def diagnose_missing_type(self, sub_answer: dict, cluster_count: int, evidence_count: int):
        """区分缺失更可能是语料不足还是检索未命中。"""
        level = sub_answer.get("grounded_level", "LOW")
        if level == "HIGH":
            return "NONE"
        if evidence_count <= 2 and cluster_count <= 1:
            return "RETRIEVAL_MISS"
        if evidence_count >= 6 and cluster_count >= 3:
            return "INSUFFICIENT_CORPUS"
        missing = (sub_answer.get("missing", "") or "").lower()
        miss_keywords = ["未检索", "不相关", "not contain", "not relevant", "no information"]
        if any(k in missing for k in miss_keywords):
            return "RETRIEVAL_MISS"
        return "INSUFFICIENT_CORPUS"

    def process_one_subquery(
        self,
        subq: str,
        per_cluster_docs: int,
        max_docs: int,
        retrieve_k: int,
        retrieve_candidate_k: int,
        retrieve_max_per_cluster: int,
        top_cluster_limit: int,
        enable_second_pass: bool,
    ):
        evidence, cluster_cnt = self.route_evidence_for_subquery(
            subq,
            per_cluster_docs=per_cluster_docs,
            max_docs=max_docs,
            retrieve_k=retrieve_k,
            retrieve_candidate_k=retrieve_candidate_k,
            retrieve_max_per_cluster=retrieve_max_per_cluster,
            top_cluster_limit=top_cluster_limit,
        )

        sa = self.answer_subquery(subq, evidence)
        missing_type = self.diagnose_missing_type(sa, cluster_cnt, len(evidence))

        second_pass_used = False
        if enable_second_pass and sa.get("grounded_level") == "LOW":
            expanded_evidence, expanded_cluster_cnt = self.route_evidence_for_subquery(
                subq,
                per_cluster_docs=min(per_cluster_docs + 1, 4),
                max_docs=min(max_docs + 4, 16),
                retrieve_k=min(retrieve_k + 30, 150),
                retrieve_candidate_k=min(retrieve_candidate_k + 120, 600),
                retrieve_max_per_cluster=min(retrieve_max_per_cluster + 4, 20),
                top_cluster_limit=min(top_cluster_limit + 2, 8),
            )
            if len(expanded_evidence) > len(evidence):
                second_sa = self.answer_subquery(subq, expanded_evidence)
                if second_sa.get("grounded_level") in {"HIGH", "PARTIAL"}:
                    sa = second_sa
                    evidence = expanded_evidence
                    cluster_cnt = expanded_cluster_cnt
                    second_pass_used = True
                    missing_type = self.diagnose_missing_type(sa, cluster_cnt, len(evidence))

        sa["evidence_count"] = len(evidence)
        sa["cluster_count"] = cluster_cnt
        sa["missing_type"] = missing_type
        sa["second_pass_used"] = second_pass_used
        return sa

    def synthesize_final(self, query: str, sub_answers):
        pack = []
        for i, sa in enumerate(sub_answers, 1):
            pack.append(
                f"SubQ{i}: {sa['sub_question']}\n"
                f"Level: {sa['grounded_level']}\n"
                f"Answer: {sa['answer']}\n"
                f"Citations: {', '.join(sa.get('citations', []))}\n"
                f"Missing: {sa.get('missing', '')}"
            )

        prompt = f"""You are a final academic answer synthesizer.
Original question: {query}

Sub-question answers:
{chr(10).join(pack)}

Requirements:
1) Output in Chinese with a natural tone, not rigid list-only style.
2) Structure strictly as:
   [总体结论]
   [核心要点]
   [分项说明]
   [证据与局限]
3) Under [分项说明], summarize each sub-question in one short paragraph.
4) Preserve citation attribution and explicitly mention evidence limitations.
"""
        try:
            resp = self.client.chat.completions.create(
                model="qwen3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=650,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            strong = [s for s in sub_answers if s["grounded_level"] in {"HIGH", "PARTIAL"}]
            if strong:
                return "\n\n".join([f"- {s['sub_question']}：{s['answer']}" for s in strong])
            return "证据不足，暂无法可靠回答该问题。"


def main():
    st.title("🧠 Clustered Multi-Document RAG")
    st.markdown("**流程：问题拆解 → 子问题检索与证据路由 → 子问题回答 → 最终证据合成**")

    st.sidebar.header("⚙️ 检索与分解参数")
    min_subqs = st.sidebar.slider("最少子问题数", 1, 4, 2)
    max_subqs = st.sidebar.slider("最多子问题数", min_subqs, 6, 4)
    per_cluster_docs = st.sidebar.slider("每簇取证据条数", 1, 4, 2)
    max_docs = st.sidebar.slider("每个子问题最大证据数", 3, 12, 8)
    retrieve_k = st.sidebar.slider("每子问题检索k", 20, 100, 40, step=10)
    retrieve_candidate_k = st.sidebar.slider("每子问题候选召回", retrieve_k, 400, 220, step=20)
    retrieve_max_per_cluster = st.sidebar.slider("检索阶段每簇上限", 2, 20, 8)
    top_cluster_limit = st.sidebar.slider("子问题路由簇数上限", 2, 8, 4)
    max_workers = st.sidebar.slider("子问题并发数", 1, 8, 4)
    enable_second_pass = st.sidebar.checkbox("低置信子问题启用二次检索回补", value=True)
    refusal_threshold = st.sidebar.slider("拒答阈值(证据置信度低于该值拒答)", 0.0, 1.0, 0.25, step=0.05)

    with st.spinner("🚀 正在加载模型（首次较慢）..."):
        try:
            embed_model, index, id_to_meta, cluster_labels, client = load_models()
            rag = StreamlitRAG(embed_model, index, id_to_meta, cluster_labels, client)
            st.success("✅ 模型加载完成！")
        except Exception as e:
            st.error(f"加载失败: {e}")
            return

    st.divider()
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("请输入您的问题", placeholder="例如：GraphRAG与SelfRAG在检索质量上有何差异？", label_visibility="collapsed")
    with col2:
        submit = st.button("🔍 开始查询", use_container_width=True, type="primary")

    if submit and query:
        process_query(
            rag,
            query,
            min_subqs=min_subqs,
            max_subqs=max_subqs,
            per_cluster_docs=per_cluster_docs,
            max_docs=max_docs,
            retrieve_k=retrieve_k,
            retrieve_candidate_k=retrieve_candidate_k,
            retrieve_max_per_cluster=retrieve_max_per_cluster,
            top_cluster_limit=top_cluster_limit,
            max_workers=max_workers,
            enable_second_pass=enable_second_pass,
            refusal_threshold=refusal_threshold,
        )


def process_query(
    rag: StreamlitRAG,
    query: str,
    min_subqs: int = 2,
    max_subqs: int = 4,
    per_cluster_docs: int = 2,
    max_docs: int = 8,
    retrieve_k: int = 40,
    retrieve_candidate_k: int = 220,
    retrieve_max_per_cluster: int = 8,
    top_cluster_limit: int = 4,
    max_workers: int = 4,
    enable_second_pass: bool = True,
    refusal_threshold: float = 0.25,
):
    t0 = time.time()

    with st.status("🧩 Step 1: Query Decomposition", expanded=True) as status:
        subqs = rag.decompose_query(query, min_subqs=min_subqs, max_subqs=max_subqs)
        st.write(f"拆分出 **{len(subqs)}** 个子问题：")
        for i, sq in enumerate(subqs, 1):
            st.markdown(f"<div class='subq-box'><b>SubQ{i}</b>: {sq}</div>", unsafe_allow_html=True)
        status.update(label="✅ 拆解完成", state="complete")

    sub_answers = []
    with st.status("📚 Step 2-3: 子问题证据路由与回答（并发）", expanded=True) as status:
        results_by_idx = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    rag.process_one_subquery,
                    sq,
                    per_cluster_docs,
                    max_docs,
                    retrieve_k,
                    retrieve_candidate_k,
                    retrieve_max_per_cluster,
                    top_cluster_limit,
                    enable_second_pass,
                ): i
                for i, sq in enumerate(subqs, 1)
            }

            for future in as_completed(futures):
                i = futures[future]
                sq = subqs[i - 1]
                try:
                    results_by_idx[i] = future.result()
                except Exception as e:
                    results_by_idx[i] = {
                        "sub_question": sq,
                        "grounded_level": "LOW",
                        "answer": "子问题处理失败。",
                        "citations": [],
                        "missing": str(e)[:180],
                        "evidence_count": 0,
                        "cluster_count": 0,
                        "missing_type": "RETRIEVAL_MISS",
                        "second_pass_used": False,
                    }

        for i in range(1, len(subqs) + 1):
            sa = results_by_idx[i]
            sub_answers.append(sa)
            missing_type = sa.get("missing_type", "UNKNOWN")
            miss_text = "检索未命中/召回不足" if missing_type == "RETRIEVAL_MISS" else "语料覆盖不足"
            second_pass_note = " | 二次检索: 已触发" if sa.get("second_pass_used") else ""

            st.markdown(f"""
            <div class='answer-box'>
              <b>SubQ{i}</b>: {sa['sub_question']}<br>
              Grounded: <b>{sa['grounded_level']}</b> | Evidence: {sa.get('evidence_count', 0)} docs | Clusters: {sa.get('cluster_count', 0)}{second_pass_note}<br><br>
              <b>回答</b>: {sa['answer']}<br>
              <b>引用</b>: {', '.join(sa.get('citations', [])[:4]) if sa.get('citations') else '无'}<br>
              <b>缺失类型</b>: {miss_text}<br>
              <b>缺失</b>: {sa.get('missing', 'None')}
            </div>
            """, unsafe_allow_html=True)

        status.update(label="✅ 子问题处理完成", state="complete")

    high = sum(1 for s in sub_answers if s["grounded_level"] == "HIGH")
    partial = sum(1 for s in sub_answers if s["grounded_level"] == "PARTIAL")
    low = sum(1 for s in sub_answers if s["grounded_level"] == "LOW")
    conf = (high + 0.5 * partial) / max(len(sub_answers), 1)

    with st.spinner("🎯 Step 4: 按子问题引用合成最终答案..."):
        retrieval_miss_count = sum(1 for s in sub_answers if s.get("missing_type") == "RETRIEVAL_MISS")
        if conf < refusal_threshold:
            final_answer = "证据不足，暂无法可靠回答该问题。\n\n可尝试：扩大检索k、增加每子问题证据数、提高路由簇数上限、启用二次检索回补。"
        else:
            final_answer = rag.synthesize_final(query, sub_answers)
            if retrieval_miss_count > 0:
                final_answer += f"\n\n[说明] 本次有 {retrieval_miss_count} 个子问题更可能属于检索未命中，而非语料绝对缺失。"

    st.subheader("🎯 最终答案（按子问题证据合成）")
    st.markdown(f"""
    <div class="final-box">
        <div style="font-size:16px; line-height:1.8;">{final_answer.replace(chr(10), '<br>')}</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("⏱️ 总耗时", f"{time.time() - t0:.1f}s")
    c2.metric("🧩 子问题数", len(sub_answers))
    c3.metric("🟩 HIGH", high)
    c4.metric("🟨 PARTIAL", partial)
    c5.metric("🟧 LOW", low)

    st.progress(conf, text=f"子问题证据置信度: {conf:.0%}")


if __name__ == "__main__":
    main()
