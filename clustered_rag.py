"""Clustered Multi-Document RAG - Improved Version"""

import json
import pickle
import numpy as np
import faiss
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
import config


class ClusteredRAG:

    def __init__(self):

        print("🚀 初始化 Clustered RAG Pipeline...")

        # Embedding model
        print("  📥 加载Embedding模型...")
        self.embed_model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            device="cpu"
        )

        # Reranker
        self.reranker = None
        if config.USE_RERANKER:
            print("  📥 加载Reranker...")
            self.reranker = CrossEncoder(
                config.RERANKER_MODEL,
                device="cpu"
            )

        # FAISS
        print("  📥 加载FAISS索引...")
        self.index = faiss.read_index(config.FAISS_INDEX)

        with open(config.ID_MAPPING, "rb") as f:
            self.id_to_meta = pickle.load(f)["index_to_metadata"]

        # Cluster labels
        with open(config.CLUSTER_LABELS, "r", encoding="utf-8") as f:
            self.cluster_labels = {
                int(k): v for k, v in json.load(f).items()
            }

        # LLM client
        self.client = OpenAI(
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.DASHSCOPE_BASE_URL,
            timeout=config.REQUEST_TIMEOUT
        )

        print(
            f"  ✅ 系统就绪 | 索引大小: {self.index.ntotal} | "
            f"Clusters: {len(self.cluster_labels)}"
        )

    # =====================================================
    # Step 1 Retrieval
    # =====================================================

    def retrieve(self, query: str) -> List[Dict]:

        # BGE requires query prefix
        query_emb = self.embed_model.encode(
            ["query: " + query],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)

        scores, indices = self.index.search(
            query_emb,
            config.TOP_K
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            meta = self.id_to_meta[idx]

            results.append({
                "score": float(score),
                "chunk_id": meta.get("chunk_id", idx),
                "text": meta["text"],
                "cluster_id": meta.get("cluster_id", 0),
                "paper_title": meta.get("paper_title", "Unknown"),
                "category": meta.get("category", ""),
                "page_number": meta.get("page_number", 0),
            })

        # Debug cluster distribution
        cluster_dist = defaultdict(int)
        for r in results:
            cluster_dist[r["cluster_id"]] += 1

        print("\n📊 Retrieval cluster distribution:")
        for cid, count in sorted(
                cluster_dist.items(),
                key=lambda x: -x[1]):
            print(f"   Cluster {cid}: {count}")

        return results

    # =====================================================
    # Step 2 Rerank
    # =====================================================

    def rerank(self, query: str, chunks: List[Dict]) -> List[Dict]:

        if not self.reranker or not chunks:
            return chunks[:config.RERANK_TOP_K]

        pairs = [[query, chunk["text"]] for chunk in chunks]

        rerank_scores = self.reranker.predict(pairs)

        for chunk, score in zip(chunks, rerank_scores):
            chunk["rerank_score"] = float(score)

        chunks = sorted(
            chunks,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return chunks[:config.RERANK_TOP_K]

    # =====================================================
    # Step 3 Cluster grouping
    # =====================================================

    def cluster_and_filter(
            self,
            chunks: List[Dict]
    ) -> List[Tuple[int, List[Dict], float]]:

        clusters = defaultdict(list)

        for chunk in chunks:
            clusters[chunk["cluster_id"]].append(chunk)

        cluster_stats = []

        for cid, docs in clusters.items():

            scores = [
                d.get("rerank_score", d["score"])
                for d in docs
            ]

            max_score = max(scores)
            mean_score = np.mean(scores)

            cluster_score = 0.7 * max_score + 0.3 * mean_score

            cluster_stats.append(
                (cid, docs, cluster_score)
            )

        cluster_stats.sort(
            key=lambda x: x[2],
            reverse=True
        )

        top_clusters = cluster_stats[:config.MAX_CLUSTERS]

        print(f"\n📊 Top Clusters:")

        for cid, docs, score in top_clusters:
            label = self.cluster_labels.get(
                cid,
                f"Cluster_{cid}"
            )

            print(
                f"   {label} | docs={len(docs)} | "
                f"score={score:.3f}"
            )

        return top_clusters

    # =====================================================
    # Step 4 Draft generation
    # =====================================================

    def generate_draft(
            self,
            query: str,
            cluster_id: int,
            docs: List[Dict],
            label: str
    ) -> Dict:

        doc_texts = []
        sources = []

        for i, doc in enumerate(docs[:6]):

            text = doc["text"][:400]

            doc_texts.append(
                f"[Document {i+1}] {text}"
            )

            sources.append({
                "paper": doc["paper_title"],
                "page": doc["page_number"],
                "category": doc["category"]
            })

        documents = "\n\n".join(doc_texts)

        prompt = f"""
Research Topic: {label}

Question: {query}

Documents:
{documents}

Generate a technical draft answer based ONLY on the documents.
Include citations in format [Source: Paper Title, Page X].
"""

        try:

            response = self.client.chat.completions.create(

                model=config.MODEL_CONFIG["drafter"]["model"],

                messages=[
                    {
                        "role": "system",
                        "content": "You are a research assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=config.MODEL_CONFIG["drafter"]["temperature"],
                max_tokens=config.MODEL_CONFIG["drafter"]["max_tokens"],
            )

            draft = response.choices[0].message.content.strip()

            return {
                "cluster_id": cluster_id,
                "cluster_label": label,
                "draft": draft,
                "sources": sources,
                "doc_count": len(docs),
                "status": "success"
            }

        except Exception as e:

            print(f"❌ Cluster {cluster_id} error: {e}")

            return {
                "cluster_id": cluster_id,
                "cluster_label": label,
                "draft": "",
                "sources": sources,
                "status": "failed"
            }

    # =====================================================
    # Step 5 Parallel drafting
    # =====================================================

    def parallel_drafting(
            self,
            query: str,
            clusters_info
    ) -> List[Dict]:

        print(
            f"\n📝 Generating {len(clusters_info)} drafts..."
        )

        drafts = []

        with ThreadPoolExecutor(
                max_workers=config.MAX_WORKERS) as executor:

            futures = {}

            for cid, docs, score in clusters_info:

                label = self.cluster_labels.get(
                    cid,
                    f"Cluster_{cid}"
                )

                future = executor.submit(
                    self.generate_draft,
                    query,
                    cid,
                    docs,
                    label
                )

                futures[future] = label

            for future in as_completed(futures):

                draft = future.result()

                if draft["status"] == "success":
                    drafts.append(draft)

        drafts.sort(key=lambda x: x["cluster_id"])

        return drafts

    # =====================================================
    # Step 6 Evaluator
    # =====================================================

    def evaluate_and_synthesize(
            self,
            query: str,
            drafts: List[Dict]
    ) -> Dict:

        if not drafts:
            return {"answer": "No drafts generated."}

        if len(drafts) == 1:
            return {
                "answer": drafts[0]["draft"],
                "citations": drafts[0]["sources"]
            }

        candidates = []

        for i, draft in enumerate(drafts, 1):

            candidates.append(
                f"""
Candidate {i} ({draft['cluster_label']}):
{draft['draft']}
"""
            )

        prompt = f"""
Question:
{query}

Candidate Answers:
{''.join(candidates)}

Combine the best insights into one final answer.
"""

        response = self.client.chat.completions.create(

            model=config.MODEL_CONFIG["evaluator"]["model"],

            messages=[
                {
                    "role": "system",
                    "content": "You synthesize research answers."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=config.MODEL_CONFIG["evaluator"]["temperature"],
            max_tokens=config.MODEL_CONFIG["evaluator"]["max_tokens"],
        )

        answer = response.choices[0].message.content.strip()

        all_sources = []
        for d in drafts:
            all_sources.extend(d["sources"])

        return {
            "answer": answer,
            "citations": all_sources
        }

    # =====================================================
    # Full pipeline
    # =====================================================

    def run(self, query: str) -> Dict:

        print("\n" + "="*60)
        print("QUERY:", query)
        print("="*60)

        chunks = self.retrieve(query)

        if config.USE_RERANKER:
            chunks = self.rerank(query, chunks)

        clusters = self.cluster_and_filter(chunks)

        drafts = self.parallel_drafting(query, clusters)

        result = self.evaluate_and_synthesize(query, drafts)

        print("\n📄 FINAL ANSWER:\n")
        print(result["answer"])

        return result