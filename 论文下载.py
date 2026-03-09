import os
import json
import re
import arxiv

# ======== 配置 ========
SAVE_DIR = "papers_by_category"
os.makedirs(SAVE_DIR, exist_ok=True)

PAPERS_PER_DIRECTION = 25
YEAR_RANGE = (2021, 2026)

# 面向“计算机论文问答”的高价值方向（可继续扩展）
DIRECTIONS = {
    "RAG_Core": [
        "retrieval augmented generation",
        "self rag",
        "corrective rag",
        "adaptive rag"
    ],
    "LLM_Reasoning": [
        "chain of thought reasoning",
        "tree of thoughts",
        "tool use llm",
        "planning with llm"
    ],
    "Agent_and_ToolLearning": [
        "llm agent",
        "multi agent collaboration",
        "function calling llm",
        "agentic workflow"
    ],
    "Code_LLM_and_SE": [
        "code language model",
        "repo level code generation",
        "program repair llm",
        "software engineering with llm"
    ],
    "IR_and_Reranking": [
        "dense retrieval",
        "cross encoder reranking",
        "hybrid retrieval",
        "learning to rank neural"
    ],
    "Hallucination_and_Factuality": [
        "llm hallucination",
        "factuality evaluation",
        "grounded generation",
        "attribution in rag"
    ],
    "LongContext_and_Compression": [
        "long context language model",
        "context compression",
        "kv cache compression",
        "memory for llm"
    ],
    "Multimodal_and_VLM": [
        "vision language model",
        "multimodal retrieval",
        "video llm",
        "document understanding model"
    ],
    "Embodied_and_RoboticsAI": [
        "embodied ai",
        "vision language action",
        "robot foundation model",
        "sim2real robot learning"
    ],
    "Graph_and_Knowledge": [
        "knowledge graph for llm",
        "graph retrieval augmented generation",
        "graph neural retrieval",
        "neuro symbolic reasoning"
    ]
}


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    return name[:180]


def paper_id_from_entry(entry_id: str) -> str:
    return entry_id.rsplit("/", 1)[-1]


def download_direction(direction: str, keywords: list, global_seen: set, all_meta: list):
    direction_dir = os.path.join(SAVE_DIR, direction)
    os.makedirs(direction_dir, exist_ok=True)

    count = 0
    print(f"\n===== {direction} =====")

    for kw in keywords:
        if count >= PAPERS_PER_DIRECTION:
            break

        search = arxiv.Search(
            query=kw,
            max_results=120,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        for result in search.results():
            if count >= PAPERS_PER_DIRECTION:
                break

            year = result.published.year
            if year < YEAR_RANGE[0] or year > YEAR_RANGE[1]:
                continue

            arxiv_id = paper_id_from_entry(result.entry_id)
            if arxiv_id in global_seen:
                continue

            title = result.title.strip().replace("\n", " ")
            safe_name = sanitize_filename(f"{arxiv_id}_{title}") + ".pdf"
            pdf_path = os.path.join(direction_dir, safe_name)

            try:
                result.download_pdf(filename=pdf_path)
                print(f"[{direction}] {count + 1}: {title}")

                meta = {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "authors": [a.name for a in result.authors],
                    "published": result.published.strftime("%Y-%m-%d"),
                    "summary": result.summary,
                    "pdf_path": pdf_path,
                    "direction": direction,
                    "matched_keyword": kw,
                }
                all_meta.append(meta)
                global_seen.add(arxiv_id)
                count += 1
            except Exception as e:
                print(f"Failed: {title[:80]}... | {e}")


def main():
    all_meta = []
    global_seen = set()

    for direction, kws in DIRECTIONS.items():
        download_direction(direction, kws, global_seen, all_meta)

    meta_path = os.path.join(SAVE_DIR, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)

    print("\nDone")
    print(f"Total papers: {len(all_meta)}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
