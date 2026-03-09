import os
import fitz
import json
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 配置
papers_dir = "papers_by_category"
output_file = "chunks.jsonl"
chunk_size = 500
chunk_overlap = 100

print(f"[INFO] 检查工作目录: {os.getcwd()}")
print(f"[INFO] 输入目录绝对路径: {os.path.abspath(papers_dir)}")
print(f"[INFO] 输出文件绝对路径: {os.path.abspath(output_file)}")

# 检查输入目录是否存在
if not os.path.exists(papers_dir):
    print(f"[ERROR] 错误：输入目录不存在！{papers_dir}")
    exit(1)

# 列出所有子目录
categories = [d for d in os.listdir(papers_dir) if os.path.isdir(os.path.join(papers_dir, d))]
print(f"[DIR] 发现分类目录: {categories}")

if not categories:
    print("[WARN] 警告：没有发现任何子分类目录")
    print(f"    当前papers_dir内容: {os.listdir(papers_dir)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


chunk_counter = 0
total_pdfs = 0
skipped_pdfs = 0

with open(output_file, "w", encoding="utf-8") as out_f:
    for category in categories:
        cat_path = os.path.join(papers_dir, category)
        print(f"\n[CAT] 处理分类: {category} (路径: {cat_path})")

        pdf_files = [f for f in os.listdir(cat_path) if f.lower().endswith(".pdf")]
        print(f"   发现 {len(pdf_files)} 个PDF文件: {pdf_files}")

        if not pdf_files:
            print(f"   [WARN] 目录 {cat_path} 下没有PDF文件")
            continue

        for paper_index, pdf_file in enumerate(pdf_files):
            total_pdfs += 1
            pdf_path = os.path.join(cat_path, pdf_file)
            print(f"   [{paper_index + 1}/{len(pdf_files)}] 处理: {pdf_file}")

            try:
                doc = fitz.open(pdf_path)
                print(f"      OK PDF打开成功，共 {len(doc)} 页")

                for page_num, page in enumerate(doc, 1):
                    try:
                        raw_text = page.get_text()
                        text_len = len(raw_text.strip())
                        print(f"      第{page_num}页: 原始文本长度={text_len}", end="")

                        if text_len < 10:
                            print(" -> 跳过(太短)")
                            continue

                        cleaned_text = clean_text(raw_text)
                        chunks = splitter.split_text(cleaned_text)
                        print(f" -> 分块数量: {len(chunks)}")

                        for chunk_idx, chunk_text in enumerate(chunks):
                            if len(chunk_text.strip()) < 20:
                                continue

                            chunk_data = {
                                "chunk_id": f"{category}_{paper_index}_{chunk_idx}",
                                "paper_title": pdf_file.replace(".pdf", ""),
                                "category": category,
                                "page_number": page_num,
                                "text": chunk_text
                            }
                            out_f.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")
                            chunk_counter += 1

                    except Exception as page_e:
                        print(f"\n      [ERROR] 第{page_num}页错误: {page_e}")
                        continue

                doc.close()

            except Exception as e:
                skipped_pdfs += 1
                print(f"      [ERROR] PDF打开失败: {e}")
                continue

print(f"\n{'=' * 50}")
print(f"[STAT] 处理统计:")
print(f"   总PDF数量: {total_pdfs}")
print(f"   失败数量: {skipped_pdfs}")
print(f"   生成chunks: {chunk_counter}")
print(f"[SAVE] 输出文件: {os.path.abspath(output_file)}")
print(f"   文件大小: {os.path.getsize(output_file)} bytes")