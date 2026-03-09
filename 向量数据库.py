import json
import numpy as np
import faiss
import pickle
from tqdm import tqdm

input_file = "chunks_with_emb.jsonl"

index_file = "faiss_index.bin"
mapping_file = "id_to_chunk.pkl"

embeddings = []
index_to_metadata = {}

print("Loading embeddings...")

with open(input_file, "r", encoding="utf-8") as f:

    for line in tqdm(f):

        data = json.loads(line)

        emb = np.array(data["embedding"], dtype=np.float32)

        embeddings.append(emb)

        idx = len(embeddings) - 1
        index_to_metadata[idx] = {
            "chunk_id": data["chunk_id"],
            "paper_title": data.get("paper_title", ""),
            "category": data.get("category", ""),
            "page_number": data.get("page_number", 0),
            "text": data["text"],
        }

embeddings = np.vstack(embeddings)

dimension = embeddings.shape[1]

print("Total vectors:", len(embeddings))
print("Dimension:", dimension)

# ===== FAISS index =====

nlist = 100

quantizer = faiss.IndexFlatIP(dimension)

index = faiss.IndexIVFFlat(
    quantizer,
    dimension,
    nlist,
    faiss.METRIC_INNER_PRODUCT
)

print("Training index...")

index.train(embeddings)

print("Adding vectors...")

index.add(embeddings)

print("Saving index...")

faiss.write_index(index, index_file)

with open(mapping_file, "wb") as f:
    pickle.dump({"index_to_metadata": index_to_metadata}, f)

print("Done")