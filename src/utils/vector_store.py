import hashlib
import os
import json
from typing import List, Dict
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from utils.chunking import semantic_merge

INDEX_PATH = "data/faiss_index"
CHUNK_METADATA_FILE = "data/index_metadata.json"
DOC_METADATA_FILE = "data/doc_metadata.json"

os.makedirs("data", exist_ok=True)


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, data: Dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


async def chunk_embed_store_documents(docs: List[Dict]) -> Dict:
    chunk_index = load_json(CHUNK_METADATA_FILE)
    doc_index = load_json(DOC_METADATA_FILE)

    new_documents = []
    new_doc_hashes = []

    for doc in docs:
        text = doc["text"]
        base_metadata = doc.get("metadata", {})
        source_id = (
            base_metadata.get("source_id")
            or base_metadata.get("source_url")
            or "unknown"
        )

        doc_hash = compute_hash(text)
        if doc_hash in doc_index:
            print(
                f"[chunk_embed_store_documents] Skipping already indexed doc: {source_id}"
            )
            continue

        chunks = await semantic_merge(text)
        chunk_hashes = []
        for i, chunk in enumerate(chunks):
            chunk_hash = compute_hash(chunk)
            if chunk_hash in chunk_index:
                continue  # Skip duplicate chunk
            metadata = base_metadata.copy()
            metadata.update(
                {
                    "chunk_index": i,
                    "chunk_hash": chunk_hash,
                    "source_type": base_metadata.get("source_type", "unknown"),
                }
            )
            new_documents.append(Document(page_content=chunk, metadata=metadata))
            chunk_index[chunk_hash] = {"source_id": source_id}
            chunk_hashes.append(chunk_hash)

        if chunk_hashes:
            doc_index[doc_hash] = {
                "source_id": source_id,
                "num_chunks": len(chunk_hashes),
                "chunk_hashes": chunk_hashes,
            }
            new_doc_hashes.append(doc_hash)

    if not new_documents:
        return {"message": "No new documents or chunks to store."}

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(INDEX_PATH):
        vectordb = FAISS.load_local(INDEX_PATH, embeddings)
    else:
        vectordb = FAISS.from_documents([], embeddings)

    vectordb.add_documents(new_documents)
    vectordb.save_local(INDEX_PATH)

    save_json(CHUNK_METADATA_FILE, chunk_index)
    save_json(DOC_METADATA_FILE, doc_index)

    return {
        "message": f"Stored {len(new_documents)} new chunks from {len(new_doc_hashes)} documents",
        "index_path": INDEX_PATH,
    }
