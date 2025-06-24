import hashlib
import os
import json
from typing import List, Dict
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from utils.chunking import semantic_merge

INDEX_PATH = "data/faiss_index"
METADATA_FILE = "data/index_metadata.json"

os.makedirs("data", exist_ok=True)


# Helper to hash chunk content
def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_index_metadata() -> Dict:
    if not os.path.exists(METADATA_FILE):
        return {"chunks": {}}
    with open(METADATA_FILE, "r") as f:
        return json.load(f)


def save_index_metadata(metadata: Dict):
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)


async def chunk_embed_store_documents(docs: List[Dict]) -> Dict:
    metadata_index = load_index_metadata()
    new_documents = []
    new_hashes = []

    for doc in docs:
        text = doc["text"]
        base_metadata = doc.get("metadata", {})

        chunks = await semantic_merge(text)
        for i, chunk in enumerate(chunks):
            chunk_hash = compute_hash(chunk)
            if chunk_hash in metadata_index["chunks"]:
                continue  # skip duplicate
            metadata = base_metadata.copy()
            metadata.update(
                {
                    "chunk_index": i,
                    "chunk_hash": chunk_hash,
                    "source_type": base_metadata.get("source_type", "unknown"),
                }
            )
            new_documents.append(Document(page_content=chunk, metadata=metadata))
            new_hashes.append(chunk_hash)

    if not new_documents:
        return {"message": "No new chunks to store"}

    # Load or initialize vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(INDEX_PATH):
        vectordb = FAISS.load_local(INDEX_PATH, embeddings)
    else:
        vectordb = FAISS.from_documents([], embeddings)

    vectordb.add_documents(new_documents)
    vectordb.save_local(INDEX_PATH)

    # Update metadata store
    for h in new_hashes:
        metadata_index["chunks"][h] = {"indexed": True}
    save_index_metadata(metadata_index)

    return {
        "message": f"Stored {len(new_documents)} new chunks",
        "index_path": INDEX_PATH,
        "num_chunks": len(new_documents),
    }
