import uuid
import json
import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from .summarizers import summarize_table, summarize_image_from_data_uri
from .utils import prepend_metadata
import faiss
import os

from pydantic import AnyUrl

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, AnyUrl):
        return str(obj)
    else:
        return obj
    
def safe_json_dump(data, target_file):
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json') as tmp:
        json.dump(data, tmp, indent=2)
        temp_path = tmp.name
    os.replace(temp_path, target_file)


class FaissStorage:
    def __init__(self, embedding_dim, index_path="faiss.index", metadata_path="faiss_meta.json"):
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.metadata_path = metadata_path

        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.metadata = {}

    def store_documents(self, docs: list[Document], embedder: SentenceTransformer):
        vectors = []
        ids = []

        for doc in docs:
            metadata = doc.metadata
            doc_type = metadata.get("type")
            meta_str = prepend_metadata(metadata)

            if doc_type == "text":
                content = f"{meta_str}\n{doc.page_content}"
                vector = embedder.encode(content).astype("float32")

            elif doc_type == "table":
                summary = summarize_table(doc.page_content)
                content = f"{meta_str}\n{summary}"
                vector = embedder.encode(summary).astype("float32")

            elif doc_type == "image":
                image_data_uri = metadata.get("image_data_uri")
                summary = summarize_image_from_data_uri(image_data_uri)
                content = f"{meta_str}\n{summary}"
                vector = embedder.encode(summary).astype("float32")

            else:
                continue

            vec_id = str(uuid.uuid4())
            vectors.append(vector)
            ids.append(vec_id)

            # Store content and metadata keyed by vector id
            self.metadata[vec_id] = {
                "content": content,
                "metadata": metadata
            }

        if vectors:
            vectors_np = np.vstack(vectors)
            self.index.add(vectors_np)
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                safe_json_dump(serializable_metadata, self.metadata_path)
            print(f"Inserted {len(ids)} records into FAISS index.")
