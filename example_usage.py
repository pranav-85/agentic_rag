import tempfile
import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from src.database.storage import FaissStorage
from src.parser.document_parser import parse_pdf

# Initialize embedding model once
embedder = SentenceTransformer("all-mpnet-base-v2")

# Initialize FAISS storage globally with the embedding dimension of the model
faiss_storage = FaissStorage(embedding_dim=768)  # adjust dim as per your model

def main():
    st.title("Agentic RAG: Upload Legal PDFs")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if not uploaded_file:
        st.info("Please upload a PDF file to start.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_file_path = tmp_file.name

    docs_data = parse_pdf(tmp_file_path, embeddings_tokenizer=embedder.tokenizer)

    # Don't forget to clean up the temp file after use
    os.unlink(tmp_file_path)

    all_docs = docs_data.get("texts", []) + docs_data.get("tables", []) + docs_data.get("images", [])

    faiss_storage.store_documents(all_docs, embedder)

    st.success(f"Uploaded and stored {len(all_docs)} chunks from {uploaded_file.name}.")


if __name__ == "__main__":
    main()