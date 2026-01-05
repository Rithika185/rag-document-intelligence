import os
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from src.utils import ensure_dir, save_store

# Lightweight, stable embedding model for local RAG
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    text_parts = []
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text_parts.append(content)
    return "\n".join(text_parts)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80):
    """
    Simple chunker (character-based) to keep dependencies minimal and stable.
    Smaller chunks reduce memory spikes during embedding.
    """
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward with overlap
        start = end - overlap
        if start < 0:
            start = 0
        if end == n:
            break

    return chunks

def ingest(folder: str = "data/sample_docs", batch_size: int = 32):
    ensure_dir("data")

    documents = []
    sources = []

    # 1) Load PDFs and chunk
    if not os.path.exists(folder):
        raise RuntimeError(f"Folder not found: {folder}")

    for fname in os.listdir(folder):
        if not fname.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(folder, fname)
        raw_text = read_pdf(pdf_path)

        if not raw_text.strip():
            continue

        chunks = chunk_text(raw_text, chunk_size=500, overlap=80)
        documents.extend(chunks)
        sources.extend([fname] * len(chunks))

    if not documents:
        raise RuntimeError("No text chunks found. Add at least one readable PDF to data/sample_docs/")

    print(f"✅ Loaded {len(documents)} chunks from PDFs in {folder}")

    # 2) Embed in batches to avoid OS killing the process due to memory spikes
    model = SentenceTransformer(EMBED_MODEL)

    embeddings_list = []
    total = len(documents)

    for i in range(0, total, batch_size):
        batch = documents[i:i + batch_size]
        batch_emb = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings_list.append(batch_emb)

        if (i // batch_size) % 10 == 0:
            print(f"Embedding progress: {min(i + batch_size, total)}/{total}")

    embeddings = np.vstack(embeddings_list).astype("float32")

    # 3) Build FAISS index (cosine similarity via normalized vectors + inner product)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    meta = {
        "embedding_model": EMBED_MODEL,
        "num_chunks": len(documents),
        "chunk_size": 500,
        "overlap": 80,
        "batch_size": batch_size,
    }

    # Save vector store
    save_store(
        index=index,
        chunks=[{"text": t, "source": s} for t, s in zip(documents, sources)],
        meta=meta,
    )

    print(f"✅ Indexed {len(documents)} chunks and saved to data/vectorstore.pkl")

if __name__ == "__main__":
    ingest()

