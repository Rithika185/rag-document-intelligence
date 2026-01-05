import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils import load_store

def get_retriever():
    store = load_store()
    index = store["index"]
    chunks = store["chunks"]
    meta = store["meta"]

    model = SentenceTransformer(meta["embedding_model"])

    def retrieve(query: str, k=5):
        q_emb = model.encode(
            [query],
            normalize_embeddings=True,
        ).astype("float32")

        scores, idxs = index.search(q_emb, k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            results.append(
                {
                    "score": float(score),
                    "text": chunks[idx]["text"],
                    "source": chunks[idx]["source"],
                }
            )
        return results, meta

    return retrieve

