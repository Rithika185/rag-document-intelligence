import os
import pickle

STORE_PATH = "data/vectorstore.pkl"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_store(index, chunks, meta: dict):
    with open(STORE_PATH, "wb") as f:
        pickle.dump(
            {
                "index": index,
                "chunks": chunks,
                "meta": meta,
            },
            f,
        )

def load_store():
    if not os.path.exists(STORE_PATH):
        raise FileNotFoundError(
            "Vector store not found. Run src/ingest.py first."
        )
    with open(STORE_PATH, "rb") as f:
        return pickle.load(f)

