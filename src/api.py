from fastapi import FastAPI
from pydantic import BaseModel
from src.retriever import get_retriever
from src.utils import load_store

app = FastAPI(title="RAG Document Intelligence")

class Query(BaseModel):
    query: str
    k: int = 5

retriever = None

@app.on_event("startup")
def startup_event():
    global retriever
    load_store()
    retriever = get_retriever()
    print("✅ Retriever ready")

@app.post("/query")
def query_docs(q: Query):
    results, meta = retriever(q.query, q.k)
    return {
        "meta": meta,
        "results": results,
    }

