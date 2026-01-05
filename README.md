# LLM-Based Document Intelligence (RAG)

An end-to-end Retrieval-Augmented Generation (RAG) system for semantic search over PDF documents, built with Sentence Transformers, FAISS, and FastAPI.

---

## Overview

This project implements a complete RAG pipeline:
- PDF ingestion and text chunking
- Dense vector embedding using Sentence Transformers
- FAISS-based similarity search
- REST API for querying documents
- Evaluation and latency benchmarking

---

## Architecture

PDFs → Chunking → Embeddings → FAISS Index → FastAPI Query API

---

## Tech Stack

- Language: Python
- Embedding Model: sentence-transformers/all-MiniLM-L6-v2
- Vector Store: FAISS
- API: FastAPI + Uvicorn

---

## Setup & Usage
Ingest documents
python src/ingest.py
Start API
uvicorn src.api:app --reload
Query
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is this lecture about?", "k":5}'


Evaluation Results
Precision@5: 1.00
Average retrieval latency: 8.2 ms/query
TopK: 5
Runs: 15

Repository Structure
src/
eval/
requirements.txt
README.md

Author
Rithika

### Install dependencies
```bash
pip install -r requirements.txt
