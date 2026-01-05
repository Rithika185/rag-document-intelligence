import time
from src.retriever import get_retriever

QUERIES = [
    "What is this lecture about?",
    "Explain knapsack problem briefly.",
    "What is greedy algorithm?",
]

def main(runs: int = 15, k: int = 5):
    retrieve = get_retriever()

    # warmup
    for q in QUERIES:
        retrieve(q, k=k)

    times = []
    for _ in range(runs):
        start = time.time()
        for q in QUERIES:
            retrieve(q, k=k)
        times.append(time.time() - start)

    avg_total = sum(times) / len(times)
    avg_per_query_ms = (avg_total / len(QUERIES)) * 1000

    print(f"Runs: {runs}, Queries per run: {len(QUERIES)}, TopK: {k}")
    print(f"Avg total time per run: {avg_total:.3f}s")
    print(f"Avg retrieval latency per query: {avg_per_query_ms:.1f} ms")

if __name__ == "__main__":
    main()

