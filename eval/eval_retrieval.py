from src.retriever import get_retriever

# Curated evaluation set (edit expected_keyword after you see your retrieved text)
EVAL_SET = [
    {"query": "What is this lecture about?", "expected_keyword": "knapsack"},
    {"query": "What algorithm is discussed?", "expected_keyword": "greedy"},
    {"query": "What graph concept is mentioned?", "expected_keyword": "graph"},
]

def hit_at_k(results, keyword: str) -> int:
    k = keyword.lower().strip()
    return 1 if any(k in r["text"].lower() for r in results) else 0

def main(top_k: int = 5):
    retrieve = get_retriever()
    hits = 0

    for item in EVAL_SET:
        results, _ = retrieve(item["query"], k=top_k)
        hits += hit_at_k(results, item["expected_keyword"])

    precision = hits / len(EVAL_SET)
    print(f"Precision@{top_k}: {precision:.2f} ({hits}/{len(EVAL_SET)} queries hit)")

if __name__ == "__main__":
    main()

