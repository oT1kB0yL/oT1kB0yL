from typing import Dict

PAGE_RETRIEVAL_CONFIG: Dict[str, float] = {
    "bm25_k1": 1.5,
    "bm25_b": 0.75,
    "keyword_weight": 0.4,
    "bm25_weight": 0.3,
    "vector_weight": 0.3,
    "page_score_top_k": 3,
}
