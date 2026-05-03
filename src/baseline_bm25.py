"""BM25 lexical retrieval baseline."""

from rank_bm25 import BM25Okapi
from tqdm import tqdm


def bm25_retrieve(corpus: dict[str, str], queries: dict[str, str],
                  top_k: int = 10) -> dict[str, list[str]]:
    """BM25 retrieval, returns {query_id: [doc_id, ...]}."""
    doc_ids = list(corpus.keys())
    # Simple whitespace tokenization
    print("Building BM25 index...")
    tokenized = [corpus[did].lower().split() for did in doc_ids]
    bm25 = BM25Okapi(tokenized)

    results = {}
    for qid, query in tqdm(queries.items(), desc="BM25 retrieval"):
        scores = bm25.get_scores(query.lower().split())
        top_indices = scores.argsort()[-top_k:][::-1]
        results[qid] = [doc_ids[i] for i in top_indices]
    return results
