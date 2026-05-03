"""BGE-base-en-v1.5 zero-shot baseline."""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

from config import FIQA_DIR, DEVICE, TOP_K_VALUES, RESULTS_DIR
from src.data_loader import load_corpus, load_queries, load_qrels
from src.retrieval import encode_corpus, build_index, search
from src.evaluate import evaluate_retrieval, print_metrics

corpus = load_corpus(FIQA_DIR)
queries = load_queries(FIQA_DIR)
test_qrels = load_qrels(FIQA_DIR, "test")
eval_queries = {qid: queries[qid] for qid in test_qrels if qid in queries}
print(f"Corpus: {len(corpus)} docs, test queries: {len(eval_queries)}")

print("\n===== BGE-base-en-v1.5 zero-shot =====")
model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=DEVICE)
embeddings, doc_ids = encode_corpus(model, corpus, cache_name="bge_base_zeroshot")
index = build_index(embeddings)
results = search(model, index, doc_ids, eval_queries, top_k=max(TOP_K_VALUES))
zs_metrics = evaluate_retrieval(results, test_qrels, TOP_K_VALUES)
print_metrics("BGE-base-en-v1.5 Zero-shot", zs_metrics)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(RESULTS_DIR / "bge_base_zeroshot.json", "w") as f:
    json.dump(zs_metrics, f, indent=2)
print(f"Saved to {RESULTS_DIR}/bge_base_zeroshot.json")
