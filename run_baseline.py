"""Step 2: baseline evaluation (BM25 + BGE zero-shot)."""

from sentence_transformers import SentenceTransformer

from config import FIQA_DIR, MODEL_NAME, DEVICE, TOP_K_VALUES, RESULTS_DIR
from src.data_loader import load_corpus, load_queries, load_qrels
from src.baseline_bm25 import bm25_retrieve
from src.retrieval import encode_corpus, build_index, search
from src.evaluate import evaluate_retrieval, print_metrics, save_metrics

# Load data
corpus = load_corpus(FIQA_DIR)
queries = load_queries(FIQA_DIR)
test_qrels = load_qrels(FIQA_DIR, "test")
eval_queries = {qid: queries[qid] for qid in test_qrels if qid in queries}
print(f"Corpus: {len(corpus)} docs, test queries: {len(eval_queries)}")

# ===== BM25 baseline =====
print("\n[1/2] BM25 baseline")
bm25_results = bm25_retrieve(corpus, eval_queries, top_k=max(TOP_K_VALUES))
bm25_metrics = evaluate_retrieval(bm25_results, test_qrels, TOP_K_VALUES)
print_metrics("BM25", bm25_metrics)
save_metrics("bm25", bm25_metrics, RESULTS_DIR)

# ===== BGE zero-shot baseline =====
print("[2/2] BGE zero-shot")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)
embeddings, doc_ids = encode_corpus(model, corpus, cache_name="bge_zeroshot")
index = build_index(embeddings)
results = search(model, index, doc_ids, eval_queries, top_k=max(TOP_K_VALUES))
zs_metrics = evaluate_retrieval(results, test_qrels, TOP_K_VALUES)
print_metrics("BGE Zero-shot", zs_metrics)
save_metrics("bge_zeroshot", zs_metrics, RESULTS_DIR)

# Summary
print("=" * 50)
print("  Baseline results")
print("=" * 50)
for name, m in [("BM25", bm25_metrics), ("BGE Zero-shot", zs_metrics)]:
    vals = "  ".join(f"{k}: {v:.4f}" for k, v in m.items())
    print(f"  {name:>14s} | {vals}")
