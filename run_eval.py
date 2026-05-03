"""Step 4: evaluate fine-tuned model against baselines and produce all visualizations."""

import json
from sentence_transformers import SentenceTransformer

from config import FIQA_DIR, MODEL_DIR, DEVICE, TOP_K_VALUES, RESULTS_DIR
from src.data_loader import load_corpus, load_queries, load_qrels
from src.retrieval import encode_corpus, build_index, search
from src.evaluate import evaluate_retrieval, print_metrics, save_metrics
from src.visualize import (
    plot_embeddings, plot_comparison, plot_training_curves, plot_improvement_heatmap
)

# Load data
corpus = load_corpus(FIQA_DIR)
queries = load_queries(FIQA_DIR)
test_qrels = load_qrels(FIQA_DIR, "test")
eval_queries = {qid: queries[qid] for qid in test_qrels if qid in queries}
print(f"Corpus: {len(corpus)} docs, test queries: {len(eval_queries)}")

# ===== Evaluate fine-tuned model =====
print("\n[1/4] Evaluating fine-tuned model")
ft_model = SentenceTransformer(str(MODEL_DIR), device=DEVICE)
ft_embeddings, doc_ids = encode_corpus(ft_model, corpus, cache_name="bge_finetuned")
ft_index = build_index(ft_embeddings)
ft_results = search(ft_model, ft_index, doc_ids, eval_queries, top_k=max(TOP_K_VALUES))
ft_metrics = evaluate_retrieval(ft_results, test_qrels, TOP_K_VALUES)
print_metrics("BGE Fine-tuned", ft_metrics)
save_metrics("bge_finetuned", ft_metrics, RESULTS_DIR)

# ===== Load baseline results =====
print("[2/4] Loading baseline results")
bm25_metrics = json.loads((RESULTS_DIR / "bm25.json").read_text())
zs_metrics = json.loads((RESULTS_DIR / "bge_zeroshot.json").read_text())

# ===== Generate visualizations =====
print("\n[3/4] Generating visualizations")

# Figure 1: training curves
print("  training curves...")
plot_training_curves(str(MODEL_DIR / "eval" / "Information-Retrieval_evaluation_dev_results.csv"))

# Figure 2: 3-model comparison bar chart
print("  comparison bar chart...")
plot_comparison(zs_metrics, ft_metrics, bm25_metrics=bm25_metrics)

# Figure 3: t-SNE embedding visualization
print("  t-SNE visualization...")
q_embs = ft_model.encode(
    list(eval_queries.values()), normalize_embeddings=True,
    show_progress_bar=True, device=DEVICE
)
plot_embeddings(q_embs, ft_embeddings, "Fine-tuned Embedding Space", "tsne_finetuned.png")

# Figure 4: summary table
print("  summary table...")
plot_improvement_heatmap(bm25_metrics, zs_metrics, ft_metrics)

# ===== Summary =====
print("\n[4/4] Final results")
print("=" * 60)
print("  Final results")
print("=" * 60)
for name, m in [("BM25", bm25_metrics), ("BGE Zero-shot", zs_metrics), ("BGE Fine-tuned", ft_metrics)]:
    vals = "  ".join(f"{k}: {v:.4f}" for k, v in m.items())
    print(f"  {name:>16s} | {vals}")

print("\n  Fine-tuned vs Zero-shot improvement:")
for k in ft_metrics:
    diff = ft_metrics[k] - zs_metrics[k]
    pct = diff / zs_metrics[k] * 100
    print(f"    {k:>12s}: {zs_metrics[k]:.4f} -> {ft_metrics[k]:.4f} ({pct:+.1f}%)")

print("\n  All figures saved to outputs/figures/")
