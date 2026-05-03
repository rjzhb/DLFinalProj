"""End-to-end pipeline: data prep -> baseline -> fine-tune -> evaluate -> visualize."""

import numpy as np
from sentence_transformers import SentenceTransformer

from config import *  # noqa: F403 - includes DEVICE
from src.data_loader import download_fiqa, load_corpus, load_queries, load_qrels, build_training_pairs
from src.baseline_bm25 import bm25_retrieve
from src.retrieval import encode_corpus, build_index, search
from src.evaluate import evaluate_retrieval, print_metrics, save_metrics
from src.train import create_evaluator, fine_tune
from src.visualize import plot_embeddings, plot_comparison


def run_dense_eval(model, corpus, queries, qrels, name):
    """Generic dense retrieval evaluation pipeline."""
    embeddings, doc_ids = encode_corpus(model, corpus)
    index = build_index(embeddings)
    results = search(model, index, doc_ids, queries, top_k=max(TOP_K_VALUES))
    metrics = evaluate_retrieval(results, qrels, TOP_K_VALUES)
    print_metrics(name, metrics)
    save_metrics(name, metrics, RESULTS_DIR)
    return metrics, embeddings


def main():
    # ========== 1. Data preparation ==========
    print("\n[1/5] Data preparation")
    data_path = download_fiqa()
    corpus = load_corpus(data_path)
    test_queries = load_queries(data_path)
    test_qrels = load_qrels(data_path, "test")
    train_qrels = load_qrels(data_path, "train")
    dev_qrels = load_qrels(data_path, "dev")

    # Keep only queries that have qrels
    train_queries = {qid: test_queries[qid] for qid in train_qrels if qid in test_queries}
    dev_queries = {qid: test_queries[qid] for qid in dev_qrels if qid in test_queries}
    eval_queries = {qid: test_queries[qid] for qid in test_qrels if qid in test_queries}

    train_pairs = build_training_pairs(train_queries, corpus, train_qrels)
    print(f"Corpus: {len(corpus)} docs, train: {len(train_pairs)} pairs, "
          f"dev: {len(dev_queries)} queries, test: {len(eval_queries)} queries")

    # ========== 2. BM25 baseline ==========
    print("\n[2/5] BM25 baseline")
    bm25_results = bm25_retrieve(corpus, eval_queries, top_k=max(TOP_K_VALUES))
    bm25_metrics = evaluate_retrieval(bm25_results, test_qrels, TOP_K_VALUES)
    print_metrics("BM25", bm25_metrics)
    save_metrics("bm25", bm25_metrics, RESULTS_DIR)

    # ========== 3. BGE zero-shot baseline ==========
    print("\n[3/5] BGE zero-shot baseline")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    zs_metrics, _ = run_dense_eval(model, corpus, eval_queries, test_qrels, "BGE Zero-shot")

    # ========== 4. Fine-tune ==========
    print("\n[4/5] Fine-tuning")
    dev_evaluator = create_evaluator(dev_queries, corpus, dev_qrels)
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)  # reload to start from pretrained
    model = fine_tune(model, train_pairs, dev_evaluator)

    # ========== 5. Evaluate fine-tuned model ==========
    print("\n[5/5] Evaluating fine-tuned model")
    ft_model = SentenceTransformer(str(MODEL_DIR), device=DEVICE)
    ft_metrics, doc_embs = run_dense_eval(ft_model, corpus, eval_queries, test_qrels, "BGE Fine-tuned")

    # Visualize
    q_embs = ft_model.encode(list(eval_queries.values()), normalize_embeddings=True)
    plot_embeddings(q_embs, doc_embs, "Fine-tuned Embedding Space", "tsne_finetuned.png")
    plot_comparison(zs_metrics, ft_metrics)

    # Summary
    print("\n" + "=" * 50)
    print("  Final results")
    print("=" * 50)
    for name, m in [("BM25", bm25_metrics), ("Zero-shot", zs_metrics), ("Fine-tuned", ft_metrics)]:
        vals = "  ".join(f"{k}: {v:.4f}" for k, v in m.items())
        print(f"  {name:>12s} | {vals}")


if __name__ == "__main__":
    main()
