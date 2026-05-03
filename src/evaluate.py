"""Retrieval evaluation metrics: nDCG@K, Recall@K."""

import numpy as np
import json
from pathlib import Path


def ndcg_at_k(retrieved: list[str], qrel: dict[str, int], k: int) -> float:
    """Compute nDCG@K for a single query."""
    # DCG
    dcg = sum(qrel.get(did, 0) / np.log2(i + 2)
              for i, did in enumerate(retrieved[:k]))
    # IDCG
    ideal = sorted(qrel.values(), reverse=True)[:k]
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved: list[str], qrel: dict[str, int], k: int) -> float:
    """Compute Recall@K for a single query."""
    relevant = {did for did, s in qrel.items() if s > 0}
    hits = relevant & set(retrieved[:k])
    return len(hits) / len(relevant) if relevant else 0.0


def evaluate_retrieval(results: dict[str, list[str]],
                       qrels: dict[str, dict[str, int]],
                       k_values: list[int] = [5, 10]) -> dict[str, float]:
    """Compute averaged metrics across all queries."""
    metrics = {}
    for k in k_values:
        ndcgs, recalls = [], []
        for qid, retrieved in results.items():
            if qid not in qrels:
                continue
            ndcgs.append(ndcg_at_k(retrieved, qrels[qid], k))
            recalls.append(recall_at_k(retrieved, qrels[qid], k))
        metrics[f"nDCG@{k}"] = np.mean(ndcgs)
        metrics[f"Recall@{k}"] = np.mean(recalls)
    return metrics


def print_metrics(name: str, metrics: dict[str, float]):
    """Pretty-print evaluation results."""
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    for k, v in metrics.items():
        print(f"  {k:>12s}: {v:.4f}")
    print()


def save_metrics(name: str, metrics: dict[str, float], path: Path):
    """Save evaluation results to a JSON file."""
    path.mkdir(parents=True, exist_ok=True)
    with open(path / f"{name}.json", "w") as f:
        json.dump(metrics, f, indent=2)
