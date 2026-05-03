"""Embedding-space visualizations - academic paper style."""

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE
from config import FIGURES_DIR

# Global academic style
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

# Color palette
COLORS = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "red": "#d62728",
    "green": "#2ca02c",
    "purple": "#9467bd",
    "gray": "#7f7f7f",
}
MARKERS = ["o", "s", "^", "D", "v", "P"]


def plot_training_curves(csv_path: str, filename: str = "training_curves.png"):
    """Plot training curves (two subplots) from the evaluator's CSV log."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    epochs = [int(float(r["epoch"])) for r in rows]
    ndcg10 = [float(r["cosine-NDCG@10"]) for r in rows]
    recall5 = [float(r["cosine-Recall@5"]) for r in rows]
    recall10 = [float(r["cosine-Recall@10"]) for r in rows]
    mrr10 = [float(r["cosine-MRR@10"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # (a) Ranking Metrics
    axes[0].plot(epochs, ndcg10, marker="o", color=COLORS["red"], label="nDCG@10")
    axes[0].plot(epochs, mrr10, marker="s", color=COLORS["blue"], label="MRR@10")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Score")
    axes[0].set_title("(a) Ranking Metrics")
    axes[0].legend(framealpha=0.9, edgecolor="gray")
    axes[0].set_xticks(epochs)

    # (b) Recall Metrics
    axes[1].plot(epochs, recall5, marker="^", color=COLORS["green"], label="Recall@5")
    axes[1].plot(epochs, recall10, marker="D", color=COLORS["orange"], label="Recall@10")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("(b) Recall Metrics")
    axes[1].legend(framealpha=0.9, edgecolor="gray")
    axes[1].set_xticks(epochs)

    fig.tight_layout(w_pad=3)
    path = FIGURES_DIR / filename
    fig.savefig(path)
    plt.close()
    print(f"Training curves saved to {path}")


def plot_comparison(baseline_metrics: dict, finetuned_metrics: dict,
                    bm25_metrics: dict = None,
                    filename: str = "comparison.png"):
    """Bar chart comparing retrieval metrics across systems."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    labels = list(baseline_metrics.keys())
    base_vals = [baseline_metrics[k] for k in labels]
    ft_vals = [finetuned_metrics[k] for k in labels]

    x = np.arange(len(labels))

    has_bm25 = bm25_metrics is not None
    n_bars = 3 if has_bm25 else 2
    width = 0.7 / n_bars

    fig, ax = plt.subplots(figsize=(8, 5))

    if has_bm25:
        bm25_vals = [bm25_metrics[k] for k in labels]
        bars_bm25 = ax.bar(x - width, bm25_vals, width, label="BM25",
                           color=COLORS["gray"], edgecolor="white", linewidth=0.5)
        bars_base = ax.bar(x, base_vals, width, label="BGE Zero-shot",
                           color=COLORS["blue"], edgecolor="white", linewidth=0.5)
        bars_ft = ax.bar(x + width, ft_vals, width, label="BGE Fine-tuned",
                         color=COLORS["red"], edgecolor="white", linewidth=0.5)
        all_bars = [bars_bm25, bars_base, bars_ft]
        all_vals = [bm25_vals, base_vals, ft_vals]
    else:
        bars_base = ax.bar(x - width / 2, base_vals, width, label="BGE Zero-shot",
                           color=COLORS["blue"], edgecolor="white", linewidth=0.5)
        bars_ft = ax.bar(x + width / 2, ft_vals, width, label="BGE Fine-tuned",
                         color=COLORS["red"], edgecolor="white", linewidth=0.5)
        all_bars = [bars_base, bars_ft]
        all_vals = [base_vals, ft_vals]

    # Numeric labels
    for bars, vals in zip(all_bars, all_vals):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_title("Retrieval Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(framealpha=0.9, edgecolor="gray", loc="upper left")
    ax.set_ylim(0, max(max(ft_vals), max(base_vals)) * 1.2)
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path)
    plt.close()
    print(f"Comparison chart saved to {path}")


def plot_embeddings(query_embs: np.ndarray, doc_embs: np.ndarray,
                    title: str = "Embedding Space", filename: str = "tsne.png"):
    """t-SNE visualization of query and document embedding distributions."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    n_queries = min(200, len(query_embs))
    n_docs = min(800, len(doc_embs))
    q_sample = query_embs[np.random.choice(len(query_embs), n_queries, replace=False)]
    d_sample = doc_embs[np.random.choice(len(doc_embs), n_docs, replace=False)]

    all_embs = np.vstack([q_sample, d_sample])
    labels = ["Query"] * n_queries + ["Document"] * n_docs

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    coords = tsne.fit_transform(all_embs)

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, color, marker in [("Document", COLORS["blue"], "o"),
                                  ("Query", COLORS["red"], "^")]:
        mask = [l == label for l in labels]
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, label=label, alpha=0.5, s=12,
                   marker=marker, edgecolors="none")

    ax.set_title(title)
    ax.legend(framealpha=0.9, edgecolor="gray", markerscale=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = FIGURES_DIR / filename
    fig.savefig(path)
    plt.close()
    print(f"Visualization saved to {path}")


def plot_improvement_heatmap(bm25_metrics: dict, zs_metrics: dict, ft_metrics: dict,
                              filename: str = "improvement_table.png"):
    """Render a summary table of metrics across systems."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics = list(bm25_metrics.keys())
    models = ["BM25", "BGE Zero-shot", "BGE Fine-tuned"]
    data = [
        [bm25_metrics[k] for k in metrics],
        [zs_metrics[k] for k in metrics],
        [ft_metrics[k] for k in metrics],
    ]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")

    table = ax.table(
        cellText=[[f"{v:.4f}" for v in row] for row in data],
        rowLabels=models,
        colLabels=metrics,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Styling
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == -1:
            cell.set_facecolor("#ecf0f1")
            cell.set_text_props(fontweight="bold")
        elif row == 3:  # Fine-tuned row (best)
            cell.set_facecolor("#e8f8f5")
        else:
            cell.set_facecolor("white")

    ax.set_title("Retrieval Performance Summary", fontsize=13, pad=20)

    path = FIGURES_DIR / filename
    fig.savefig(path)
    plt.close()
    print(f"Summary table saved to {path}")
