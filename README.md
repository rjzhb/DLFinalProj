# DLFinalProj: Domain-Adaptive Embedding Fine-Tuning for Financial Retrieval

CS 7643 Final Project. Fine-tunes BGE embedding models on FiQA-2018 for financial retrieval in RAG systems.

## Authors

- Jiaqi Deng
- Hongbin Zhong

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

```bash
# 1. Encode corpus and cache embeddings
python run_embedding.py

# 2. Run BM25 + zero-shot baselines
python run_baseline.py

# 3. Fine-tune BGE encoder
python run_train.py

# 4. Evaluate fine-tuned model on test set
python run_eval.py
```

## Configuration

All hyperparameters in `config.py`. Switch model with `MODEL_NAME` (e.g., `BAAI/bge-small-en` vs `BAAI/bge-base-en-v1.5`).

## Report

Final report: `report/iclr2025_conference.pdf`

## Key Results (FiQA test set)

| Model | nDCG@10 | Recall@10 |
|-------|---------|-----------|
| BM25 | 0.1591 | 0.2037 |
| BGE-small-en zero-shot | 0.3109 | 0.3653 |
| BGE-small-en fine-tuned | 0.3916 | 0.4572 |
| BGE-base-en-v1.5 zero-shot | 0.4062 | 0.4814 |
| BGE-base-en-v1.5 fine-tuned | 0.4053 | 0.4779 |
