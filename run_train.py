"""Fine-tune BGE-base-en-v1.5 on FiQA (single stage, used for the model-scale ablation)."""

from sentence_transformers import SentenceTransformer

from config import FIQA_DIR, MODEL_NAME, DEVICE, EPOCHS, LR
from src.data_loader import load_corpus, load_queries, load_qrels, build_training_pairs
from src.train import create_evaluator, fine_tune

# Load data
corpus = load_corpus(FIQA_DIR)
queries = load_queries(FIQA_DIR)
train_qrels = load_qrels(FIQA_DIR, "train")
dev_qrels = load_qrels(FIQA_DIR, "dev")

train_queries = {qid: queries[qid] for qid in train_qrels if qid in queries}
dev_queries = {qid: queries[qid] for qid in dev_qrels if qid in queries}

print(f"Corpus: {len(corpus)} docs")
print(f"Train queries: {len(train_queries)}, dev queries: {len(dev_queries)}")

train_pairs = build_training_pairs(train_queries, corpus, train_qrels)
dev_evaluator = create_evaluator(dev_queries, corpus, dev_qrels)

# Single-stage fine-tuning
print(f"\n===== Fine-tuning {MODEL_NAME} =====")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)
model = fine_tune(model, train_pairs, dev_evaluator, epochs=EPOCHS, lr=LR)

print("Training complete.")
