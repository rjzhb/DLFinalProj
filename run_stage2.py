"""Stage 2 only: load best Stage 1 checkpoint, mine hard negatives, continue training."""

from sentence_transformers import SentenceTransformer

from config import FIQA_DIR, MODEL_DIR, DEVICE
from src.data_loader import load_corpus, load_queries, load_qrels
from src.train import create_evaluator, fine_tune, mine_hard_negatives

# Stage 2 specific hyperparameters
EPOCHS_STAGE2 = 2
LR_STAGE2 = 5e-6

corpus = load_corpus(FIQA_DIR)
queries = load_queries(FIQA_DIR)
train_qrels = load_qrels(FIQA_DIR, "train")
dev_qrels = load_qrels(FIQA_DIR, "dev")

train_queries = {qid: queries[qid] for qid in train_qrels if qid in queries}
dev_queries = {qid: queries[qid] for qid in dev_qrels if qid in queries}

dev_evaluator = create_evaluator(dev_queries, corpus, dev_qrels)

print("===== Stage 2: hard negative mining =====")
best_model = SentenceTransformer(str(MODEL_DIR), device=DEVICE)

print("Mining hard negatives...")
hn_triplets = mine_hard_negatives(best_model, train_queries, corpus, train_qrels,
                                   top_k=50, n_negatives=3)

print(f"\nStarting Stage 2 training: {EPOCHS_STAGE2} epochs, LR={LR_STAGE2}, batch_size=32")
best_model = fine_tune(best_model, hn_triplets, dev_evaluator,
                        epochs=EPOCHS_STAGE2, lr=LR_STAGE2, batch_size=32)

print("\nStage 2 training complete.")
