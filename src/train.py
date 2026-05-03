"""BGE fine-tuning with optional two-stage hard negative mining."""

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

from config import *


def create_evaluator(queries, corpus, qrels, name="dev"):
    relevant_docs = {
        qid: {did for did, s in rels.items() if s > 0}
        for qid, rels in qrels.items()
    }
    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        ndcg_at_k=[10],
        precision_recall_at_k=[5, 10],
        show_progress_bar=True,
        query_prompt=QUERY_PREFIX,
        batch_size=256,
        corpus_chunk_size=50000,
    )


def fine_tune(model: SentenceTransformer,
              train_pairs: list[InputExample],
              dev_evaluator=None,
              epochs=None, lr=None, output_path=None,
              batch_size=None) -> SentenceTransformer:
    epochs = epochs or EPOCHS
    lr = lr or LR
    output_path = output_path or str(MODEL_DIR)
    batch_size = batch_size or BATCH_SIZE

    model.max_seq_length = MAX_SEQ_LEN

    train_dataloader = DataLoader(train_pairs, shuffle=True, batch_size=batch_size)
    loss = losses.MultipleNegativesRankingLoss(model, scale=SCALE)

    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)

    print(f"Training: {epochs} epochs, {len(train_pairs)} samples, batch_size={batch_size}, lr={lr}")
    print(f"Total steps: {total_steps}, warmup: {warmup_steps}, scale={SCALE}")

    model.fit(
        train_objectives=[(train_dataloader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        weight_decay=WEIGHT_DECAY,
        evaluator=dev_evaluator,
        evaluation_steps=len(train_dataloader),
        output_path=output_path,
        save_best_model=True,
    )

    print(f"Best model saved to {output_path}")
    return model


def mine_hard_negatives(model: SentenceTransformer,
                        queries: dict[str, str],
                        corpus: dict[str, str],
                        qrels: dict[str, dict[str, int]],
                        top_k: int = 50,
                        n_negatives: int = 3) -> list[InputExample]:
    from src.retrieval import encode_corpus, build_index, search

    embeddings, doc_ids = encode_corpus(model, corpus, cache_name="hn_mining")
    index = build_index(embeddings)
    results = search(model, index, doc_ids, queries, top_k=top_k)

    triplets = []
    for qid, retrieved in results.items():
        if qid not in qrels:
            continue
        positives = {did for did, s in qrels[qid].items() if s > 0}
        negatives = [did for did in retrieved if did not in positives][:n_negatives]

        for pos_id in positives:
            if pos_id not in corpus:
                continue
            for neg_id in negatives:
                triplets.append(InputExample(
                    texts=[QUERY_PREFIX + queries[qid], corpus[pos_id], corpus[neg_id]]
                ))

    print(f"Mined {len(triplets)} hard negative triplets")
    return triplets
