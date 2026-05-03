"""Step 1: encode corpus embeddings and cache to disk."""

from sentence_transformers import SentenceTransformer
from src.data_loader import load_corpus, download_fiqa
from src.retrieval import encode_corpus
from config import FIQA_DIR, MODEL_NAME, DEVICE

download_fiqa()
corpus = load_corpus(FIQA_DIR)
print(f"Corpus: {len(corpus)} docs")

model = SentenceTransformer(MODEL_NAME, device=DEVICE)
print(f"Model device: {model.device}")

embeddings, doc_ids = encode_corpus(model, corpus, cache_name="bge_zeroshot")
print(f"Embedding shape: {embeddings.shape}")
print("Done.")
