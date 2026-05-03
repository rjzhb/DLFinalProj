"""FAISS vector index construction and retrieval (with embedding cache)."""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import OUTPUT_DIR, DEVICE, QUERY_PREFIX

CACHE_DIR = OUTPUT_DIR / "embeddings"


def encode_corpus(model: SentenceTransformer, corpus: dict[str, str],
                  cache_name: str = "corpus",
                  batch_size: int = 256,
                  chunk_size: int = 5000) -> tuple[np.ndarray, list[str]]:
    """Encode the corpus in chunks with disk persistence and resume support."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    emb_path = CACHE_DIR / f"{cache_name}.npy"
    ids_path = CACHE_DIR / f"{cache_name}_ids.json"

    if emb_path.exists() and ids_path.exists():
        print(f"Loading embeddings from cache: {emb_path}")
        embeddings = np.load(emb_path)
        doc_ids = json.loads(ids_path.read_text())
        return embeddings, doc_ids

    doc_ids = list(corpus.keys())
    texts = [corpus[did] for did in doc_ids]
    n_chunks = (len(texts) + chunk_size - 1) // chunk_size

    existing_chunks = sorted(CACHE_DIR.glob(f"{cache_name}_chunk_*.npy"))
    start_chunk = len(existing_chunks)
    if start_chunk > 0:
        print(f"Found {start_chunk} completed chunks, resuming from chunk {start_chunk}")

    for i in range(start_chunk, n_chunks):
        lo, hi = i * chunk_size, min((i + 1) * chunk_size, len(texts))
        print(f"Encoding chunk {i+1}/{n_chunks} ({lo}-{hi}, {hi-lo} docs)...")
        chunk_emb = model.encode(texts[lo:hi], batch_size=batch_size,
                                 show_progress_bar=True, normalize_embeddings=True,
                                 device=DEVICE).astype(np.float32)
        np.save(CACHE_DIR / f"{cache_name}_chunk_{i:04d}.npy", chunk_emb)
        print(f"  chunk {i+1} saved")

    chunks = [np.load(p) for p in sorted(CACHE_DIR.glob(f"{cache_name}_chunk_*.npy"))]
    embeddings = np.vstack(chunks)
    np.save(emb_path, embeddings)
    ids_path.write_text(json.dumps(doc_ids))

    for p in CACHE_DIR.glob(f"{cache_name}_chunk_*.npy"):
        p.unlink()
    print(f"Encoding complete, cache saved to {emb_path}")

    return embeddings, doc_ids


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index (equivalent to cosine similarity for normalized vectors)."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def search(model: SentenceTransformer, index: faiss.IndexFlatIP,
           doc_ids: list[str], queries: dict[str, str],
           top_k: int = 10) -> dict[str, list[str]]:
    """Retrieve top-K documents for each query."""
    qids = list(queries.keys())
    q_texts = [queries[qid] for qid in qids]
    print("Encoding queries...")
    q_embeddings = model.encode(q_texts, prompt=QUERY_PREFIX,
                                show_progress_bar=True, normalize_embeddings=True,
                                device=DEVICE).astype(np.float32)

    _, indices = index.search(q_embeddings, top_k)

    results = {}
    for i, qid in enumerate(qids):
        results[qid] = [doc_ids[idx] for idx in indices[i]]
    return results
