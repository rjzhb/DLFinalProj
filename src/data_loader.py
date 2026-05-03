"""FiQA-2018 dataset download and loading."""

import json
import csv
import zipfile
import urllib.request
from pathlib import Path

from sentence_transformers import InputExample
from config import FIQA_URL, FIQA_DIR, DATA_DIR, QUERY_PREFIX


def download_fiqa():
    """Download and extract the FiQA dataset, returning the data directory."""
    if FIQA_DIR.exists():
        print("FiQA dataset already exists, skipping download")
        return FIQA_DIR

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "fiqa.zip"

    print("Downloading FiQA dataset...")
    urllib.request.urlretrieve(FIQA_URL, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)
    zip_path.unlink()

    print(f"Dataset saved to {FIQA_DIR}")
    return FIQA_DIR


def load_corpus(data_path: Path) -> dict[str, str]:
    """Load corpus, returns {doc_id: text}."""
    corpus = {}
    with open(data_path / "corpus.jsonl") as f:
        for line in f:
            doc = json.loads(line)
            title = doc.get("title", "").strip()
            text = doc["text"].strip()
            corpus[doc["_id"]] = f"{title} {text}".strip()
    return corpus


def load_queries(data_path: Path) -> dict[str, str]:
    """Load queries, returns {query_id: query_text}."""
    queries = {}
    with open(data_path / "queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    return queries


def load_qrels(data_path: Path, split: str) -> dict[str, dict[str, int]]:
    """Load relevance labels, returns {query_id: {doc_id: score}}."""
    qrels = {}
    with open(data_path / "qrels" / f"{split}.tsv") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid, did = row["query-id"], row["corpus-id"]
            qrels.setdefault(qid, {})[did] = int(row["score"])
    return qrels


def build_training_pairs(queries, corpus, qrels) -> list[InputExample]:
    """Build (query, positive_passage) training pairs with the BGE query prefix."""
    pairs = []
    for qid, rels in qrels.items():
        query = QUERY_PREFIX + queries[qid]
        for doc_id, score in rels.items():
            if score > 0 and doc_id in corpus:
                pairs.append(InputExample(texts=[query, corpus[doc_id]]))
    print(f"Built {len(pairs)} training pairs")
    return pairs
