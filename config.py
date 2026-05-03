import platform
from pathlib import Path
import torch

# Paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
MODEL_DIR = OUTPUT_DIR / "models" / "bge-base-en-v1.5-fiqa"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Dataset
FIQA_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
FIQA_DIR = DATA_DIR / "fiqa"

# Model
MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768
MAX_SEQ_LEN = 512
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Training
BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5
WARMUP_RATIO = 0.1
SCALE = 20.0
WEIGHT_DECAY = 0.01

# Evaluation
TOP_K_VALUES = [5, 10]

# Device
if platform.system() == "Darwin" and torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
