"""Single source of truth for paths and frozen hyperparameters.

Every choice that determines the frozen test run lives here — nothing left in
notebook cells. Hyperparameter values are ``None`` until tuned in Arms 1-3,
then frozen before ``05_results``.
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_CSV = DATA_DIR / "patient_qa_classification_train.csv"
TEST_CSV = DATA_DIR / "patient_qa_classification_test.csv"
NHS_DOCS_DIR = DATA_DIR / "db_nhs_qa_classification"
ARTEFACTS_DIR = PROJECT_ROOT / "artefacts"
FIGURES_DIR = PROJECT_ROOT / "figures"

VAL_SIZE = 200  # mirrors len(test)
VAL_N_CLASSES = 102  # mirrors test class count

# Literal type alias for the frozen Arm 1 index-variant choice (populated after ablation)
IndexVariant = Literal["Q", "QL", "QLA", "QLAD"]


@dataclass(frozen=True)
class Hyperparameters:
    # Arm 1 — TF-IDF / k-NN (vectoriser + index + retrieval)
    # Frozen from notebooks/02_arm1.ipynb (validation-only, stratified 200/102-class hold-out).
    # Selection used a within-1-SE-prefer-simplest rule throughout, not raw argmax, given
    # SE ≈ 3.5pp at n=200. Q, class_blob, k=1, unigrams, no stop words, no lemmatisation was
    # simplest within 1 SE of the grid argmax (ngram_range=(1,2), stop_words="english",
    # accuracy=0.835); the variant-specific vectoriser re-check confirmed the same config.
    ngram_range: tuple[int, int] | None = (1, 1)
    min_df: int | None = 1
    max_df: float | None = 1.0
    sublinear_tf: bool | None = False
    stop_words: str | list[str] | None = None  # ablation outcome, e.g. None or "english"
    lemmatise: bool | None = False  # ablation outcome (spaCy en_core_web_sm)
    index_variant: IndexVariant | None = "Q"  # Q / QL / QLA / QLAD — biggest Arm 1 lever
    index_scheme: Literal["class_blob", "additive_per_row"] | None = "class_blob"
    k_neighbors: int | None = 1
    # Arm 2 — BERT
    bert_model_name: str | None = None
    max_length: int | None = None
    learning_rate: float | None = None
    batch_size: int | None = None
    num_epochs: int | None = None
    # Arm 3 — LLM shortlist + prompting
    shortlist_k: int | None = None
    llm_temperature: float | None = None
    prompt_mode: str | None = None  # e.g. zero_shot / few_shot / cot
    n_shots: int | None = None


HYPERPARAMETERS = Hyperparameters()


def set_seed(seed: int = SEED) -> None:
    """Seed every RNG this project touches. Call before every stochastic stage."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
