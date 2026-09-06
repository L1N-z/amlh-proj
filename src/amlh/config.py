"""Paths and frozen hyperparameters for the pipeline.

Hyperparameter values are ``None`` until tuned on validation in Arms 1-3, then
frozen here for the final test run.
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

SEED = 44

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_CSV = DATA_DIR / "patient_qa_classification_train.csv"
TEST_CSV = DATA_DIR / "patient_qa_classification_test.csv"
NHS_DOCS_DIR = DATA_DIR / "db_nhs_qa_classification"
ARTEFACTS_DIR = PROJECT_ROOT / "artefacts"
FIGURES_DIR = PROJECT_ROOT / "figures"

VAL_SIZE = 200  # matches the test set size
VAL_N_CLASSES = 102  # matches the test set class count

# Index-variant options for the Arm 1 class-evidence blob.
IndexVariant = Literal["Q", "QL", "QLA", "QLAD"]


@dataclass(frozen=True)
class Hyperparameters:
    # Arm 1 - TF-IDF / k-NN. Vectoriser, index variant, indexing scheme and k are
    # fixed on validation with a within-1-SE, prefer-simplest rule; where the
    # standard hold-out does not separate the index variants a shift-aware
    # hold-out breaks the tie. ngram_range is re-tuned for the selected variant,
    # whose blobs are reference-document prose rather than short questions.
    ngram_range: tuple[int, int] | None = (1, 2)
    min_df: int | None = 1
    max_df: float | None = 1.0
    sublinear_tf: bool | None = False
    stop_words: str | list[str] | None = None  # ablation outcome
    lemmatise: bool | None = False  # ablation outcome (spaCy en_core_web_sm)
    index_variant: IndexVariant | None = "QLAD"  # question / label / answer / NHS-document blob
    index_scheme: Literal["class_blob", "additive_per_row"] | None = "class_blob"
    k_neighbors: int | None = 1
    # Arm 2 - fine-tuned BERT with a 906-way classification head. The two encoders
    # are trained with identical settings and compared per item on the standard
    # hold-out with McNemar's exact test. max_length follows the question-length
    # percentiles and is not tuned; learning_rate and batch_size are standard
    # fine-tuning defaults held fixed across both encoders; num_epochs is the
    # earliest epoch within 1 SE of the peak validation accuracy in a 24-epoch sweep.
    bert_model_name: str | None = "emilyalsentzer/Bio_ClinicalBERT"
    max_length: int | None = 48
    learning_rate: float | None = 2e-5
    batch_size: int | None = 16
    num_epochs: int | None = 15
    # Arm 3 - LLM selection over the frozen Arm 1 shortlist. Prompt condition then
    # generator are chosen on validation, each with McNemar's exact test: the
    # condition on the primary generator alone, then the generator at that
    # condition. arm3_max_new_tokens is a name-length budget; arm3_cot_max_new_tokens
    # is wider so the chain-of-thought condition has room for reasoning and an answer.
    shortlist_k: int | None = 20
    llm_temperature: float | None = 0.0
    prompt_mode: str | None = "zero_shot"  # zero_shot / few_shot / cot
    n_shots: int | None = 2
    arm3_model_name: str | None = "microsoft/MediPhi-Guidelines"
    arm3_secondary_model_name: str | None = "google/flan-t5-large"
    arm3_max_new_tokens: int | None = 20
    arm3_cot_max_new_tokens: int | None = 128


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
