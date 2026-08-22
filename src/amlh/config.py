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

SEED = 44

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
    # Frozen from notebooks/02_arm1.ipynb (validation only; no test-set quantity was read).
    # Selection used a within-1-SE-prefer-simplest rule throughout, not raw argmax, given
    # SE ~ 3.5pp at n=200.
    #
    # index_variant: the standard hold-out ties all four variants at 0.820 (SE 0.027) and so
    # cannot choose. Under CLAUDE.md's pre-registered tie-break the shift-aware hold-out decides,
    # and it ranks QLAD first at 0.328 by 0.065 over the runner-up against its own SE of 0.023.
    # index_scheme: both protocols rank class_blob first, each by more than its own SE
    # (standard 0.850 by 0.120, SE 0.025; shift-aware 0.398 by 0.073, SE 0.024). They agree, so
    # the tie-break never fires.
    # ngram_range: (1,2) comes from the variant-specific re-check (notebook §4b) — QLAD blobs are
    # NHS prose rather than short questions, and the step-2 optimum tuned on variant Q did not
    # transfer. That re-check reached 0.850 on the standard hold-out at k=1.
    ngram_range: tuple[int, int] | None = (1, 2)
    min_df: int | None = 1
    max_df: float | None = 1.0
    sublinear_tf: bool | None = False
    stop_words: str | list[str] | None = None  # ablation outcome, e.g. None or "english"
    lemmatise: bool | None = False  # ablation outcome (spaCy en_core_web_sm)
    index_variant: IndexVariant | None = "QLAD"  # Q / QL / QLA / QLAD — biggest Arm 1 lever
    index_scheme: Literal["class_blob", "additive_per_row"] | None = "class_blob"
    k_neighbors: int | None = 1
    # Arm 2 — BERT
    # Frozen from notebooks/03_arm2_bert_colab.ipynb, run on a Colab T4 with
    # QUICK_SMOKE_TEST=False (validation only; no test-set quantity was read).
    #
    # bert_model_name: the encoder comparison is UNRESOLVED, not won. Both encoders were trained
    # with identical hyperparameters, seed and epoch budget, then compared per-item on the standard
    # hold-out with McNemar's exact test (artefacts/arm2_encoder_mcnemar.csv): 6 items only
    # Bio_ClinicalBERT got right, 11 only bert-base-uncased got right, 17 discordant, p = 0.3323.
    # p >= 0.05, so CLAUDE.md's Arm 2 tie-break fires and the in-domain clinical encoder is kept on
    # the declared prior. This RETAINS THE LOWER-SCORING ENCODER — Bio_ClinicalBERT 0.850 vs
    # bert-base-uncased 0.875 at their respective selected epochs. That is the intended behaviour
    # of a prior, not an accuracy judgement, and the report must say so. Unlike the Arm 1
    # tie-break, this rule was formalised AFTER the comparison was seen; disclose that timing.
    # max_length: 48, from the 01_eda question-length percentiles (mean 8.4 words, 99th pct 18),
    # not tuned here. Truncation rate at 48 is printed in the notebook before training.
    # learning_rate / batch_size: 2e-5 / 16, standard BERT fine-tuning defaults held FIXED across
    # both encoders so the ablation isolates the checkpoint. Neither was swept — with SE ~ 3.5pp at
    # n=200 a 200-item hold-out cannot resolve a learning-rate grid, so sweeping would have
    # manufactured a selection the data does not support. Report this as a declared default.
    # num_epochs: 15 is the SELECTED CHECKPOINT, not the training budget — the ablation swept 24
    # epochs and this is the epoch chosen from that sweep (0-indexed epoch 14, +1). Selection used
    # select_best_epoch_within_one_se on the standard hold-out: peak val_accuracy 0.870 gives
    # SE 0.0238 and a threshold of 0.8462, which 7 of the 24 epochs meet ([14, 17, 19, 20, 21, 22,
    # 23]); the rule takes the earliest, buying the peak's accuracy for 8 fewer epochs of training.
    # Caveat for the report: val_loss is still falling across that band (1.034 at epoch 14 -> 0.807
    # at 22), so epoch 22 is not a degraded model — the rule reads accuracy only.
    bert_model_name: str | None = "emilyalsentzer/Bio_ClinicalBERT"
    max_length: int | None = 48
    learning_rate: float | None = 2e-5
    batch_size: int | None = 16
    num_epochs: int | None = 15
    # Arm 3 — LLM shortlist + prompting
    # Frozen from notebooks/04_arm3_llm.ipynb after the prompt-budget check and condition
    # comparison. The selection rule is pre-registered in CLAUDE.md, and the notebook must print
    # the chosen values before they are copied here.
    shortlist_k: int | None = 20
    llm_temperature: float | None = 0.0
    prompt_mode: str | None = None  # zero_shot / few_shot / cot — frozen after McNemar selection
    n_shots: int | None = None
    arm3_model_name: str | None = "google/flan-t5-large"


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
