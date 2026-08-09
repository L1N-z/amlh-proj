"""Load, audit, and split the patient question classification dataset.

No modelling code lives here. ``load_test`` strips the ``answer`` column at
load time — the primary enforcement point for the rule that ``answer`` is
never an inference-time input (see CLAUDE.md hard rule #1).
"""

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from amlh.config import ARTEFACTS_DIR, SEED, TEST_CSV, TRAIN_CSV, VAL_N_CLASSES, VAL_SIZE

EXPECTED_COLUMNS = {"question", "answer", "disease", "reference_url"}
MIN_FIT_SUPPORT = 2  # examples each class must retain for fitting


@dataclass
class SplitResult:
    fit: pd.DataFrame
    val: pd.DataFrame
    val_classes: list[str]


def load_train() -> pd.DataFrame:
    """Full training DataFrame, including `answer` (permitted as class evidence)."""
    df = pd.read_csv(TRAIN_CSV)
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"train missing columns: {missing}")
    return df.dropna(subset=["question", "disease"]).reset_index(drop=True)


def load_test() -> pd.DataFrame:
    """Test DataFrame with `answer` dropped — enforces hard rule #1."""
    df = pd.read_csv(TEST_CSV)
    missing = {"question", "disease", "reference_url"} - set(df.columns)
    if missing:
        raise ValueError(f"test missing columns: {missing}")
    df = df.dropna(subset=["question", "disease"]).reset_index(drop=True)
    return df.drop(columns=[c for c in df.columns if c == "answer"])


def run_integrity_audit(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Compute the integrity audit dict. Never reads `test.answer` (column is absent)."""
    cnt = train.disease.value_counts()
    audit = {
        "n_train": len(train),
        "n_test": len(test),
        "n_classes_train": int(train.disease.nunique()),
        "n_classes_test": int(test.disease.nunique()),
        "test_labels_unseen_in_train": sorted(set(test.disease) - set(train.disease)),
        "class_support": {
            "min": int(cnt.min()),
            "median": float(cnt.median()),
            "mean": round(float(cnt.mean()), 2),
            "max": int(cnt.max()),
            "sd": round(float(cnt.std()), 2),
            "n_singletons": int((cnt == 1).sum()),
            "n_below_5": int((cnt < 5).sum()),
        },
        "exact_dup_question_disease": int(train.duplicated(["question", "disease"]).sum()),
        "duplicate_question_strings": int(train.duplicated(["question"]).sum()),
    }

    # Ambiguity: the same question string mapped to more than one disease.
    g = train[train.duplicated("question", keep=False)].groupby("question").disease.nunique()
    audit["ambiguous_question_strings"] = int((g > 1).sum())

    # Label granularity: shared-prefix families.
    fam = pd.Series(sorted(train.disease.unique())).str.split("_").str[0]
    fam_counts = fam.value_counts()
    audit["labels_in_a_family"] = int((fam.map(fam_counts) > 1).sum())
    audit["largest_families"] = fam_counts.head(5).to_dict()

    # Train-test near-duplication. Vectoriser fit on train questions only, transform test.
    dup_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(train.question)
    sim = cosine_similarity(dup_vec.transform(test.question), dup_vec.transform(train.question))
    max_sim = sim.max(axis=1)
    audit["near_duplication"] = {
        f"frac_cos_ge_{t}": round(float((max_sim >= t).mean()), 4) for t in (0.95, 0.90, 0.80, 0.70)
    }
    audit["near_duplication"]["median_max_cosine"] = round(float(np.median(max_sim)), 4)

    return audit


def make_validation_split(train: pd.DataFrame, seed: int = SEED) -> SplitResult:
    """Class-aware stratified hold-out matching the test set's size and class count.

    Allocates the VAL_SIZE quota across VAL_N_CLASSES up front (via divmod), so no
    truncation is needed and every selected class is guaranteed at least one item.
    """
    rng = np.random.default_rng(seed)
    cnt = train.disease.value_counts()

    base, extra = divmod(VAL_SIZE, VAL_N_CLASSES)

    eligible = cnt[cnt >= base + MIN_FIT_SUPPORT].index.to_numpy()
    if len(eligible) < VAL_N_CLASSES:
        raise ValueError(
            f"only {len(eligible)} classes can supply {base} item(s) "
            f"while retaining {MIN_FIT_SUPPORT}"
        )
    val_classes = rng.choice(eligible, size=VAL_N_CLASSES, replace=False)

    quota = {c: base for c in val_classes}
    bonus_pool = [c for c in val_classes if cnt[c] >= base + 1 + MIN_FIT_SUPPORT]
    if len(bonus_pool) < extra:
        raise ValueError(
            f"only {len(bonus_pool)} classes can supply an extra item; {extra} needed"
        )
    for c in rng.choice(bonus_pool, size=extra, replace=False):
        quota[c] += 1

    val_idx = []
    for c in val_classes:
        pool = train.index[train.disease == c].to_numpy()
        val_idx += list(rng.choice(pool, size=quota[c], replace=False))
    val_idx = list(rng.permutation(val_idx))

    fit = train.drop(index=val_idx)
    val = train.loc[val_idx]

    assert len(val) == VAL_SIZE
    assert val.disease.nunique() == VAL_N_CLASSES
    assert set(val.disease) <= set(fit.disease)
    assert fit.disease.value_counts().reindex(train.disease.unique()).min() >= MIN_FIT_SUPPORT
    assert set(val.index) & set(fit.index) == set()

    return SplitResult(fit=fit, val=val, val_classes=sorted(val_classes))


def save_splits(fit: pd.DataFrame, val: pd.DataFrame, audit: dict) -> None:
    """Write splits + audit to artefacts/. split_fit.csv retains `answer` (needed for
    Arm 1 index variants QLA/QLAD) — do not strip it on save."""
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fit.to_csv(ARTEFACTS_DIR / "split_fit.csv", index=False)
    val.to_csv(ARTEFACTS_DIR / "split_val.csv", index=False)
    with open(ARTEFACTS_DIR / "integrity_audit.json", "w") as f:
        json.dump(audit, f, indent=2)


if __name__ == "__main__":
    train = load_train()
    test = load_test()

    audit = run_integrity_audit(train, test)
    split = make_validation_split(train)
    save_splits(split.fit, split.val, audit)

    print(json.dumps(audit, indent=2))
    print(
        f"\nfit={len(split.fit)} | val={len(split.val)} ({split.val.disease.nunique()} classes) "
        f"| test={len(test)}"
    )

    # Unstratified split for the ablation table, reported alongside the stratified one.
    _, unstrat_val = train_test_split(train, test_size=VAL_SIZE, random_state=SEED)
    print(f"unstratified val: {len(unstrat_val)} rows, {unstrat_val.disease.nunique()} classes")
