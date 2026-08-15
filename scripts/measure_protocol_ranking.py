"""One-off diagnostic: does the standard class-aware hold-out or the hard
(shift-aware) hold-out better predict Arm 1 index-variant/scheme ranking on
the real test set?

CLAUDE.md declared exception #2. Test-set access is confined to this script.
This is a protocol diagnostic, not model selection: it measures which
validation protocol correlates better with test ranking and reports the
result for review. It does not modify config.py's frozen index_variant /
index_scheme, and it does not persist test_acc (or anything derived from it)
to artefacts/ — nothing test-tainted should end up somewhere a notebook could
later load it.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from amlh import arm1_experiments as ae
from amlh import arm1_tfidf, features
from amlh.config import HYPERPARAMETERS, SEED, set_seed
from amlh.data import load_test, load_train, make_hard_validation_split, make_validation_split
from amlh.evaluate import pairwise_top1_disagreement

VARIANTS = ["Q", "QL", "QLA", "QLAD"]


def _vec_kwargs() -> dict:
    return {
        "ngram_range": HYPERPARAMETERS.ngram_range,
        "sublinear_tf": HYPERPARAMETERS.sublinear_tf,
        "min_df": HYPERPARAMETERS.min_df,
        "stop_words": HYPERPARAMETERS.stop_words,
    }


def _class_blob_top1(fit_df: pd.DataFrame, val_questions: list[str], vec_kwargs: dict, k: int) -> dict[str, list]:
    top1_by_variant = {}
    for variant in VARIANTS:
        index_texts, index_labels = features.build_index(fit_df, variant)
        ranked, _ = arm1_tfidf.knn_rank(index_texts, index_labels, val_questions, k, vec_kwargs)
        top1_by_variant[variant] = [r[0] if r else None for r in ranked]
    return top1_by_variant


def std_val_tie_verification(train: pd.DataFrame, test: pd.DataFrame, std_split, vec_kwargs: dict, k: int) -> None:
    """Diagnostic 1: the four class_blob variants tie at std_val_acc=0.820 —
    is that identical predictions or coincidentally equal error counts?"""
    print("=== Diagnostic 1: std-val tie verification (class_blob) ===")

    std_top1 = _class_blob_top1(std_split.fit, std_split.val["question"].tolist(), vec_kwargs, k)
    std_changed = pairwise_top1_disagreement(std_top1)
    print(f"changed-prediction counts, pairwise, out of {len(std_split.val)} std-val items:")
    print(std_changed.to_string())
    print()

    # Reference: same comparison on the test set, index built from the full train
    # (no held-out portion needed — same fit set used for test_acc above).
    test_top1 = _class_blob_top1(train, test["question"].tolist(), vec_kwargs, k)
    test_changed = pairwise_top1_disagreement(test_top1)
    print(f"changed-prediction counts, pairwise, out of {len(test)} test items (reference):")
    print(test_changed.to_string())
    print()

    if (std_changed.to_numpy() == 0).all():
        print("std-val: all four variants produce IDENTICAL top-1 predictions on every item — "
              "the 0.820 tie is a real identity, not a coincidence.")
    else:
        print("std-val: variants disagree on some items (see matrix above) yet still tie at "
              "0.820 accuracy — the tie is a coincidence of equal error COUNTS, not identical "
              "predictions.")
    print()


def idf_distortion_diagnostic(train: pd.DataFrame, vec_kwargs: dict) -> None:
    """Diagnostic 2: does additive_per_row's larger corpus distort IDF for
    disease-specific vs generic terms at variant QLAD?

    NOTE on the premise: `features.build_index_additive` appends each class's
    L/D text exactly ONCE per class, not once per training row (fixed in an
    earlier session specifically to avoid ~10x-per-row duplication of the NHS
    document). So there is no document-frequency inflation of D-derived terms
    from row-duplication under additive_per_row. Whatever IDF shift is
    measured below comes from additive_per_row's larger document count (n) in
    the smoothed IDF formula, not from df inflation of disease-specific terms.
    """
    print("=== Diagnostic 2: IDF distortion under additive_per_row + D (variant QLAD) ===")
    print("NOTE: build_index_additive appends L/D once per class in BOTH schemes (fixed "
          "earlier) — there is no ~10x-per-row duplication of the NHS document to measure. "
          "What follows measures the corpus-size-driven IDF shift instead.")
    print()

    index_texts_blob, _ = features.build_index(train, "QLAD")
    index_texts_add, _ = features.build_index_additive(train, "QLAD")

    vec_blob = features.build_vectoriser(**vec_kwargs).fit(index_texts_blob)
    vec_add = features.build_vectoriser(**vec_kwargs).fit(index_texts_add)

    n_docs_blob = len(index_texts_blob)
    n_docs_add = len(index_texts_add)
    print(f"n_documents: class_blob={n_docs_blob}, additive_per_row={n_docs_add}")

    diseases = train["disease"].unique().tolist()
    term_class_count = features.term_class_coverage(diseases)

    common_vocab = set(vec_blob.vocabulary_) & set(vec_add.vocabulary_)
    disease_specific_pool = sorted(t for t, c in term_class_count.items() if c == 1 and t in common_vocab)
    generic_pool = sorted(t for t, c in term_class_count.items() if c > 50 and t in common_vocab)
    print(f"disease-specific term pool (in exactly 1 class doc): {len(disease_specific_pool)}")
    print(f"generic term pool (in >50 class docs): {len(generic_pool)}")

    rng = np.random.default_rng(SEED)
    ds_sample = rng.choice(disease_specific_pool, size=min(200, len(disease_specific_pool)), replace=False)
    gen_sample = rng.choice(generic_pool, size=min(200, len(generic_pool)), replace=False)
    print(f"sampled: {len(ds_sample)} disease-specific terms, {len(gen_sample)} generic terms")
    print()

    def _idf_stats(vec, terms):
        vals = [vec.idf_[vec.vocabulary_[t]] for t in terms]
        return float(np.mean(vals)), float(np.median(vals))

    blob_ds_mean, blob_ds_median = _idf_stats(vec_blob, ds_sample)
    add_ds_mean, add_ds_median = _idf_stats(vec_add, ds_sample)
    blob_gen_mean, blob_gen_median = _idf_stats(vec_blob, gen_sample)
    add_gen_mean, add_gen_median = _idf_stats(vec_add, gen_sample)

    idf_table = pd.DataFrame(
        [
            {"scheme": "class_blob", "term_group": "disease_specific", "mean_idf": blob_ds_mean, "median_idf": blob_ds_median},
            {"scheme": "additive_per_row", "term_group": "disease_specific", "mean_idf": add_ds_mean, "median_idf": add_ds_median},
            {"scheme": "class_blob", "term_group": "generic", "mean_idf": blob_gen_mean, "median_idf": blob_gen_median},
            {"scheme": "additive_per_row", "term_group": "generic", "mean_idf": add_gen_mean, "median_idf": add_gen_median},
        ]
    )
    print(idf_table.to_string(index=False))
    print()

    ratio_blob = blob_ds_mean / blob_gen_mean
    ratio_add = add_ds_mean / add_gen_mean
    cross_scheme_ratio_ds = add_ds_mean / blob_ds_mean
    cross_scheme_ratio_gen = add_gen_mean / blob_gen_mean
    print(f"within-scheme ratio (disease_specific / generic): class_blob={ratio_blob:.3f}, "
          f"additive_per_row={ratio_add:.3f}")
    print(f"cross-scheme ratio (additive_per_row / class_blob): disease_specific="
          f"{cross_scheme_ratio_ds:.3f}, generic={cross_scheme_ratio_gen:.3f}")
    print()

    if cross_scheme_ratio_ds < 0.95:
        verdict = "DEFLATES"
    elif cross_scheme_ratio_ds > 1.05:
        verdict = "does NOT deflate (in fact INFLATES)"
    else:
        verdict = "does not materially change"
    print(f"Plainly: additive_per_row's larger corpus {verdict} IDF for disease-specific terms "
          f"relative to class_blob (cross-scheme ratio {cross_scheme_ratio_ds:.3f}). There is no "
          f"per-row D duplication in the current code to deflate it via document-frequency "
          f"inflation — any shift here is the n-size effect on the smoothed IDF formula, not the "
          f"originally-hypothesised duplication mechanism.")
    print()


def main() -> None:
    set_seed()

    train = load_train()
    test = load_test()

    std_split = make_validation_split(train, seed=SEED)
    hard_split = make_hard_validation_split(train, seed=SEED, n=400)

    vec_kwargs = _vec_kwargs()
    k = HYPERPARAMETERS.k_neighbors

    print(f"frozen Arm 1 vectoriser config: {vec_kwargs} | k={k}")
    print(f"std val: {len(std_split.val)} items, {std_split.val.disease.nunique()} classes")
    print(f"hard val: {len(hard_split.val)} items, {hard_split.val.disease.nunique()} classes")
    print()

    std_grid = ae.run_variant_scheme_grid(std_split.fit, std_split.val, VARIANTS, vec_kwargs, k)
    hard_grid = ae.run_variant_scheme_grid(hard_split.fit, hard_split.val, VARIANTS, vec_kwargs, k)
    # Full train as the index (no held-out portion needed — there's no further
    # selection left to make on this side of the comparison).
    test_grid = ae.run_variant_scheme_grid(train, test, VARIANTS, vec_kwargs, k)

    merged = std_grid[["scheme", "variant", "accuracy"]].rename(columns={"accuracy": "std_val_acc"})
    merged = merged.merge(
        hard_grid[["scheme", "variant", "accuracy"]].rename(columns={"accuracy": "hard_val_acc"}),
        on=["scheme", "variant"],
    )
    merged = merged.merge(
        test_grid[["scheme", "variant", "accuracy"]].rename(columns={"accuracy": "test_acc"}),
        on=["scheme", "variant"],
    )

    merged["picked_by_std_val"] = merged["std_val_acc"] == merged["std_val_acc"].max()
    merged["picked_by_hard_val"] = merged["hard_val_acc"] == merged["hard_val_acc"].max()

    pd.set_option("display.width", 120)
    print("=== Arm 1 index-variant/scheme grid: std-val vs hard-val vs test ===")
    print(merged.to_string(index=False))
    print()

    rho_std, p_std = spearmanr(merged["std_val_acc"], merged["test_acc"])
    rho_hard, p_hard = spearmanr(merged["hard_val_acc"], merged["test_acc"])
    print(f"Spearman rho(std_val_acc, test_acc)  = {rho_std:.3f} (p={p_std:.3f})")
    print(f"Spearman rho(hard_val_acc, test_acc) = {rho_hard:.3f} (p={p_hard:.3f})")
    print()

    std_pick = merged.loc[merged["picked_by_std_val"], ["scheme", "variant", "test_acc"]]
    hard_pick = merged.loc[merged["picked_by_hard_val"], ["scheme", "variant", "test_acc"]]
    print("std-val argmax pick(s):")
    print(std_pick.to_string(index=False))
    print("hard-val argmax pick(s):")
    print(hard_pick.to_string(index=False))
    print()

    best_std_test_acc = std_pick["test_acc"].max()
    best_hard_test_acc = hard_pick["test_acc"].max()
    if best_std_test_acc == best_hard_test_acc:
        print(f"Both protocols' picks achieve the same test_acc ({best_std_test_acc:.3f}).")
    elif best_hard_test_acc > best_std_test_acc:
        print(
            f"hard-val's pick achieves higher test_acc "
            f"({best_hard_test_acc:.3f} vs std-val's {best_std_test_acc:.3f})."
        )
    else:
        print(
            f"std-val's pick achieves higher test_acc "
            f"({best_std_test_acc:.3f} vs hard-val's {best_hard_test_acc:.3f})."
        )

    print()

    std_val_tie_verification(train, test, std_split, vec_kwargs, k)
    idf_distortion_diagnostic(train, vec_kwargs)

    print("config.py's frozen index_variant/index_scheme were NOT changed by this script.")


if __name__ == "__main__":
    main()
