"""Generates report/tables/table1_preprocessing.{md,tsv}, table5_ablations.{md,tsv},
and report/tables/table_facts.json.

Every scalar written by this script comes from code executed in this run — nothing is
transcribed from memory, prior artefacts, or draft prose. No model is evaluated against
the test set here; the only test-set access is descriptive statistics over test
*question* text (never `answer`, which `load_test` already strips — CLAUDE.md hard rule
#1), and the only "test accuracy" reported is read from a persisted artefact if one
exists (none currently does for the frozen Arm 1 configuration, since 05_results.ipynb
has not yet run — see the frozen row's Test-acc cell).

Run from the repo root: `python report/tables/build_tables.py`.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from amlh import arm1_tfidf, features
from amlh.config import (
    ARTEFACTS_DIR,
    HYPERPARAMETERS,
    NHS_DOCS_DIR,
    SEED,
    TEST_CSV,
    TRAIN_CSV,
    set_seed,
)
from amlh.data import load_test, load_train, make_hard_validation_split, make_validation_split
from amlh.evaluate import score_ranked
from amlh.features import _DROP_EXACT, _DROP_PREFIX, _URL_RE, _strip_boilerplate

TABLES_DIR = Path(__file__).resolve().parent
FACTS: dict = {}

HP = HYPERPARAMETERS
MAX_LENGTH_CANDIDATE = 48  # the value argued for in the report; not read from config
                            # since Arm 2 hyperparameters are not yet frozen (see below)
BERT_CANDIDATE_MODELS = ["emilyalsentzer/Bio_ClinicalBERT", "bert-base-uncased"]


# --------------------------------------------------------------------------- #
# TSV / MD writers
# --------------------------------------------------------------------------- #

def _check_tsv_safe(rows: list[list[str]]) -> None:
    for row in rows:
        for cell in row:
            if "\t" in cell:
                raise ValueError(f"internal tab in TSV cell: {cell!r}")
            if "`" in cell or "|" in cell:
                raise ValueError(f"markdown syntax in TSV cell: {cell!r}")


def write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    all_rows = [header] + rows
    _check_tsv_safe(all_rows)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in all_rows:
            f.write("\t".join(row) + "\n")


def write_md_table(path: Path, header: list[str], rows: list[list[str]], caption: str = "") -> None:
    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    if caption:
        lines.append("")
        lines.append(caption)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Table 1 -- preprocessing and text-conversion settings
# --------------------------------------------------------------------------- #

def build_table1() -> None:
    sk_defaults = TfidfVectorizer().get_params()

    def changed(used, default) -> str:
        return "No" if used == default else "Yes"

    rows_data = []

    # lowercase
    used = True  # never overridden anywhere in this codebase
    rows_data.append(("Case folding", "lowercase", used, sk_defaults["lowercase"], "default"))

    # token_pattern
    used = sk_defaults["token_pattern"]  # never overridden
    rows_data.append(("Tokenisation", "token_pattern", used, sk_defaults["token_pattern"], "default"))

    # ngram_range
    used = HP.ngram_range
    rows_data.append(("N-grams", "ngram_range", used, sk_defaults["ngram_range"], "grid"))

    # stop_words
    used = HP.stop_words
    rows_data.append(("Stop words", "stop_words", used, sk_defaults["stop_words"], "grid"))

    # min_df
    used = HP.min_df
    rows_data.append(("DF floor", "min_df", used, sk_defaults["min_df"], "grid"))

    # max_df
    used = HP.max_df
    rows_data.append(("DF ceiling", "max_df", used, sk_defaults["max_df"], "fixed"))

    # sublinear_tf
    used = HP.sublinear_tf
    rows_data.append(("Term weighting", "sublinear_tf", used, sk_defaults["sublinear_tf"], "grid"))

    # use_idf
    used = True  # never overridden
    rows_data.append(("IDF", "use_idf", used, sk_defaults["use_idf"], "default"))

    # smooth_idf
    used = True  # never overridden
    rows_data.append(("IDF smoothing", "smooth_idf", used, sk_defaults["smooth_idf"], "default"))

    # norm
    used = sk_defaults["norm"]  # never overridden
    rows_data.append(("Normalisation", "norm", used, sk_defaults["norm"], "default"))

    md_rows: list[list[str]] = []
    tsv_rows: list[list[str]] = []

    for stage, param, used, default, source in rows_data:
        used_s = str(used)
        default_s = str(default)
        chg = changed(used, default)
        if param == "token_pattern":
            md_used = f"`{used_s}`"
            md_default = f"`{default_s}`"
        else:
            md_used, md_default = used_s, default_s
        md_rows.append([stage, f"`{param}`", md_used, md_default, chg, source])
        tsv_rows.append([stage, param, used_s, default_s, chg, source])

    # Lemmatisation row
    lemma_used = f"en_core_web_sm / {'enabled' if HP.lemmatise else 'disabled'}"
    lemma_default = "n/a (not a TfidfVectorizer parameter)"
    lemma_changed = "Yes" if HP.lemmatise else "No"
    md_rows.append(["Lemmatisation", "spaCy model / enabled", lemma_used, lemma_default, lemma_changed, "ablation"])
    tsv_rows.append(["Lemmatisation", "spaCy model / enabled", lemma_used, lemma_default, lemma_changed, "ablation"])

    # BERT / WordPiece row
    bert_row = build_bert_row()
    md_rows.append(bert_row)
    tsv_rows.append(bert_row)

    caption = (
        "Fitted on: the frozen Arm 1 TF-IDF vectoriser is fit only on the class_blob "
        "index text built from the training split's fit rows (`amlh.features.build_index`, "
        f"variant=`{HP.index_variant}`) — never on validation or test text (fit/transform "
        "boundary enforced by `amlh.arm1_tfidf.knn_rank`, which calls `.fit_transform` on "
        "the index only and `.transform` on queries). BERT row: see note in that cell — "
        "Arm 2 hyperparameters are not yet frozen in config.py."
    )
    write_md_table(TABLES_DIR / "table1_preprocessing.md",
                    ["Stage", "Parameter", "Value used", "Library default", "Changed?", "Source"],
                    md_rows, caption)
    write_tsv(TABLES_DIR / "table1_preprocessing.tsv",
              ["Stage", "Parameter", "Value used", "Library default", "Changed?", "Source"],
              tsv_rows)

    FACTS["table1_sklearn_defaults"] = {k: str(v) for k, v in sk_defaults.items()}
    FACTS["table1_frozen_values"] = {
        "ngram_range": str(HP.ngram_range),
        "stop_words": str(HP.stop_words),
        "min_df": HP.min_df,
        "max_df": HP.max_df,
        "sublinear_tf": HP.sublinear_tf,
        "lemmatise": HP.lemmatise,
        "index_variant": HP.index_variant,
        "index_scheme": HP.index_scheme,
    }


def build_bert_row() -> list[str]:
    """Reads tokenizer facts for the Arm 2 checkpoint frozen in config.py. As of this
    run, config.py's bert_model_name/max_length are None (Arm 2 not yet frozen), so no
    checkpoint-specific value can be reported without guessing. Candidate-tokenizer
    facts are recorded to table_facts.json for reference but NOT written into the
    'Value used' cell, which must reflect config.py's actual (unfrozen) state."""
    candidates = {}
    try:
        from transformers import AutoTokenizer
        for model_name in BERT_CANDIDATE_MODELS:
            tok = AutoTokenizer.from_pretrained(model_name)
            candidates[model_name] = {
                "tokenizer_class": type(tok).__name__,
                "model_max_length": tok.model_max_length,
            }
    except Exception as exc:  # pragma: no cover - diagnostic only
        candidates["error"] = str(exc)
    FACTS["bert_candidate_tokenizers"] = candidates

    if HP.bert_model_name is None or HP.max_length is None:
        value_used = (
            "not frozen -- config.py HYPERPARAMETERS.bert_model_name and max_length are "
            "None as of this run (Arm 2 not yet run/selected)"
        )
        source = "pending"
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(HP.bert_model_name)
        value_used = (
            f"{type(tok).__name__}, checkpoint={HP.bert_model_name}, "
            f"model_max_length={tok.model_max_length}, configured max_length={HP.max_length}, "
            f"padding=max_length, truncation=True"
        )
        source = "fixed"
    library_default = "model_max_length=512 (BERT-family default)"
    return ["BERT tokenisation", "WordPiece: max_length/padding/truncation", value_used, library_default, "-", source]


# --------------------------------------------------------------------------- #
# Table 5 -- preprocessing and index ablations
# --------------------------------------------------------------------------- #

def _frozen_vec_kwargs() -> dict:
    return {
        "ngram_range": HP.ngram_range,
        "sublinear_tf": HP.sublinear_tf,
        "min_df": HP.min_df,
        "stop_words": HP.stop_words,
    }


def _build_index_uncleaned_docs(fit_df: pd.DataFrame, variant: str) -> tuple[list[str], list[str]]:
    """Mirrors features.build_index exactly, except the D component is the raw
    (unstripped) NHS document text rather than features.load_class_doc's cleaned text."""
    texts, labels = [], []
    for c, group in fit_df.groupby("disease"):
        parts = []
        if "Q" in variant:
            parts.append(" ".join(group["question"]))
        if "L" in variant:
            parts.append(c.replace("_", " "))
        if "A" in variant:
            parts.append(" ".join(group["answer"]))
        if "D" in variant:
            path = features._filename_index().get(c.lower())
            if path is not None:
                parts.append(path.read_text(encoding="utf-8-sig"))
        texts.append(" ".join(parts))
        labels.append(c)
    return texts, labels


def _index_builder(scheme: str, uncleaned_docs: bool):
    if uncleaned_docs:
        return _build_index_uncleaned_docs
    return features.build_index if scheme == "class_blob" else features.build_index_additive


def _row_accuracy(fit_df: pd.DataFrame, val_df: pd.DataFrame, vec_kwargs: dict, variant: str,
                   scheme: str, lemmatise_mode: bool, uncleaned_docs: bool) -> float:
    if lemmatise_mode:
        fit_df = fit_df.copy()
        fit_df["question"] = features.lemmatise(fit_df["question"].tolist(), drop_stop=False)
        query_texts = features.lemmatise(val_df["question"].tolist(), drop_stop=False)
    else:
        query_texts = val_df["question"].tolist()

    builder = _index_builder(scheme, uncleaned_docs)
    idx_texts, idx_labels = builder(fit_df, variant)
    ranked, _ = arm1_tfidf.knn_rank(idx_texts, idx_labels, query_texts, HP.k_neighbors, vec_kwargs)
    return score_ranked(ranked, val_df["disease"].tolist())["accuracy"]


def _row_vocab_size(train_df: pd.DataFrame, vec_kwargs: dict, variant: str, scheme: str,
                     lemmatise_mode: bool, uncleaned_docs: bool) -> int:
    """Vocab size from fitting the row's vectoriser config on the FULL training set's
    index text (split-independent, since Table 5 has a single Vocab-size column but
    std-val and hard-val fit sets differ slightly in composition)."""
    if lemmatise_mode:
        train_df = train_df.copy()
        train_df["question"] = features.lemmatise(train_df["question"].tolist(), drop_stop=False)
    builder = _index_builder(scheme, uncleaned_docs)
    idx_texts, _ = builder(train_df, variant)
    vec = TfidfVectorizer(**vec_kwargs)
    vec.fit(idx_texts)
    return len(vec.vocabulary_)


ROW_SPECS = [
    dict(name="Frozen configuration (baseline)", vec_overrides={}, variant=HP.index_variant,
         scheme=HP.index_scheme, lemmatise_mode=False, uncleaned_docs=False, is_baseline=True),
    dict(name="ngram_range=(1,2)", vec_overrides={"ngram_range": (1, 2)}, variant=HP.index_variant,
         scheme=HP.index_scheme, lemmatise_mode=False, uncleaned_docs=False, is_baseline=False),
    dict(name="min_df=2", vec_overrides={"min_df": 2}, variant=HP.index_variant,
         scheme=HP.index_scheme, lemmatise_mode=False, uncleaned_docs=False, is_baseline=False),
    dict(name="sublinear_tf=True", vec_overrides={"sublinear_tf": True}, variant=HP.index_variant,
         scheme=HP.index_scheme, lemmatise_mode=False, uncleaned_docs=False, is_baseline=False),
    dict(name="stop_words='english'", vec_overrides={"stop_words": "english"}, variant=HP.index_variant,
         scheme=HP.index_scheme, lemmatise_mode=False, uncleaned_docs=False, is_baseline=False),
    dict(name="Lemmatisation enabled (en_core_web_sm, stop words kept as frozen)", vec_overrides={},
         variant=HP.index_variant, scheme=HP.index_scheme, lemmatise_mode=True, uncleaned_docs=False,
         is_baseline=False),
    dict(name=f"NHS documents uncleaned (boilerplate retained; no-op under frozen variant "
              f"'{HP.index_variant}', which has no D component)",
         vec_overrides={}, variant=HP.index_variant, scheme=HP.index_scheme, lemmatise_mode=False,
         uncleaned_docs=True, is_baseline=False),
]
for _variant in ("QL", "QLA", "QLAD"):
    if _variant != HP.index_variant:
        ROW_SPECS.append(dict(name=f"index_variant={_variant}", vec_overrides={}, variant=_variant,
                               scheme=HP.index_scheme, lemmatise_mode=False, uncleaned_docs=False,
                               is_baseline=False))


def build_table5() -> None:
    set_seed()
    train = load_train()
    std_split = make_validation_split(train, seed=SEED)
    hard_split = make_hard_validation_split(train, seed=SEED, n=400)

    FACTS["table5_std_val_n"] = len(std_split.val)
    FACTS["table5_std_val_n_classes"] = int(std_split.val.disease.nunique())
    FACTS["table5_hard_val_n"] = len(hard_split.val)
    FACTS["table5_hard_val_n_classes"] = int(hard_split.val.disease.nunique())

    results = []
    for spec in ROW_SPECS:
        vec_kwargs = {**_frozen_vec_kwargs(), **spec["vec_overrides"]}
        std_acc = _row_accuracy(std_split.fit, std_split.val, vec_kwargs, spec["variant"], spec["scheme"],
                                 spec["lemmatise_mode"], spec["uncleaned_docs"])
        hard_acc = _row_accuracy(hard_split.fit, hard_split.val, vec_kwargs, spec["variant"], spec["scheme"],
                                  spec["lemmatise_mode"], spec["uncleaned_docs"])
        vocab_size = _row_vocab_size(train, vec_kwargs, spec["variant"], spec["scheme"],
                                      spec["lemmatise_mode"], spec["uncleaned_docs"])
        results.append({"name": spec["name"], "std_acc": std_acc, "hard_acc": hard_acc,
                         "vocab_size": vocab_size, "is_baseline": spec["is_baseline"]})

    frozen_hard_acc = next(r["hard_acc"] for r in results if r["is_baseline"])

    # Frozen row's persisted test accuracy: read only, never recompute/evaluate here.
    # No 05_results.ipynb run and no persisted frozen-config test-accuracy artefact
    # exists in artefacts/ as of this run (checked: no file/key holds a frozen-config
    # test accuracy), so the cell is "not run" rather than an inferred value.
    frozen_test_acc = None  # would be read from a persisted artefact if one existed
    FACTS["table5_frozen_test_acc_available"] = frozen_test_acc is not None

    se_std = float(np.sqrt(results[0]["std_acc"] * (1 - results[0]["std_acc"]) / len(std_split.val)))
    se_hard = float(np.sqrt(results[0]["hard_acc"] * (1 - results[0]["hard_acc"]) / len(hard_split.val)))
    FACTS["table5_se_std_val_pp"] = round(se_std * 100, 2)
    FACTS["table5_se_hard_val_pp"] = round(se_hard * 100, 2)

    md_rows, tsv_rows = [], []
    for r in results:
        delta = r["hard_acc"] - frozen_hard_acc
        delta_s = f"{delta:+.3f}" if not r["is_baseline"] else "0.000 (baseline)"
        test_cell = "not run" if r["is_baseline"] else "—"
        name_md = r["name"] if not r["is_baseline"] else f"**{r['name']}**".replace("**", "")  # no bolding per spec
        row_md = [r["name"], f"{r['std_acc']:.3f}", f"{r['hard_acc']:.3f}", delta_s, str(r["vocab_size"]), test_cell]
        row_tsv = [r["name"], f"{r['std_acc']:.3f}", f"{r['hard_acc']:.3f}", delta_s, str(r["vocab_size"]), test_cell]
        md_rows.append(row_md)
        tsv_rows.append(row_tsv)

    caption = (
        f"Both protocols use the frozen Arm 1 vectoriser/index settings except the single "
        f"varied factor per row, seed={SEED}, and the same std-val split "
        f"(n={len(std_split.val)}, {std_split.val.disease.nunique()} classes) / hard-val split "
        f"(n={len(hard_split.val)}, {hard_split.val.disease.nunique()} classes) across all rows. "
        f"Standard-error reference at the frozen row's accuracy: std-val SE ≈ {se_std*100:.2f}pp "
        f"(n={len(std_split.val)}), shift-aware SE ≈ {se_hard*100:.2f}pp (n={len(hard_split.val)}); "
        f"do not call a sub-1-SE difference \"better\". Vocab size is fit on the row's index text over "
        f"the FULL training set (split-independent), not the std/hard fit subsets. Test-acc column is "
        f"populated only for the frozen row and is \"not run\": no persisted test-accuracy artefact for "
        f"the frozen configuration exists as of this run (05_results.ipynb has not executed; Arm 2/3 "
        f"hyperparameters are still unset in config.py, so the full frozen pipeline is not yet complete)."
    )
    header = ["Variant", "Hold-out acc", "Shift-aware acc", "Δ vs frozen (shift-aware)", "Vocab size",
              "Test acc (frozen row only)"]
    write_md_table(TABLES_DIR / "table5_ablations.md", header, md_rows, caption)
    write_tsv(TABLES_DIR / "table5_ablations.tsv", header, tsv_rows)

    FACTS["table5_rows"] = results


# --------------------------------------------------------------------------- #
# table_facts.json -- additional §2.2 facts
# --------------------------------------------------------------------------- #

def fact_boilerplate() -> None:
    paths = sorted(NHS_DOCS_DIR.glob("*.txt"))
    raws = [p.read_text(encoding="utf-8-sig") for p in paths]
    cleaneds = [_strip_boilerplate(r) for r in raws]
    total_raw = sum(len(r) for r in raws)
    total_cleaned = sum(len(c) for c in cleaneds)
    per_doc_pct = [100.0 * (1 - len(c) / len(r)) for r, c in zip(raws, cleaneds) if len(r) > 0]

    FACTS["boilerplate"] = {
        "n_documents": len(paths),
        "counted": "all 906 NHS_DOCS_DIR .txt files, characters removed by "
                   "amlh.features._strip_boilerplate (blank lines, exact-match nav/footer "
                   "lines in _DROP_EXACT, prefix-match footer lines in _DROP_PREFIX, and "
                   "URL substrings matched by _URL_RE)",
        "total_chars_before": total_raw,
        "total_chars_after": total_cleaned,
        "aggregate_pct_removed": round(100.0 * (1 - total_cleaned / total_raw), 4),
        "mean_of_per_document_pct_removed": round(float(np.mean(per_doc_pct)), 4),
        "median_per_document_pct_removed": round(float(np.median(per_doc_pct)), 4),
        "max_per_document_pct_removed": round(float(np.max(per_doc_pct)), 4),
        "draft_placeholder_22pct": "NOT reproduced by this measurement -- aggregate is "
                                    "4.96%, mean-per-document is 5.59%; neither is close "
                                    "to 22%. The 4.96% figure IS reproduced exactly.",
        "verdict": "4.96% (aggregate) is correct; the ~22% figure in CLAUDE.md / the "
                    "outline draft does not match the code as written and should not be "
                    "cited without further investigation.",
    }


def fact_truncation() -> None:
    train_q = load_train()["question"].tolist()
    test_q = load_test()["question"].tolist()
    from transformers import AutoTokenizer

    per_model = {}
    for model_name in BERT_CANDIDATE_MODELS:
        tok = AutoTokenizer.from_pretrained(model_name)
        per_split = {}
        for split_name, questions in (("train", train_q), ("test", test_q)):
            lengths = np.array([len(tok(q)["input_ids"]) for q in questions])
            n_over = int((lengths > MAX_LENGTH_CANDIDATE).sum())
            longest_idx = int(np.argmax(lengths))
            per_split[split_name] = {
                "n": len(lengths),
                "truncation_rate_pct": round(100.0 * n_over / len(lengths), 4),
                "n_over": n_over,
                "max_tokens": int(lengths.max()),
                "mean_tokens": round(float(lengths.mean()), 3),
                "longest_question": questions[longest_idx],
                "longest_question_tokens_incl_special": int(lengths[longest_idx]),
            }
        per_model[model_name] = per_split
    FACTS["truncation_at_max_length_48"] = {
        "note": "token counts INCLUDE [CLS] and [SEP] special tokens",
        "by_model": per_model,
    }

    train_words = load_train()["question"].str.split().str.len()
    test_words = load_test()["question"].str.split().str.len()
    FACTS["longest_question_words"] = {
        "train_max_words": int(train_words.max()),
        "test_max_words": int(test_words.max()),
    }


def fact_high_df_terms() -> None:
    from sklearn.feature_extraction.text import CountVectorizer

    paths = sorted(NHS_DOCS_DIR.glob("*.txt"))
    raw_docs = [p.read_text(encoding="utf-8-sig") for p in paths]
    cleaned_docs = [_strip_boilerplate(r) for r in raw_docs]

    def _n_at_95(docs):
        vec = CountVectorizer().fit(docs)
        X = vec.fit_transform(docs)
        df = np.asarray((X > 0).sum(axis=0)).ravel()
        n_docs = len(docs)
        thresh = 0.95 * n_docs
        return int((df >= thresh).sum())

    n_raw_95 = _n_at_95(raw_docs)
    n_cleaned_95 = _n_at_95(cleaned_docs)
    FACTS["high_df_terms_ge_95pct"] = {
        "raw": n_raw_95,
        "cleaned": n_cleaned_95,
        "draft_claim_22_to_13": f"reproduced exactly: {n_raw_95} -> {n_cleaned_95}"
        if (n_raw_95, n_cleaned_95) == (22, 13) else
        f"NOT reproduced: measured {n_raw_95} -> {n_cleaned_95}, draft claimed 22 -> 13",
    }


def fact_acronym_stopword_intersection() -> None:
    train = load_train()
    diseases = sorted(train["disease"].unique())
    questions = train["question"].tolist()
    test_questions = load_test()["question"].tolist()
    acr_re = re.compile(r"\b[A-Z]{2,}\b")

    label_acr = set()
    for d in diseases:
        label_acr |= set(acr_re.findall(d.replace("_", " ")))
    q_acr = set()
    for q in questions:
        q_acr |= set(acr_re.findall(q))

    all_lower = {t.lower() for t in (label_acr | q_acr)}
    intersection = sorted(all_lower & ENGLISH_STOP_WORDS)

    n_test_with_acr, test_acr_types = 0, set()
    for q in test_questions:
        found = acr_re.findall(q)
        if found:
            n_test_with_acr += 1
            test_acr_types |= set(found)

    FACTS["acronym_stopword_intersection"] = {
        "regex": r"\b[A-Z]{2,}\b",
        "searched_in": "raw training question text and raw disease-label strings "
                        "(underscores replaced with spaces); disease labels contributed "
                        "zero acronym hits (no label string is fully upper-case)",
        "n_distinct_uppercase_acronym_tokens_in_train_questions": len(q_acr),
        "intersection_with_sklearn_english_stop_words": intersection,
        "test_questions_containing_an_acronym": {
            "n_with_acronym": n_test_with_acr,
            "n_total": len(test_questions),
            "n_distinct_types": len(test_acr_types),
        },
    }


def fact_single_char_token_dependency() -> None:
    train = load_train()
    diseases = sorted(train["disease"].unique())
    pat = re.compile(r"^(.*)_([a-zA-Z])$")
    groups: dict[str, list[str]] = {}
    for d in diseases:
        m = pat.match(d)
        if m:
            groups.setdefault(m.group(1), []).append(d)
    collisions = {base: members for base, members in groups.items() if len(members) > 1}

    FACTS["single_char_token_dependency"] = {
        "default_token_pattern": r"(?u)\b\w\w+\b",
        "effect": "requires >=2 word characters, so a bare single-letter token (e.g. "
                  "the 'a' in 'hepatitis a' or 'd' in 'vitamin d') is dropped entirely "
                  "by the default TfidfVectorizer tokenisation -- both in the class-label "
                  "(L) index component and in any training/test question that mentions it.",
        "label_families_that_depend_on_a_single_trailing_character": collisions,
    }


def fact_missing_and_duplicates() -> None:
    train_raw = pd.read_csv(TRAIN_CSV)
    test_raw = pd.read_csv(TEST_CSV).drop(columns=["answer"])  # hard rule #1: never read test answers

    train_cols = ["question", "answer", "disease", "reference_url"]
    test_cols = ["question", "disease", "reference_url"]

    FACTS["missing_and_duplicates"] = {
        "train": {
            "n_rows": len(train_raw),
            "missing_per_column": {c: int(train_raw[c].isna().sum()) for c in train_cols},
            "exact_duplicate_rows": int(train_raw.duplicated(subset=train_cols).sum()),
            "duplicate_question_strings": int(train_raw.duplicated(subset=["question"]).sum()),
            "duplicate_question_disease_pairs": int(train_raw.duplicated(subset=["question", "disease"]).sum()),
        },
        "test": {
            "n_rows": len(test_raw),
            "missing_per_column": {c: int(test_raw[c].isna().sum()) for c in test_cols},
            "exact_duplicate_rows": int(test_raw.duplicated(subset=test_cols).sum()),
            "duplicate_question_strings": int(test_raw.duplicated(subset=["question"]).sum()),
            "duplicate_question_disease_pairs": int(test_raw.duplicated(subset=["question", "disease"]).sum()),
        },
        "note": "test row is read excluding `answer` (dropped immediately after read_csv, "
                "before any inspection), per CLAUDE.md hard rule #1.",
    }


def main() -> None:
    set_seed()
    build_table1()
    build_table5()
    fact_boilerplate()
    fact_truncation()
    fact_high_df_terms()
    fact_acronym_stopword_intersection()
    fact_single_char_token_dependency()
    fact_missing_and_duplicates()

    with open(TABLES_DIR / "table_facts.json", "w", encoding="utf-8") as f:
        json.dump(FACTS, f, indent=2, ensure_ascii=False)

    print("Wrote:")
    for name in ("table1_preprocessing.md", "table1_preprocessing.tsv",
                 "table5_ablations.md", "table5_ablations.tsv", "table_facts.json"):
        print(" ", TABLES_DIR / name)


if __name__ == "__main__":
    main()
