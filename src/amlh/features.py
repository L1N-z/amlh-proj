"""Feature and class-evidence index construction for Arm 1 (TF-IDF / k-NN).

No modelling code lives here — this module builds the texts that
``arm1_tfidf.knn_rank`` vectorises. ``build_index`` accepts only a fit/train
frame; a variant that needs ``answer`` raises ``KeyError`` on a test-shaped
frame rather than silently degrading (see CLAUDE.md hard rule #1).
"""

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from amlh.config import NHS_DOCS_DIR

_COMPONENTS = "QLAD"

_DROP_EXACT = {"skip to main content", "- nhs"}
_DROP_PREFIX = ("page last reviewed", "next review due", "credit:")
_URL_RE = re.compile(r"https?://\S+")


def build_vectoriser(**kwargs) -> TfidfVectorizer:
    return TfidfVectorizer(**kwargs)


@lru_cache(maxsize=1)
def _filename_index() -> dict[str, Path]:
    """stem.lower() -> Path, built once over NHS_DOCS_DIR's .txt files."""
    return {p.stem.lower(): p for p in NHS_DOCS_DIR.glob("*.txt")}


def _strip_boilerplate(raw: str) -> str:
    kept = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if low in _DROP_EXACT or low.startswith(_DROP_PREFIX):
            continue
        kept.append(_URL_RE.sub("", s).strip())
    return re.sub(r"\s+", " ", " ".join(kept)).strip()


def load_class_doc(disease: str) -> str:
    """Cleaned NHS class document for `disease`, or "" if none exists."""
    path = _filename_index().get(disease.lower())
    if path is None:
        return ""
    return _strip_boilerplate(path.read_text(encoding="utf-8-sig"))


def term_class_coverage(diseases) -> dict[str, int]:
    """For each token (sklearn's default token pattern `(?u)\\b\\w\\w+\\b`,
    lowercased) appearing in at least one class's NHS document, the number of
    distinct classes among `diseases` whose document contains it."""
    token_re = re.compile(r"(?u)\b\w\w+\b")
    counts: dict[str, int] = {}
    for d in diseases:
        tokens = set(token_re.findall(load_class_doc(d).lower()))
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
    return counts


def doc_coverage(diseases) -> dict:
    """NHS class-document coverage over `diseases` (case-insensitive filename match)."""
    idx = _filename_index()
    diseases = list(diseases)
    missing = sorted(d for d in diseases if d.lower() not in idx)
    return {
        "n_total": len(diseases),
        "n_found": len(diseases) - len(missing),
        "missing": missing,
    }


@lru_cache(maxsize=1)
def _nlp():
    import spacy

    return spacy.load("en_core_web_sm", disable=["parser", "ner"])


def lemmatise(texts: list[str], drop_stop: bool = True) -> list[str]:
    """Lemmatise `texts` with spaCy, batched via nlp.pipe. Deterministic,
    preserves list length."""
    nlp = _nlp()
    out = []
    for doc in nlp.pipe(texts):
        toks = [
            t.lemma_.lower()
            for t in doc
            if not t.is_punct and not t.is_space and (not drop_stop or not t.is_stop)
        ]
        out.append(" ".join(toks))
    return out


def build_index(fit_df: pd.DataFrame, variant: str) -> tuple[list[str], list[str]]:
    """One row per class in `fit_df`, concatenating the requested Q/L/A/D
    components in that canonical order regardless of `variant`'s letter order.

    `fit_df` must be a fit/train frame. A variant containing "A" on a frame
    without an `answer` column raises KeyError — this is how a test-frame
    leak surfaces, not incidentally.
    """
    bad = set(variant) - set(_COMPONENTS)
    if bad:
        raise ValueError(f"variant must be subset of {_COMPONENTS}, got unknown chars {bad}")

    texts: list[str] = []
    labels: list[str] = []
    for c, group in fit_df.groupby("disease"):
        parts = []
        if "Q" in variant:
            parts.append(" ".join(group["question"]))
        if "L" in variant:
            parts.append(c.replace("_", " "))
        if "A" in variant:
            parts.append(" ".join(group["answer"]))
        if "D" in variant:
            doc = load_class_doc(c)
            if doc:
                parts.append(doc)
        texts.append(" ".join(parts))
        labels.append(c)
    return texts, labels


def build_index_additive(fit_df: pd.DataFrame, variant: str) -> tuple[list[str], list[str]]:
    """One row per training example for the per-example components (Q, A),
    plus exactly one extra row per class for the class-level components (L, D).

    L and D do not scale with class size (a class doc is ~4000 chars vs an
    ~8-word question), so repeating them per example would dominate the index
    and make this scheme incomparable to `build_index`'s class-blob. Same
    canonical component order and leak-guard as `build_index`.
    """
    bad = set(variant) - set(_COMPONENTS)
    if bad:
        raise ValueError(f"variant must be subset of {_COMPONENTS}, got unknown chars {bad}")

    texts: list[str] = []
    labels: list[str] = []

    if "Q" in variant or "A" in variant:
        questions = fit_df["question"] if "Q" in variant else None
        answers = fit_df["answer"] if "A" in variant else None  # KeyError if missing, by design
        diseases = fit_df["disease"]
        for i in range(len(fit_df)):
            parts = []
            if questions is not None:
                parts.append(questions.iloc[i])
            if answers is not None:
                parts.append(answers.iloc[i])
            texts.append(" ".join(parts))
            labels.append(diseases.iloc[i])

    if "L" in variant or "D" in variant:
        for c in fit_df["disease"].unique():
            parts = []
            if "L" in variant:
                parts.append(c.replace("_", " "))
            if "D" in variant:
                doc = load_class_doc(c)
                if doc:
                    parts.append(doc)
            texts.append(" ".join(parts))
            labels.append(c)

    return texts, labels
