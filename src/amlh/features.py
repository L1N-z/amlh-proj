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
