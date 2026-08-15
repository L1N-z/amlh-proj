"""One-off measurement script for report §2.2: BERT truncation rate, NHS
document boilerplate share, and high-document-frequency term counts.

Read-only diagnostics. Does not modify config.py, features.py's cleaning
logic, or any Arm 1/2/3 selection code. Test data use is limited to test
*questions* (never answers, which load_test already strips) for the
truncation-rate count in Task 1 -- an inference-time input, not a selection
decision.
"""

import random
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer

from amlh.config import NHS_DOCS_DIR, SEED
from amlh.data import load_test, load_train
from amlh.features import _DROP_EXACT, _DROP_PREFIX, _URL_RE, _strip_boilerplate

MAX_LENGTH = 48
BERT_MODELS = ["emilyalsentzer/Bio_ClinicalBERT", "bert-base-uncased"]


def task1_truncation_rate() -> None:
    print("=" * 100)
    print("TASK 1 - BERT truncation rate at max_length=48")
    print("=" * 100)
    print("Special tokens ([CLS]/[SEP]) ARE included in every token count below.\n")

    train_q = load_train()["question"].tolist()
    test_q = load_test()["question"].tolist()
    splits = {"train": train_q, "test": test_q}

    for model_name in BERT_MODELS:
        print(f"--- {model_name} ---")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        for split_name, questions in splits.items():
            lengths = [len(tokenizer(q)["input_ids"]) for q in questions]
            lengths_arr = np.array(lengths)
            n = len(lengths_arr)
            n_over = int((lengths_arr > MAX_LENGTH).sum())
            trunc_rate = 100.0 * n_over / n

            longest_idx = int(np.argmax(lengths_arr))
            longest_q = questions[longest_idx]
            longest_len = int(lengths_arr[longest_idx])

            print(f"  [{split_name}] n={n}")
            if trunc_rate == 0.0:
                print(f"    truncation rate (> {MAX_LENGTH} tokens): 0.00% (exactly zero)")
            else:
                print(f"    truncation rate (> {MAX_LENGTH} tokens): {trunc_rate:.2f}% ({n_over}/{n})")
            print(f"    max={lengths_arr.max()}  mean={lengths_arr.mean():.2f}  "
                  f"p95={np.percentile(lengths_arr, 95):.1f}  "
                  f"p99={np.percentile(lengths_arr, 99):.1f}  "
                  f"p99.9={np.percentile(lengths_arr, 99.9):.1f}")
            print(f"    longest question ({longest_len} tokens): {longest_q!r}")
        print()


def _classify_lines(raw: str, cleaned: str) -> dict[str, int]:
    """Line-by-line character attribution matching features._strip_boilerplate's
    logic, for rule-level bookkeeping. Read-only instrumentation -- does not
    alter or reimplement the cleaning decision itself beyond mirroring it for
    counting purposes. blank/exact_drop/prefix_drop/url are counted directly;
    whitespace_norm is the exact residual (total removed minus those four) so
    the table's rows always sum to the true total removed, regardless of any
    line-ending / splitlines() accounting quirks in the other four."""
    counts = {"blank": 0, "exact_drop": 0, "prefix_drop": 0, "url": 0}
    for line in raw.splitlines():
        line_len = len(line) + 1  # +1 for the newline this line consumed in raw
        s = line.strip()
        if not s:
            counts["blank"] += line_len
            continue
        low = s.lower()
        if low in _DROP_EXACT:
            counts["exact_drop"] += line_len
            continue
        if low.startswith(_DROP_PREFIX):
            counts["prefix_drop"] += line_len
            continue
        urls = _URL_RE.findall(s)
        counts["url"] += sum(len(u) for u in urls)

    total_removed = len(raw) - len(cleaned)
    counts["whitespace_norm"] = total_removed - sum(counts.values())
    return counts


def task2_boilerplate_share() -> None:
    print("=" * 100)
    print("TASK 2 - Boilerplate share of NHS documents")
    print("=" * 100)

    paths = sorted(NHS_DOCS_DIR.glob("*.txt"))
    print(f"n_documents = {len(paths)}\n")

    raws, cleaneds = [], []
    per_doc_pct = []
    rule_totals = Counter()

    for p in paths:
        raw = p.read_text(encoding="utf-8-sig")
        cleaned = _strip_boilerplate(raw)
        raws.append(raw)
        cleaneds.append(cleaned)
        if len(raw) > 0:
            per_doc_pct.append(100.0 * (1 - len(cleaned) / len(raw)))
        rule_counts = _classify_lines(raw, cleaned)
        rule_totals.update(rule_counts)

    total_raw = sum(len(r) for r in raws)
    total_cleaned = sum(len(c) for c in cleaneds)
    agg_pct = 100.0 * (1 - total_cleaned / total_raw)
    mean_of_pct = float(np.mean(per_doc_pct))

    print(f"total chars before: {total_raw:,}")
    print(f"total chars after:  {total_cleaned:,}")
    print(f"aggregate % removed (1 - sum(cleaned)/sum(raw)): {agg_pct:.2f}%")
    print(f"mean of per-document % removed:                  {mean_of_pct:.2f}%")
    print()
    print(f"per-document % removed distribution: "
          f"min={min(per_doc_pct):.2f}%  median={np.median(per_doc_pct):.2f}%  max={max(per_doc_pct):.2f}%")
    print()

    print("Removal breakdown by rule (chars removed, % of total removed):")
    total_removed = total_raw - total_cleaned
    label_map = {
        "blank": "blank lines",
        "exact_drop": "exact-match drop lines (_DROP_EXACT)",
        "prefix_drop": "prefix-match drop lines (_DROP_PREFIX, footers)",
        "url": "URL substrings stripped from kept lines",
        "whitespace_norm": "whitespace normalisation (join + collapse, not a content rule)",
    }
    rule_sum = sum(rule_totals.values())
    for key, label in label_map.items():
        chars = rule_totals[key]
        pct_of_removed = 100.0 * chars / total_removed if total_removed else 0.0
        print(f"  {label:<58} {chars:>10,} chars  {pct_of_removed:6.2f}% of total removed")
    print(f"  {'(sum of rule attributions vs actual total removed)':<58} "
          f"{rule_sum:>10,} vs {total_removed:>10,}")
    print()

    print("Note: 'blank lines' counts newline-only separators in the raw .txt structure, not")
    print("boilerplate content per se; 'whitespace normalisation' is the join/collapse step, also")
    print("not content removal. Only exact_drop, prefix_drop, and url rows remove actual boilerplate")
    print("text (nav labels, review-date footers, credit URLs).\n")

    rng = random.Random(SEED)
    sample_idx = rng.sample(range(len(paths)), 3)
    print("Sample before/after (first 400 chars), 3 documents chosen via random.Random(SEED):")
    for i in sample_idx:
        print(f"\n--- {paths[i].name} ---")
        print("BEFORE:")
        print(raws[i][:400])
        print("AFTER:")
        print(cleaneds[i][:400])
    print()


def _df_report(name: str, docs: list[str]) -> None:
    vec = CountVectorizer()
    X = vec.fit_transform(docs)
    vocab = vec.get_feature_names_out()
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    n_docs = len(docs)

    order = np.argsort(-df)
    df_sorted = df[order]
    vocab_sorted = vocab[order]

    print(f"--- {name} corpus ---")
    print(f"vocabulary size: {len(vocab)}")

    n_exact = int((df == n_docs).sum())
    exact_terms = sorted(vocab[df == n_docs].tolist())
    print(f"terms with df == {n_docs} (present in every document): {n_exact}")
    if n_exact:
        print(f"  {exact_terms}")

    thresh_95 = 0.95 * n_docs
    mask_95 = df >= thresh_95
    n_95 = int(mask_95.sum())
    terms_95 = sorted(vocab[mask_95].tolist())[:30]
    print(f"terms with df >= 95% of documents ({thresh_95:.1f}): {n_95}")
    if n_95:
        print(f"  (up to 30 shown) {terms_95}")

    for frac in (0.90, 0.75, 0.50):
        n_at = int((df >= frac * n_docs).sum())
        print(f"terms with df >= {int(frac * 100)}%: {n_at}")

    print("top 30 terms by document frequency:")
    for term, d in zip(vocab_sorted[:30], df_sorted[:30]):
        print(f"  {term:<25} df={int(d):>4}  ({100 * int(d) / n_docs:.1f}%)")
    print()


def task3_high_df_terms() -> None:
    print("=" * 100)
    print("TASK 3 - Low-information (high document-frequency) terms across documents")
    print("=" * 100)

    paths = sorted(NHS_DOCS_DIR.glob("*.txt"))
    raw_docs = [p.read_text(encoding="utf-8-sig") for p in paths]
    cleaned_docs = [_strip_boilerplate(r) for r in raw_docs]

    _df_report("RAW (pre-cleaning)", raw_docs)
    _df_report("CLEANED (post-cleaning)", cleaned_docs)

    print("Interpretation: check the top-df term lists above by eye. Ordinary English function")
    print("words (the, and, of, is, ...) are EXPECTED at high df regardless of cleaning -- IDF")
    print("weighting already discounts them, so their presence is not evidence cleaning helped.")
    print("NHS-specific boilerplate tokens (e.g. 'nhs', 'page', 'reviewed', 'cookies') dropping out")
    print("of the high-df list after cleaning IS the relevant evidence for the report's claim.\n")


if __name__ == "__main__":
    task1_truncation_rate()
    task2_boilerplate_share()
    task3_high_df_terms()
