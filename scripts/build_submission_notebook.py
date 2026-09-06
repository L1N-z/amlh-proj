"""Generate `submission/AMLH_patient_question_classification.ipynb`.

The coursework brief asks for *one* reproducible notebook, but the project keeps all
substantive code in `src/amlh/` and splits the work over eight notebooks and two machines.
This script reconciles the two: it emits a single notebook whose §0.4 writes the twelve
`src/amlh` modules to disk with `%%writefile` and then imports them normally.

The point of doing it that way rather than pasting the code into the analysis cells is that
the notebook stays a *projection* of the package. Each module cell body is read from
`src/amlh/*.py` at build time and checked byte for byte afterwards, so the submitted notebook
cannot drift from the code that produced the frozen results.

Usage:
    python scripts/build_submission_notebook.py            # the submission notebook
    python scripts/build_submission_notebook.py --cpu-only  # CPU slice, for local verification
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src" / "amlh"
SUBMISSION_DIR = PROJECT_ROOT / "submission"
OUTPUT = SUBMISSION_DIR / "AMLH_patient_question_classification.ipynb"
CPU_OUTPUT_NAME = "AMLH_cpu_slice.ipynb"

# Dependency order: each module only imports from ones already written.
MODULES = [
    ("__init__", "package shim; lazy re-exports so `import amlh` does not pull torch"),
    ("config", "SEED, paths, and the frozen `Hyperparameters` dataclass"),
    ("data", "loading, integrity audit, and the three split designs"),
    ("features", "TF-IDF vectoriser, NHS document cleaning, index construction"),
    ("evaluate", "accuracy@k, macro-F1, MRR, McNemar's exact test, bootstrap CIs"),
    ("arm1_tfidf", "k-NN retrieval over the TF-IDF index; supervised baselines"),
    ("arm1_experiments", "Arm 1 grids, ablations, and the within-1-SE selection rule"),
    ("arm2_bert", "BERT fine-tuning, ranked prediction, epoch selection"),
    ("arm3_llm", "shortlist construction, prompt building, parsing, Arm 3 selection"),
    ("results", "frozen test run, cross-arm scoring, error analysis"),
    ("eda", "label families, sibling homogeneity, novelty calibration"),
    ("diagrams", "workflow diagrams F4 and F5"),
]

# Cells tagged "gpu" or "colab-setup" are dropped from the CPU verification slice; the slice
# stops after the cell tagged "cpu-slice-end".
GPU = "gpu"
COLAB_SETUP = "colab-setup"
CPU_SLICE_END = "cpu-slice-end"


# --------------------------------------------------------------------------------------
# cell helpers
# --------------------------------------------------------------------------------------

_CELLS: list[dict] = []


def md(text: str, tags: list[str] | None = None) -> None:
    _CELLS.append(
        {
            "cell_type": "markdown",
            "metadata": {"tags": tags or []},
            "source": _lines(text.strip("\n")),
        }
    )


def code(text: str, tags: list[str] | None = None) -> None:
    _CELLS.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"tags": tags or []},
            "outputs": [],
            "source": _lines(text.strip("\n")),
        }
    )


def module_cell(name: str) -> None:
    """Emit one `%%writefile` cell carrying a module's source verbatim."""
    body = (SRC_DIR / f"{name}.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    code(f"%%writefile src/amlh/{name}.py\n{body}".rstrip("\n"))


def _lines(text: str) -> list[str]:
    """Split into the newline-terminated list the .ipynb format wants."""
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def _module_body_from_cell(source: str) -> str:
    first, _, rest = source.partition("\n")
    assert first.startswith("%%writefile"), first
    return rest


# --------------------------------------------------------------------------------------
# §0 — setup
# --------------------------------------------------------------------------------------

def section_0() -> None:
    md(
        """
# Patient Question Classification — AMLH Coursework (NLP Dataset C)

Predict a disease label from a patient question: 906 classes, 8,891 training questions,
200 test questions. Three approaches are implemented and compared:

| | Arm | Report section |
|---|---|---|
| **Arm 1** | TF-IDF retrieval + k-NN over a class-blob index | §2.3 |
| **Arm 2** | Fine-tuned Bio_ClinicalBERT, 906-way classification head | §2.4 |
| **Arm 3** | Arm 1 shortlist re-ranked by an LLM (MediPhi-Guidelines) | §2.4 |

**This notebook is the appendix to the report and reproduces every number in it.** Section
headers carry the report section they back, so any figure or table in the text can be traced
to the cell that produced it.

### Two rules the whole protocol rests on

1. **`answer` is never an inference-time input.** The only input at prediction time is
   `question`. Training-side answers may be indexed as class evidence; test-side answers are
   stripped from the CSV before it ever reaches this runtime, and `data.load_test` strips them
   again at the loader.
2. **The test set informs no selection decision.** Every model, hyperparameter, preprocessing,
   index-variant and prompt choice is made on validation, in §2–§4. Test predictions are
   generated once, in §5, after every hyperparameter is frozen. The one declared exception is
   §1.4–§1.6, which read test *questions and labels* for distributional diagnostics — sibling
   homogeneity, near-duplication, novelty calibration. Those select nothing, and they are
   labelled where they occur.
"""
    )

    md(
        """
## §0.1 Environment

Versions are printed by the run rather than stated, so this cell describes the environment the
outputs below actually came from. Target runtime is a free Google Colab **T4** — Arms 2 and 3
need a GPU.
"""
    )

    code(
        """
# Colab ships most of this; `-U transformers` is here because MediPhi-Guidelines needs a
# recent release, and the spaCy model is a separate download that no pip dependency pulls in.
!pip install -q -U transformers accelerate
!python -m spacy download en_core_web_sm -q
""",
        tags=[COLAB_SETUP],
    )

    code(
        '''
import importlib
import platform
import sys
import time

NOTEBOOK_START = time.perf_counter()
SECTION_TIMES = {}


def mark(section):
    """Record wall clock at the end of a section; §6 prints the table."""
    elapsed = time.perf_counter() - NOTEBOOK_START
    SECTION_TIMES[section] = elapsed
    print(f"[{section}] cumulative wall clock: {elapsed / 60:.1f} min")


# ---- the one flag that changes what this notebook computes -------------------------------
# False : the two expensive sweeps (Arm 2's 24-epoch x 2-encoder comparison, Arm 3's
#         3-condition x 2-model grid) are loaded from the recorded run in
#         artefacts/precomputed/. Every *selection rule* still executes live on them.
# True  : both sweeps are re-run from scratch. Adds roughly 4-5 hours on a T4 and will not
#         fit inside one free Colab session.
RUN_FULL_SELECTION = False
# ------------------------------------------------------------------------------------------

print(f"python           {platform.python_version()}")
for name in ("numpy", "pandas", "sklearn", "scipy", "matplotlib", "seaborn",
             "spacy", "torch", "transformers"):
    try:
        print(f"{name:16s} {importlib.import_module(name).__version__}")
    except ImportError:
        print(f"{name:16s} NOT INSTALLED")

try:
    import torch
    print(f"\\ncuda available   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu              {torch.cuda.get_device_name(0)}")
        print(f"gpu memory       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except ImportError:
    print("\\ntorch not installed")

print(f"\\nRUN_FULL_SELECTION = {RUN_FULL_SELECTION}")
'''
    )

    md(
        """
## §0.2 Inputs

Everything this notebook reads arrives in **`amlh_submission_inputs.zip`**, submitted alongside
it and built by `scripts/make_submission_inputs.py`. Upload it when prompted. The archive holds:

- `data/patient_qa_classification_train.csv` — 8,891 questions over 906 classes
- `data/patient_qa_classification_test.csv` — 200 questions, **`answer` column removed**
- `data/db_nhs_qa_classification/*.txt` — the 906 NHS reference documents
- `artefacts/split_fit.csv`, `artefacts/split_val.csv` — the canonical validation split every
  result in the report was computed on. Shipped rather than regenerated, for the reason §1.3 sets
  out and demonstrates.
- `artefacts/precomputed/*.csv` — the recorded sweeps §3 and §4 load when
  `RUN_FULL_SELECTION = False`

Members are stored at project-relative paths, so unzipping at the working directory puts every
file exactly where `config.py` expects it.
"""
    )

    code(
        '''
import os
import zipfile
from pathlib import Path

ARCHIVE = "amlh_submission_inputs.zip"

try:
    from google.colab import files  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if not Path(ARCHIVE).exists():
    if IN_COLAB:
        from google.colab import files
        files.upload()  # select amlh_submission_inputs.zip
    else:
        raise FileNotFoundError(
            f"{ARCHIVE} not found in {Path.cwd()}. Place it beside the notebook and re-run."
        )

with zipfile.ZipFile(ARCHIVE) as zf:
    zf.extractall(".")
    members = zf.namelist()

print(f"unzipped {len(members)} members into {Path.cwd()}")
'''
    )

    code(
        '''
# The integrity check that a previous run of this project did not have. A missing NHS document
# directory does not raise on its own -- it makes every class document an empty string, which
# silently degrades the frozen QLAD index to QLA and invalidates the whole run. Fail here,
# loudly, before anything is computed.
import pandas as pd

docs = sorted(Path("data/db_nhs_qa_classification").glob("*.txt"))
assert len(docs) == 906, f"expected 906 NHS documents, found {len(docs)}"

train_check = pd.read_csv("data/patient_qa_classification_train.csv")
assert train_check["disease"].nunique() == 906, train_check["disease"].nunique()

test_check = pd.read_csv("data/patient_qa_classification_test.csv")
assert "answer" not in test_check.columns, "test CSV carries answers -- rebuild the archive"
assert len(test_check) == 200, len(test_check)

for name in ("split_fit.csv", "split_val.csv"):
    assert Path("artefacts", name).is_file(), f"missing artefacts/{name} -- see §1.3"

precomputed = sorted(Path("artefacts/precomputed").glob("*.csv"))

print(f"train            {len(train_check)} rows, {train_check['disease'].nunique()} classes")
print("splits           split_fit.csv, split_val.csv present (shipped, see §1.3)")
print(f"test             {len(test_check)} rows, columns {list(test_check.columns)}  <- no `answer`")
print(f"NHS documents    {len(docs)}")
print(f"precomputed      {len(precomputed)} files: {[p.name for p in precomputed]}")
'''
    )

    md(
        """
## §0.3 Source modules

The project keeps its logic in a package rather than in notebook cells, so that the same code
path serves the validation grids and the frozen test run. To submit one file while keeping that
structure, the twelve modules are written to `src/amlh/` by the `%%writefile` cells below and
then imported normally. The cell bodies are the package's real source, verbatim.

| Module | Role | Backs report section |
|---|---|---|
| `config.py` | `SEED = 44`, paths, the frozen `Hyperparameters` dataclass | §2.5 |
| `data.py` | loading, integrity audit, the three split designs | §2.1, §2.5 |
| `features.py` | TF-IDF vectoriser, NHS document cleaning, index construction | §2.2, §2.3 |
| `evaluate.py` | accuracy@k, macro-F1, MRR, McNemar's exact test, bootstrap CIs | §3.2 |
| `arm1_tfidf.py` | k-NN retrieval over the index; supervised baselines | §2.3 |
| `arm1_experiments.py` | Arm 1 grids, ablations, within-1-SE selection | §2.3, §4.2 |
| `arm2_bert.py` | BERT fine-tuning, ranked prediction, epoch selection | §2.4 |
| `arm3_llm.py` | shortlists, prompt building, output parsing, Arm 3 selection | §2.4 |
| `results.py` | frozen test run, cross-arm scoring, error analysis | §3.2, §3.3 |
| `eda.py` | label families, sibling homogeneity, novelty calibration | §3.1 |
| `diagrams.py` | workflow diagrams F4 and F5 | §2.3, §2.4 |
"""
    )

    code(
        """
from pathlib import Path

Path("src/amlh").mkdir(parents=True, exist_ok=True)
Path("artefacts").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)
print("src/amlh, artefacts and figures ready")
"""
    )

    for name, _role in MODULES:
        module_cell(name)

    code(
        '''
import sys

sys.path.insert(0, "src")

from amlh.config import ARTEFACTS_DIR, FIGURES_DIR, HYPERPARAMETERS, PROJECT_ROOT, SEED, set_seed

# PROJECT_ROOT is derived from config.py's own location -- parents[2] of src/amlh/config.py --
# so it follows the working directory rather than any absolute path.
print(f"SEED         = {SEED}")
print(f"PROJECT_ROOT = {PROJECT_ROOT}")
print(f"data present = {(PROJECT_ROOT / 'data' / 'patient_qa_classification_train.csv').is_file()}")

PRECOMPUTED_DIR = ARTEFACTS_DIR / "precomputed"
mark("0. setup")
'''
    )


# --------------------------------------------------------------------------------------
# §1 — data and preprocessing
# --------------------------------------------------------------------------------------

def section_1() -> None:
    md(
        """
# §1 Dataset and preprocessing

*Backs report §2.1 (dataset), §2.2 (preprocessing, Table 1), §3.1 (EDA, Figures 1–3).*
"""
    )

    code(
        """
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from amlh import eda, features
from amlh.data import load_test, load_train, make_validation_split, run_integrity_audit

set_seed()
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.width", 170)
pd.set_option("display.max_colwidth", 60)
"""
    )

    md(
        """
## §1.1 Load and audit

`load_test` drops the `answer` column at the loader, so no code path downstream can read a
test-side answer even by mistake. The audit reports class counts, unseen labels and train–test
near-duplication.
"""
    )

    code(
        """
train = load_train()
test = load_test()

print(f"train columns: {list(train.columns)}")
print(f"test  columns: {list(test.columns)}  <- no `answer`")

audit = run_integrity_audit(train, test)
print(json.dumps(audit, indent=2))
"""
    )

    md(
        """
## §1.2 NHS document coverage and boilerplate

The `D` component of the Arm 1 index is the NHS reference document for each class. Six labels
break the naming convention (`Bronchitis`, `Pneumonia`, …), so `features` resolves filenames
case-insensitively rather than assuming a lowercase stem. `features._strip_boilerplate` removes
navigation text, image credits and review dates — roughly 22% of each raw document — before
anything is indexed.
"""
    )

    code(
        """
coverage = features.doc_coverage(train.disease.unique())
print(json.dumps(coverage, indent=2))
assert coverage["n_found"] == coverage["n_total"] == 906, "NHS document coverage is incomplete"

with open(ARTEFACTS_DIR / "doc_coverage.json", "w") as f:
    json.dump(coverage, f, indent=2)

# What stripping actually removes, measured over the whole corpus rather than one document.
raw_chars = clean_chars = 0
for disease in train.disease.unique():
    raw_chars += len(features.load_class_doc_raw(disease))
    clean_chars += len(features.load_class_doc(disease))

print(f"\\nboilerplate stripping over all {train.disease.nunique()} documents: "
      f"{raw_chars:,} chars -> {clean_chars:,} ({1 - clean_chars / raw_chars:.1%} removed)")
print(f"\\nexample -- gallstones, first 200 chars after cleaning:")
print(features.load_class_doc("gallstones")[:200] + "...")
"""
    )

    md(
        """
## §1.3 Validation split

A single stratified hold-out of 200 items over 102 classes, mirroring the shape of the test set.
`make_validation_split` allocates the 200-item quota across classes up front, so every selected
class is guaranteed at least one validation item while keeping at least two rows in the fit half —
no class can disappear from the index.

**The split is shipped in the archive rather than regenerated here, and the reason is worth
stating.** `make_validation_split` is deterministic within an environment — calling it twice gives
the same split — but it does not reproduce the committed split bit for bit on a current
numpy/pandas. The algorithm is unchanged: the cell below confirms the shipped split has exactly
the structure this code produces. What differs is the `rng.choice` draw itself, against the
environment the split was originally made in.

Every result in the report was computed on the shipped split, so regenerating it here would move
every number in §2–§5 by roughly a standard error while appearing to reproduce them. The
comparison is printed rather than hidden, and the shipped split is what §2 onwards uses.
"""
    )

    code(
        """
fit = pd.read_csv(ARTEFACTS_DIR / "split_fit.csv")
val = pd.read_csv(ARTEFACTS_DIR / "split_val.csv")
print(f"shipped split: fit={len(fit)} ({fit.disease.nunique()} classes) | "
      f"val={len(val)} ({val.disease.nunique()} classes) | test={len(test)}")

# The structural check: does the shipped split have the shape this algorithm produces?
base, extra = divmod(200, 102)
per_class = val.disease.value_counts().value_counts().to_dict()
print(f"quota structure: base={base}, extra={extra} -> expect {extra} classes with {base + 1} "
      f"items and {102 - extra} with {base}")
print(f"shipped         : {per_class}")
assert per_class == {base + 1: extra, base: 102 - extra}, "shipped split is not this algorithm's output"

# And the honest comparison: a live call, against the shipped file.
regenerated = make_validation_split(train, seed=SEED)
again = make_validation_split(train, seed=SEED)
overlap = len(set(regenerated.val.question) & set(val.question))
print(f"\\nlive call is deterministic in-process : "
      f"{regenerated.val.question.tolist() == again.val.question.tolist()}")
print(f"live call vs shipped, items in common : {overlap}/{len(val)}")
print("=> the algorithm matches; the draw does not. §2 onwards uses the shipped split, so the "
      "numbers below are the ones the report quotes.")
"""
    )

    md(
        """
## §1.4 Figure 1 — class support, question length, label families, near-duplication

*Report Figure 1.* Panel (a) is the fact that shapes the whole design: class support is
**near-uniform** at around 10 questions per class, not long-tailed, so there is very little
per-class data for a classifier to learn a boundary from.

**Declared exception.** Panels (b) and (d) read test *questions*. This is a distributional
diagnostic and selects nothing.
"""
    )

    code(
        """
for d in (train, test):
    d["q_len"] = d.question.str.split().str.len()

cnt = train.disease.value_counts()
fam = eda.label_family(pd.Series(sorted(train.disease.unique())))
fam_counts = fam.value_counts()
near_dup_sim = eda.near_duplication_similarities(train, test)

print(f"class support: min {cnt.min()} | median {cnt.median():.0f} | max {cnt.max()} | sd {cnt.std():.2f}")

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

sns.histplot(cnt.values, bins=range(int(cnt.min()), int(cnt.max()) + 2), ax=ax[0, 0])
ax[0, 0].set(xlabel="Training questions per class", ylabel="Number of classes",
             title=f"(a) Near-uniform class support ({len(cnt)} classes)")

sns.histplot(train.q_len, bins=30, stat="density", ax=ax[0, 1], label="train")
sns.histplot(test.q_len, bins=30, stat="density", ax=ax[0, 1], label="test", color="darkorange")
ax[0, 1].legend()
ax[0, 1].set(xlabel="Question length (words)", title="(b) Question length, train vs test")

fam_counts.head(12).plot.barh(ax=ax[1, 0])
ax[1, 0].invert_yaxis()
ax[1, 0].set(xlabel="Labels in family", title="(c) Top-12 label families")

ax[1, 1].hist(near_dup_sim, bins=30)
ax[1, 1].axvline(0.9, ls="--", c="r")
ax[1, 1].set(xlabel="Max cosine to any training question",
             title="(d) Train-test near-duplication")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig1_eda.png", dpi=150)
plt.show()
"""
    )

    md(
        """
## §1.5 Preprocessing evidence — Table 1

Question-length percentiles fix BERT's `max_length=48` (§2.2): they are measured here, not
chosen. The label-family and ambiguity tables quantify how fine-grained the label space is —
355 of the 906 labels share a prefix family, and identical question strings map to more than
one disease, which is an irreducible error floor.
"""
    )

    code(
        """
q_len_percentiles = train.q_len.describe(percentiles=[.5, .9, .95, .99])
print(q_len_percentiles)
q_len_percentiles.to_csv(ARTEFACTS_DIR / "question_length_percentiles.csv", header=["value"])

family_sizes = fam_counts.rename_axis("family").reset_index(name="n_labels")
family_sizes.to_csv(ARTEFACTS_DIR / "label_family_sizes.csv", index=False)
print()
print(f"labels in a multi-member family: {family_sizes.loc[family_sizes.n_labels > 1, 'n_labels'].sum()}"
      f" of {len(fam)}")
print(family_sizes.head(6).to_string(index=False))
"""
    )

    code(
        """
examples = eda.ambiguous_examples(train, n=4)
print("identical question strings mapping to more than one disease:")
for ex in examples:
    print(f"  '{ex['question']}' -> {ex['diseases']}")

pd.DataFrame(examples).to_csv(ARTEFACTS_DIR / "ambiguous_examples.csv", index=False)
"""
    )

    md(
        """
## §1.6 Why the hold-out over-estimates — sibling homogeneity

*Report §3.1, Figure 2.* This is the measurement that explains the validation→test gap in §5.
The ~10 questions per disease were generated in one pass, so a held-out question is a **phrasing
sibling** of the ones left in training, while a test question is not. Cross-validation would not
fix it — siblings remain inside every fold.

**Declared exception.** These cells read `test.disease`. They characterise the validation–test
relationship and select nothing.
"""
    )

    code(
        """
homogeneity = eda.sibling_homogeneity(train, test)
print(homogeneity.summary)

wrong_closer = eda.wrong_class_closer_fraction(homogeneity.own, homogeneity.other)
print(f"\\nwrong class closer for {wrong_closer:.1%} of test questions")

homogeneity.summary.to_csv(ARTEFACTS_DIR / "sibling_homogeneity.csv", index=False)
"""
    )

    code(
        """
plt.figure(figsize=(8, 5))
sns.kdeplot(homogeneity.sib, label="train q -> same-class sibling", fill=True)
sns.kdeplot(homogeneity.own, label="test q -> own-class train q", fill=True)
sns.kdeplot(homogeneity.other, label="test q -> other-class train q", fill=True)
plt.xlabel("Cosine similarity")
plt.title("Sibling phrasing homogeneity explains the val-test gap")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig2_novelty.png", dpi=150)
plt.show()
"""
    )

    code(
        """
# Pruning near-sibling fit questions from the index until validation novelty matches the test
# level recovers a more realistic accuracy estimate. A secondary diagnostic, not the protocol.
test_novelty = round(float(np.median(homogeneity.own)), 3)
print(f"target novelty (test) = {test_novelty}")

rows = []
for t in (1.01, 0.70, 0.60, 0.50, 0.45, 0.40):
    val_accuracy, median_novelty = eda.novelty_calibrated_eval(fit, val, prune_threshold=t)
    rows.append({"prune_threshold": t, "val_accuracy": val_accuracy,
                 "median_own_class_novelty": median_novelty})

calibration = pd.DataFrame(rows)
print(calibration.to_string(index=False))
calibration.to_csv(ARTEFACTS_DIR / "novelty_calibration.csv", index=False)
"""
    )

    md(
        """
## §1.7 Persist the enriched audit

The split files are the shipped ones and are deliberately **not** rewritten here — overwriting
them is exactly the failure §1.3 describes. Only the audit, now carrying the coverage and
homogeneity measurements, is written.
"""
    )

    code(
        """
audit["doc_coverage"] = coverage
audit["sibling_homogeneity"] = homogeneity.summary.set_index("comparison").mean_cosine.to_dict()
audit["wrong_class_closer_fraction"] = round(wrong_closer, 4)
audit["test_median_own_class_novelty"] = test_novelty

with open(ARTEFACTS_DIR / "integrity_audit.json", "w") as f:
    json.dump(audit, f, indent=2)
print("wrote artefacts/integrity_audit.json")
print("split_fit.csv and split_val.csv left as shipped -- see §1.3")
"""
    )

    md(
        """
## §1.8 Figure 3 — PCA of TF-IDF question vectors

*Report Figure 3.* The ten largest classes projected to two dimensions. Classes overlap heavily,
which is the visual form of the same problem the sibling measurement quantifies.
"""
    )

    code(
        """
big_classes = cnt.head(10).index
subset = train[train.disease.isin(big_classes)]

vec = TfidfVectorizer(sublinear_tf=True, min_df=2).fit(subset.question)
pcs = PCA(n_components=2, random_state=SEED).fit_transform(vec.transform(subset.question).toarray())

plt.figure(figsize=(8, 6))
sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=subset.disease.values, s=25)
plt.title("PCA of TF-IDF question vectors, 10 largest classes")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig3_pca.png", dpi=150)
plt.show()

mark("1. data and preprocessing")
"""
    )


# --------------------------------------------------------------------------------------
# §2 — Arm 1
# --------------------------------------------------------------------------------------

def section_2() -> None:
    md(
        """
# §2 Arm 1 — TF-IDF retrieval + k-NN

*Backs report §2.3 (method), §4.2 (design choices), Figure 7.*

Patient questions are represented as TF-IDF vectors and classified by nearest-neighbour
retrieval against an index built from the training half. Everything in this section is chosen on
**validation only**; no test quantity is read anywhere in §2.

Selection uses a **within-1-SE, prefer-simplest** rule rather than raw argmax throughout. With
n = 200 validation items, SE ≈ 3.5pp, so a raw argmax would routinely pick a configuration the
hold-out cannot actually distinguish from a simpler one.
"""
    )

    code(
        """
from amlh import arm1_experiments as ae
from amlh import arm1_tfidf, evaluate
from amlh.data import make_hard_validation_split, make_random_split

set_seed()

fit = pd.read_csv(ARTEFACTS_DIR / "split_fit.csv")
val = pd.read_csv(ARTEFACTS_DIR / "split_val.csv")
print(f"fit={len(fit)} ({fit.disease.nunique()} classes) | val={len(val)} ({val.disease.nunique()} classes)")
"""
    )

    md(
        """
## §2.1 Vectoriser grid

`ngram_range × sublinear_tf × min_df × stop_words × k` — 80 configurations, evaluated on the
question-only (`Q`) class-blob index. Both the argmax row and the within-1-SE selection are
printed, so the effect of the rule is visible rather than asserted.
"""
    )

    code(
        """
grid_Q = ae.run_vectoriser_grid(
    fit, val, "Q",
    ngram_ranges=[(1, 1), (1, 2)],
    sublinear_tf_opts=[False, True],
    min_dfs=[1, 2],
    stop_words_opts=[None, "english"],
    ks=[1, 3, 5, 10, 20],
)
grid_Q.to_csv(ARTEFACTS_DIR / "arm1_grid_Q.csv", index=False)
print(f"{len(grid_Q)} configs run")
grid_Q.sort_values("accuracy", ascending=False).head(10)
"""
    )

    code(
        """
argmax_row_Q = grid_Q.loc[grid_Q["accuracy"].idxmax()]
selected_row_Q = ae.select_within_one_se(grid_Q, n_val=len(val))
print("argmax row:")
print(argmax_row_Q)
print()
print("selected (simplest within 1 SE) row:")
print(selected_row_Q)

best_vec_kwargs = {
    "ngram_range": selected_row_Q["ngram_range"],
    "sublinear_tf": selected_row_Q["sublinear_tf"],
    "min_df": selected_row_Q["min_df"],
    "stop_words": selected_row_Q["stop_words"],
}
best_k = int(selected_row_Q["k"])
print()
print("best_vec_kwargs (Q, from step 1):", best_vec_kwargs)
print("best_k (Q, from step 1):", best_k)
"""
    )

    md(
        """
## §2.2 Preprocessing ablation — lemmatisation and stop words

*Report §2.2, Table 1.* `raw` versus spaCy lemmatisation with and without stop-word removal, at
the selected vectoriser config. The same within-1-SE rule applies: lemmatisation is only adopted
if it beats `raw` by more than the hold-out can resolve.
"""
    )

    code(
        """
preprocessing_ablation = ae.run_preprocessing_ablation(fit, val, best_vec_kwargs, best_k)
preprocessing_ablation.to_csv(ARTEFACTS_DIR / "arm1_preprocessing_ablation.csv", index=False)
print(preprocessing_ablation.to_string(index=False))
"""
    )

    code(
        """
best_acc_ablation = preprocessing_ablation["accuracy"].max()
se_ablation = (best_acc_ablation * (1 - best_acc_ablation) / len(val)) ** 0.5
raw_acc = preprocessing_ablation.loc[preprocessing_ablation["preprocessing"] == "raw", "accuracy"].item()

if raw_acc >= best_acc_ablation - se_ablation:
    best_preprocessing = "raw"
else:
    best_preprocessing = preprocessing_ablation.loc[
        preprocessing_ablation["accuracy"].idxmax(), "preprocessing"
    ]

print(f"best accuracy: {best_acc_ablation:.3f} (SE={se_ablation:.3f}) | raw accuracy: {raw_acc:.3f}")
print("selected preprocessing (prefer raw within 1 SE):", best_preprocessing)
"""
    )

    md(
        """
## §2.3 Index variants — Q / QL / QLA / QLAD

*Report §2.3, §4.2.* The largest single lever in Arm 1. Each class contributes one indexed
"blob"; the variant controls what goes into it:

| | Component |
|---|---|
| `Q` | the class's training **q**uestions |
| `L` | the class **l**abel text |
| `A` | the training-side **a**nswers (permitted — these are training data) |
| `D` | the NHS reference **d**ocument |

`QLAD` needs full document coverage, checked before use.
"""
    )

    code(
        """
coverage = features.doc_coverage(fit["disease"].unique())
assert coverage["n_found"] == coverage["n_total"] == 906, "expected full 906/906 NHS doc coverage"
print(f"NHS document coverage: {coverage['n_found']}/{coverage['n_total']}")

index_variants = ae.run_index_variant_comparison(
    fit, val, ["Q", "QL", "QLA", "QLAD"], best_vec_kwargs, best_k
)
index_variants.to_csv(ARTEFACTS_DIR / "arm1_index_variants.csv", index=False)
print(index_variants.to_string(index=False))
"""
    )

    md(
        """
### §2.3a The shift-aware hold-out, and the pre-registered tie-break

The standard hold-out is drawn from the same one-pass generation as the fit split, so it inherits
the sibling phrasing measured in §1.6. The **shift-aware** hold-out is a class-aware sample drawn
only from training rows whose question shares no substantive word with its own disease label, so
it approximates the lexeme-absent stratum the test set resembles. It is built from training data
alone — no test text or label is read.

Its role is narrow and was **pre-registered before these numbers were seen**: the standard
hold-out decides wherever it discriminates. Only where the candidates fall within 1 SE of each
other — so the ranking is noise — does the shift-aware hold-out break the tie. A protocol cannot
overturn a comparison it does not itself resolve at better than 1 SE.
"""
    )

    code(
        """
hard_split = make_hard_validation_split(load_train(), seed=SEED)
print(f"hard fit={len(hard_split.fit)} | hard val={len(hard_split.val)} "
      f"({hard_split.val.disease.nunique()} classes)")

index_variants_hard = ae.run_index_variant_comparison(
    hard_split.fit, hard_split.val, ["Q", "QL", "QLA", "QLAD"], best_vec_kwargs, best_k
)
index_variants_hard.to_csv(ARTEFACTS_DIR / "arm1_index_variants_hardval.csv", index=False)
print(index_variants_hard.to_string(index=False))
"""
    )

    code(
        """
std_best = index_variants["accuracy"].max()
std_se = (std_best * (1 - std_best) / len(val)) ** 0.5
std_tied = index_variants.loc[index_variants["accuracy"] >= std_best - std_se, "variant"].tolist()

if len(std_tied) == 1:
    best_variant = std_tied[0]
    selection_basis = "standard hold-out (discriminates without a tie-break)"
else:
    contenders = index_variants_hard[index_variants_hard["variant"].isin(std_tied)]
    best_variant = contenders.loc[contenders["accuracy"].idxmax(), "variant"]
    hard_se = (contenders["accuracy"].max() * (1 - contenders["accuracy"].max()) / len(hard_split.val)) ** 0.5
    runner_up = contenders["accuracy"].nlargest(2).iloc[-1]
    selection_basis = (
        f"shift-aware tie-break over {std_tied}; margin over runner-up "
        f"{contenders['accuracy'].max() - runner_up:.3f} vs SE {hard_se:.3f}"
    )

print(f"standard-val accuracies: {dict(zip(index_variants['variant'], index_variants['accuracy']))}")
print(f"within-1-SE set at {std_best:.3f} +/- {std_se:.3f}: {std_tied}")
print(f"hard-val accuracies:     {dict(zip(index_variants_hard['variant'], index_variants_hard['accuracy']))}")
print(f"selected index variant:  {best_variant}  [{selection_basis}]")
"""
    )

    md(
        """
### §2.3b Variant-specific vectoriser re-check

Blob length differs enormously by variant — short questions versus long NHS prose — so the
optimum tuned on `Q` need not transfer. The grid is re-run restricted to the selected variant and
re-selected within 1 SE. If the winner differs, the variant-specific one is used downstream.
"""
    )

    code(
        """
grid_variant = ae.run_vectoriser_grid(
    fit, val, best_variant,
    ngram_ranges=[(1, 1), (1, 2)],
    sublinear_tf_opts=[False, True],
    min_dfs=[1, 2],
    stop_words_opts=[None, "english"],
    ks=[1, 3, 5, 10, 20],
)
grid_variant.to_csv(ARTEFACTS_DIR / f"arm1_grid_{best_variant}.csv", index=False)
selected_row_variant = ae.select_within_one_se(grid_variant, n_val=len(val))
print(selected_row_variant)
"""
    )

    code(
        """
variant_vec_kwargs = {
    "ngram_range": selected_row_variant["ngram_range"],
    "sublinear_tf": selected_row_variant["sublinear_tf"],
    "min_df": selected_row_variant["min_df"],
    "stop_words": selected_row_variant["stop_words"],
}
variant_k = int(selected_row_variant["k"])

config_unchanged = variant_vec_kwargs == best_vec_kwargs and variant_k == best_k
print("variant-specific config matches the step-1 config:", config_unchanged)

if not config_unchanged:
    print("Using the variant-specific winner downstream.")
    best_vec_kwargs = variant_vec_kwargs
    best_k = variant_k

print("final best_vec_kwargs:", best_vec_kwargs)
print("final best_k:", best_k)
"""
    )

    md(
        """
## §2.4 Indexing scheme — class blob vs additive per row

Does spreading the index over one row per training example, rather than one blob per class,
change accuracy? Reported on both protocols. Here the standard hold-out separates the schemes by
far more than its own SE, so the tie-break never fires — a protocol that cannot resolve a
comparison cannot overturn one that can.
"""
    )

    code(
        """
indexing_scheme = ae.run_indexing_scheme_comparison(fit, val, best_variant, best_vec_kwargs, best_k)
indexing_scheme.to_csv(ARTEFACTS_DIR / "arm1_indexing_scheme.csv", index=False)

indexing_scheme_hard = ae.run_indexing_scheme_comparison(
    hard_split.fit, hard_split.val, best_variant, best_vec_kwargs, best_k
)
indexing_scheme_hard.to_csv(ARTEFACTS_DIR / "arm1_indexing_scheme_hardval.csv", index=False)

print("standard hold-out:")
print(indexing_scheme.to_string(index=False))
print("\\nshift-aware hold-out:")
print(indexing_scheme_hard.to_string(index=False))
"""
    )

    code(
        """
std_scheme_best = indexing_scheme["accuracy"].max()
std_scheme_se = (std_scheme_best * (1 - std_scheme_best) / len(val)) ** 0.5
std_scheme_margin = std_scheme_best - indexing_scheme["accuracy"].nlargest(2).iloc[-1]

hard_scheme_best = indexing_scheme_hard["accuracy"].max()
hard_scheme_se = (hard_scheme_best * (1 - hard_scheme_best) / len(hard_split.val)) ** 0.5
hard_scheme_margin = hard_scheme_best - indexing_scheme_hard["accuracy"].nlargest(2).iloc[-1]

std_pick = indexing_scheme.loc[indexing_scheme["accuracy"].idxmax(), "scheme"]
hard_pick = indexing_scheme_hard.loc[indexing_scheme_hard["accuracy"].idxmax(), "scheme"]

print(f"standard-val: {std_pick} by {std_scheme_margin:.3f} (SE {std_scheme_se:.3f}) "
      f"-> {'decisive' if std_scheme_margin > std_scheme_se else 'not decisive'}")
print(f"hard-val:     {hard_pick} by {hard_scheme_margin:.3f} (SE {hard_scheme_se:.3f}) "
      f"-> {'decisive' if hard_scheme_margin > hard_scheme_se else 'not decisive'}")

if hard_pick != std_pick and hard_scheme_margin > hard_scheme_se and std_scheme_margin <= std_scheme_se:
    best_scheme = hard_pick
    scheme_basis = "shift-aware hold-out (standard hold-out did not discriminate)"
else:
    best_scheme = std_pick
    scheme_basis = "standard hold-out (shift-aware hold-out does not overturn it at >1 SE)"

print(f"selected indexing scheme: {best_scheme}  [{scheme_basis}]")
"""
    )

    md(
        """
## §2.5 Split-robustness check

The top-3 configurations re-run under an unstratified split across three seeds. **Absolute
accuracy is not comparable across split designs** — the fit/validation composition differs — so
the only thing under test is whether the *ranking* survives.
"""
    )

    code(
        """
top3_configs = (
    grid_Q.sort_values("accuracy", ascending=False)
    .head(3)[["ngram_range", "sublinear_tf", "min_df", "stop_words", "k"]]
    .to_dict("records")
)
for i, cfg in enumerate(top3_configs, start=1):
    print(i, cfg)

split_robustness = ae.run_split_robustness(load_train(), top3_configs, variant="Q", seeds=(42, 43, 44))
split_robustness.to_csv(ARTEFACTS_DIR / "arm1_split_robustness.csv", index=False)

mean_by_rank = split_robustness.groupby("config_rank")["accuracy"].mean()
print("\\nmean unstratified-split accuracy across 3 seeds, by config_rank:")
print(mean_by_rank.to_string())
ranking_preserved = list(mean_by_rank.sort_values(ascending=False).index) == list(mean_by_rank.index)
print(f"\\nconfig ranking order preserved under the random split: {ranking_preserved}")
"""
    )

    md(
        """
## §2.6 Supervised baselines

*Report §2.3.* Per-row TF-IDF classification with LinearSVC, LogisticRegression and
RandomForest, at the selected vectoriser config. These contrast learned decision boundaries
against neighbour matching under ~10 examples per class. They are comparison points, not
candidates for the frozen system.
"""
    )

    code(
        """
supervised_baselines = ae.run_supervised_baselines(fit, val, best_vec_kwargs)
supervised_baselines.to_csv(ARTEFACTS_DIR / "arm1_supervised_baselines.csv", index=False)
print(supervised_baselines.to_string(index=False))
"""
    )

    md(
        """
## §2.7 Figure 7 — accuracy–coverage curve

*Report Figure 7, §4.3.* As the similarity threshold for abstention rises, coverage falls and the
accuracy of the predictions that are still made rises. This is the quantitative basis for
discussing abstention as a safety mechanism in a clinical setting.
"""
    )

    code(
        """
if best_scheme == "class_blob":
    index_texts, index_labels = features.build_index(fit, best_variant)
else:
    index_texts, index_labels = features.build_index_additive(fit, best_variant)

ranked, top_sim = arm1_tfidf.knn_rank(
    index_texts, index_labels, val["question"].tolist(), best_k, best_vec_kwargs
)
gold = val["disease"].tolist()

thresholds = [round(0.05 * i, 2) for i in range(21)]
coverage_curve = evaluate.accuracy_coverage_curve(ranked, gold, top_sim, thresholds)
coverage_curve.to_csv(ARTEFACTS_DIR / "arm1_coverage_curve.csv", index=False)

fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(coverage_curve["threshold"], coverage_curve["accuracy"], marker="o",
         color="tab:blue", label="accuracy")
ax1.set_xlabel("similarity threshold")
ax1.set_ylabel("accuracy (retained predictions)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(coverage_curve["threshold"], coverage_curve["coverage"], marker="s",
         color="tab:orange", label="coverage")
ax2.set_ylabel("coverage (fraction retained)", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")

fig.suptitle("Arm 1 accuracy-coverage curve (validation)")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig7_coverage_curve.png", dpi=150)
plt.show()
"""
    )

    md(
        """
## §2.8 Freeze Arm 1

The values selected above, assembled from the DataFrames rather than typed in. These are the
values already recorded in `config.py`'s `HYPERPARAMETERS`; the assertion confirms this run
reproduces them, so everything downstream reads the frozen config rather than this section's
local variables.
"""
    )

    code(
        """
frozen_arm1 = {
    "ngram_range": tuple(int(x) for x in best_vec_kwargs["ngram_range"]),
    "min_df": int(best_vec_kwargs["min_df"]),
    "max_df": 1.0,
    "sublinear_tf": bool(best_vec_kwargs["sublinear_tf"]),
    "stop_words": best_vec_kwargs["stop_words"],
    "lemmatise": best_preprocessing != "raw",
    "index_variant": str(best_variant),
    "index_scheme": str(best_scheme),
    "k_neighbors": int(best_k),
}
print("selected by this run:")
for key, value in frozen_arm1.items():
    recorded = getattr(HYPERPARAMETERS, key)
    flag = "ok " if recorded == value else "DIFFERS"
    print(f"  {flag} {key:16s} = {value!r}   (config.py: {recorded!r})")

mismatched = [k for k, v in frozen_arm1.items() if getattr(HYPERPARAMETERS, k) != v]
assert not mismatched, f"this run did not reproduce the frozen Arm 1 config: {mismatched}"
print("\\nall Arm 1 hyperparameters reproduce config.py")
"""
    )

    md(
        """
## §2.9 Per-item predictions and the shortlist ceiling

At the frozen `k_neighbors=1` the similarity-weighted vote holds exactly one label, so Arm 1's
ranked list has length 1 and its Top-5 and MRR columns are accuracy repeated — worth stating
plainly, because they are genuine ranking metrics for Arm 2 but not here.

`knn_rank`'s optional `depth` parameter appends lower-confidence labels *below* the unchanged
head, which is what gives Arm 3 a 20-label shortlist to re-rank. The assertion proves the head is
untouched on all 200 items, so no frozen prediction is affected.
"""
    )

    code(
        """
SHORTLIST_DEPTH = 20

ranked_frozen, top_sim_frozen = ae.frozen_ranking(fit, val, HYPERPARAMETERS)
ranked_deep, top_sim_deep = ae.frozen_ranking(fit, val, HYPERPARAMETERS, depth=SHORTLIST_DEPTH)

assert [r[0] for r in ranked_frozen] == [r[0] for r in ranked_deep], \\
    "ranking-depth path changed a frozen top-1 prediction"
assert top_sim_frozen == top_sim_deep, "ranking-depth path changed the reported top similarity"
print(f"depth={SHORTLIST_DEPTH} leaves all {len(val)} frozen top-1 predictions unchanged")

arm1_val_predictions = ae.build_val_predictions(fit, val, HYPERPARAMETERS, depth=SHORTLIST_DEPTH)
arm1_val_predictions.to_csv(ARTEFACTS_DIR / "arm1_val_predictions.csv", index=False)

arm1_val_accuracy = (arm1_val_predictions["pred"] == arm1_val_predictions["gold"]).mean()
print(f"Arm 1 validation accuracy: {arm1_val_accuracy:.4f}")

arm1_shortlist_ceiling = ae.shortlist_ceiling(arm1_val_predictions, [1, 5, 10, 20])
arm1_shortlist_ceiling.to_csv(ARTEFACTS_DIR / "arm1_shortlist_ceiling.csv", index=False)
print()
print(arm1_shortlist_ceiling.to_string(index=False))

mark("2. Arm 1")
"""
    )


# --------------------------------------------------------------------------------------
# §3 — Arm 2
# --------------------------------------------------------------------------------------

def section_3() -> None:
    md(
        """
# §3 Arm 2 — fine-tuned Bio_ClinicalBERT

*Backs report §2.4 (method), Figure 6, §4.2.*

`question` → WordPiece (`max_length=48`) → encoder → 906-way linear classification head →
argmax. `answer` is never an input to this arm. Labels are encoded against the **full 906-class
training universe**, not the classes present in the fit split, so the output layer can predict
anything the frozen test run might need.

Requires a GPU. Validation only — no test quantity is read in §3.
""",
        tags=[GPU],
    )

    code(
        """
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from amlh import arm2_bert as ab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU'}")
assert device.type == "cuda", "Arms 2 and 3 need a GPU -- Runtime > Change runtime type > T4"
""",
        tags=[GPU],
    )

    md(
        """
## §3.1 Label encoding and tokenisation

`max_length=48` comes from §1.5's question-length percentiles (mean 8.4 words, 99th percentile
18) — fixed from the data, not tuned. The truncation rate it produces is printed here and belongs
in report §2.2.
""",
        tags=[GPU],
    )

    code(
        """
set_seed()

label_to_id, id_to_label = ab.encode_labels(load_train())
assert len(label_to_id) == 906
print(f"{len(label_to_id)} classes encoded over the full training universe")

MAX_LENGTH = HYPERPARAMETERS.max_length
LEARNING_RATE = HYPERPARAMETERS.learning_rate
BATCH_SIZE = HYPERPARAMETERS.batch_size
SWEEP_EPOCHS = 24
MODEL_NAMES = ["emilyalsentzer/Bio_ClinicalBERT", "bert-base-uncased"]

sanity_tokeniser = AutoTokenizer.from_pretrained("bert-base-uncased")
rate = ab.truncation_rate(fit["question"].tolist(), sanity_tokeniser, MAX_LENGTH)
print(f"truncation rate at max_length={MAX_LENGTH}: {rate:.4f}")
print(f"lr={LEARNING_RATE} | batch_size={BATCH_SIZE}  (standard fine-tuning defaults, held fixed "
      f"across both encoders and not swept -- at SE ~3.5pp a 200-item hold-out cannot resolve a "
      f"learning-rate grid)")
""",
        tags=[GPU],
    )

    md(
        """
## §3.2 Encoder comparison sweep — *gated*

`run_model_ablation` trains Bio_ClinicalBERT and `bert-base-uncased` for 24 epochs with
**identical** hyperparameters, seed and epoch budget, so the only thing that varies is the
checkpoint. That is what makes the in-domain-pretraining claim measurable rather than asserted.

**This sweep took 3,998 s (66.6 min) on a T4 across both encoders** — the wall clock recorded in
`arm2_model_ablation.csv`. With `RUN_FULL_SELECTION = False` its output is loaded from
`artefacts/precomputed/`; the selection rules in §3.3 then run live on it either way.
""",
        tags=[GPU],
    )

    code(
        """
if RUN_FULL_SELECTION:
    def report(model_name, row):
        print(f"{model_name.split('/')[-1]} | epoch {row['epoch'] + 1}/{SWEEP_EPOCHS} | "
              f"train_loss {row['train_loss']:.4f} | val_loss {row['val_loss']:.4f} | "
              f"val_acc {row['val_accuracy']:.4f}")

    set_seed()
    start = time.perf_counter()
    ablation_summary, histories, ranked_by_epoch = ab.run_model_ablation(
        fit, val, label_to_id, MODEL_NAMES,
        lr=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=SWEEP_EPOCHS,
        max_length=MAX_LENGTH, seed=SEED, on_epoch_end=report,
    )
    print(f"\\ntotal wall clock for both encoders: {time.perf_counter() - start:.1f}s")

    histories["emilyalsentzer/Bio_ClinicalBERT"].to_csv(
        ARTEFACTS_DIR / "arm2_history_bioclinicalbert.csv", index=False)
    histories["bert-base-uncased"].to_csv(
        ARTEFACTS_DIR / "arm2_history_bertbase.csv", index=False)
    ablation_summary.to_csv(ARTEFACTS_DIR / "arm2_model_ablation.csv", index=False)

    encoder_predictions = None  # rebuilt from ranked_by_epoch in §3.3
else:
    print("RUN_FULL_SELECTION = False -- loading the recorded sweep from artefacts/precomputed/")
    histories = {
        "emilyalsentzer/Bio_ClinicalBERT": pd.read_csv(PRECOMPUTED_DIR / "arm2_history_bioclinicalbert.csv"),
        "bert-base-uncased": pd.read_csv(PRECOMPUTED_DIR / "arm2_history_bertbase.csv"),
    }
    ablation_summary = pd.read_csv(PRECOMPUTED_DIR / "arm2_model_ablation.csv")
    ranked_by_epoch = None

print()
print(ablation_summary.to_string(index=False))
""",
        tags=[GPU],
    )

    md(
        """
### §3.2a Figure 6 — training and validation loss

*Report Figure 6.* Both curves for both encoders. Report §4.2 reads the train/validation gap as
evidence of memorisation under ~10 examples per class, which is only legible with both plotted.
""",
        tags=[GPU],
    )

    code(
        """
fig, ax = plt.subplots(figsize=(7, 5))
for name, history in histories.items():
    short = name.split("/")[-1]
    ax.plot(history["epoch"], history["train_loss"], marker="o", label=f"{short} train")
    ax.plot(history["epoch"], history["val_loss"], marker="o", linestyle="--", label=f"{short} val")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_title("Arm 2 - training/validation loss")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig6_loss_curves.png", dpi=150)
plt.show()
""",
        tags=[GPU],
    )

    md(
        """
## §3.3 Epoch and encoder selection — always live

Two decisions, both made on the standard hold-out. Arm 2 does not use §2.3a's shift-aware
tie-break; that rule governs Arm 1's index selection only.

1. **Epoch** — within 1 SE of the best validation accuracy, take the *fewest* epochs. This buys
   the peak's accuracy for less training.
2. **Encoder** — McNemar's exact test over the discordant pairs, not an accuracy difference
   judged against a single-proportion SE. Both encoders answer the same 200 items, so the
   comparison is paired and the items they agree on carry no evidence.

If McNemar returns p ≥ 0.05 the comparison is **reported as unresolved** and Bio_ClinicalBERT is
kept on the declared prior that an in-domain clinical encoder is the appropriate default for a
clinical task. **This can retain the lower-scoring encoder — and here it does.** That is the
intended behaviour of a prior, not an accuracy judgement.

*Disclosure:* unlike §2.3a's rule, this one was written down **after** the comparison was seen. An
earlier implementation applied the same preference as a hardcoded fallback gated on a hand-rolled
single-proportion SE; it is recorded as a rule here with the instrument corrected to McNemar.
""",
        tags=[GPU],
    )

    code(
        """
selected_by_encoder = {
    name: ab.select_best_epoch_within_one_se(history, n_val=len(val))
    for name, history in histories.items()
}
for name, row in selected_by_encoder.items():
    peak = histories[name]["val_accuracy"].max()
    se = (peak * (1 - peak) / len(val)) ** 0.5
    n_within = int((histories[name]["val_accuracy"] >= peak - se).sum())
    print(f"{name}")
    print(f"  peak val_accuracy {peak:.4f} (SE {se:.4f}) -> {n_within} of {len(histories[name])} "
          f"epochs within 1 SE")
    print(f"  selected epoch {int(row['epoch'])}, val_accuracy {row['val_accuracy']:.4f}")
""",
        tags=[GPU],
    )

    code(
        """
SHORT_NAME = {"emilyalsentzer/Bio_ClinicalBERT": "bioclinicalbert", "bert-base-uncased": "bertbase"}

if RUN_FULL_SELECTION:
    # Taken from the rankings train_model captured during the sweep -- no retraining, so these
    # are exactly the checkpoints whose accuracies were just compared.
    encoder_predictions = {}
    for model_name, row in selected_by_encoder.items():
        ranked_at_epoch = ranked_by_epoch[model_name][int(row["epoch"])]
        frame = pd.DataFrame({"question": val["question"], "gold": val["disease"]})
        frame["pred"] = [r[0] for r in ranked_at_epoch]
        for i in range(5):
            frame[f"top_{i + 1}"] = [r[i] if i < len(r) else None for r in ranked_at_epoch]
        frame.to_csv(ARTEFACTS_DIR / f"arm2_val_predictions_{SHORT_NAME[model_name]}.csv", index=False)
        encoder_predictions[model_name] = frame
else:
    encoder_predictions = {
        name: pd.read_csv(PRECOMPUTED_DIR / f"arm2_val_predictions_{short}.csv")
        for name, short in SHORT_NAME.items()
    }

for name, frame in encoder_predictions.items():
    print(f"{SHORT_NAME[name]:16s} accuracy {(frame['pred'] == frame['gold']).mean():.4f}")

encoder_mcnemar = evaluate.mcnemar_exact(
    encoder_predictions["emilyalsentzer/Bio_ClinicalBERT"]["pred"].tolist(),
    encoder_predictions["bert-base-uncased"]["pred"].tolist(),
    val["disease"].tolist(),
)
print("\\nMcNemar exact -- a = Bio_ClinicalBERT, b = bert-base-uncased, same 200 items:")
for key, value in encoder_mcnemar.items():
    print(f"  {key}: {value}")

pd.DataFrame([{"system_a": "emilyalsentzer/Bio_ClinicalBERT",
               "system_b": "bert-base-uncased", **encoder_mcnemar}]).to_csv(
    ARTEFACTS_DIR / "arm2_encoder_mcnemar.csv", index=False)
""",
        tags=[GPU],
    )

    code(
        """
ALPHA = 0.05

if encoder_mcnemar["p_value"] >= ALPHA:
    print(f"McNemar p = {encoder_mcnemar['p_value']:.4f} >= alpha = {ALPHA}")
    print("UNRESOLVED -- the hold-out does not separate these encoders. No winner is declared.")
    print("Pre-registered tie-break fires: keep the in-domain clinical encoder on the declared prior.")
    selected_model_name = "emilyalsentzer/Bio_ClinicalBERT"
    encoder_tie_break_fired = True
else:
    print(f"McNemar p = {encoder_mcnemar['p_value']:.4f} < alpha = {ALPHA}: the difference is reliable.")
    print("Tie-break does not fire; measured accuracy decides.")
    accs = {n: (f["pred"] == f["gold"]).mean() for n, f in encoder_predictions.items()}
    selected_model_name = max(accs, key=accs.get)
    encoder_tie_break_fired = False

selected_row = selected_by_encoder[selected_model_name]
selected_epochs = int(selected_row["epoch"]) + 1

print(f"\\nselected encoder : {selected_model_name}")
print(f"selected epochs  : {selected_epochs} (0-indexed epoch {int(selected_row['epoch'])})")
if encoder_tie_break_fired:
    kept = (encoder_predictions[selected_model_name]["pred"]
            == encoder_predictions[selected_model_name]["gold"]).mean()
    other = [n for n in encoder_predictions if n != selected_model_name][0]
    other_acc = (encoder_predictions[other]["pred"] == encoder_predictions[other]["gold"]).mean()
    if other_acc > kept:
        print(f"NOTE: this retains the LOWER-scoring encoder ({kept:.3f} vs {other_acc:.3f} for "
              f"{other}). The prior selected it, not the accuracy.")

assert selected_model_name == HYPERPARAMETERS.bert_model_name, "encoder differs from config.py"
assert selected_epochs == HYPERPARAMETERS.num_epochs, "epoch count differs from config.py"
print("\\nboth reproduce config.py")
""",
        tags=[GPU],
    )

    md(
        """
## §3.4 Train the frozen model

Trained at the selected epoch count on `split_fit` only — **no refit on fit + val**, so the model
evaluated on test in §5 is the exact model that was validated here, and the validation→test
comparison in §5.6 describes one object rather than two.

The state dict is kept and the GPU copy released, so the encoder is not resident while §4 loads a
3.8B-parameter generator.
""",
        tags=[GPU],
    )

    code(
        """
set_seed()
start = time.perf_counter()

state_dict, frozen_history, _ = ab.train_model(
    fit, val, label_to_id, HYPERPARAMETERS.bert_model_name,
    lr=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=selected_epochs,
    max_length=MAX_LENGTH, seed=SEED,
)
print(f"trained {selected_epochs} epochs in {(time.perf_counter() - start) / 60:.1f} min")
print(frozen_history.tail().to_string(index=False))
""",
        tags=[GPU],
    )

    code(
        """
model = AutoModelForSequenceClassification.from_pretrained(
    HYPERPARAMETERS.bert_model_name, num_labels=len(label_to_id)
)
model.load_state_dict(state_dict)
model = model.to(device)
tokeniser = AutoTokenizer.from_pretrained(HYPERPARAMETERS.bert_model_name)

# top_k=None ranks all 906 labels. Arm 1 ranks every class, so a top-5 ranking here would make
# Arm 2's MRR an MRR@5 and the two arms' columns non-comparable.
val_ranked = ab.predict_ranked(
    model, tokeniser, val["question"].tolist(), id_to_label,
    max_length=MAX_LENGTH, batch_size=BATCH_SIZE, device=device, top_k=None,
)
arm2_scores = evaluate.score_ranked(val_ranked, val["disease"].tolist())
print(arm2_scores)

# The refit replays the same seed and data, so its top-1 should reproduce what the sweep
# captured for this encoder. Printed rather than asserted: cuDNN kernel selection is not
# bit-deterministic, so a small disagreement is a hardware artefact. A large one is a logic error.
captured_top1 = encoder_predictions[HYPERPARAMETERS.bert_model_name]["pred"].tolist()
agreement = sum(a == b[0] for a, b in zip(captured_top1, val_ranked)) / len(val)
print(f"refit vs sweep-captured top-1 agreement: {agreement:.4f}")

arm2_val_predictions = pd.DataFrame({"question": val["question"], "gold": val["disease"]})
arm2_val_predictions["pred"] = [r[0] for r in val_ranked]
for i in range(5):
    arm2_val_predictions[f"top_{i + 1}"] = [r[i] if i < len(r) else None for r in val_ranked]
arm2_val_predictions.to_csv(ARTEFACTS_DIR / "arm2_val_predictions.csv", index=False)

pd.DataFrame([{**arm2_scores, "model_name": HYPERPARAMETERS.bert_model_name,
               "epoch": selected_epochs - 1, "refit_agreement": agreement}]).to_csv(
    ARTEFACTS_DIR / "arm2_val_metrics.csv", index=False)

del model
torch.cuda.empty_cache()
print(f"encoder released | GPU allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
mark("3. Arm 2")
""",
        tags=[GPU],
    )


# --------------------------------------------------------------------------------------
# §4 — Arm 3
# --------------------------------------------------------------------------------------

def section_4() -> None:
    md(
        """
# §4 Arm 3 — retrieval shortlist + LLM re-ranking

*Backs report §2.4 (method), Table 2 (prompts), §3.3, §4.1.*

The frozen Arm 1 retriever hands the LLM a 20-label shortlist; the LLM picks one by name. This is
the Week 10 practical's diagnosis-selection task with the candidate list narrowed from all 906
diseases to the retriever's shortlist, and the generator is called through the same
`pipe(prompt, max_new_tokens=...)` interface.

The retriever's own top-1 is already a prediction, so **Arm 1 is the baseline this arm has to
beat**, not a component whose contribution can be assumed positive.
""",
        tags=[GPU],
    )

    code(
        """
from amlh import arm3_llm as a3
from amlh import results

set_seed()

shortlist_k = a3.shortlist_k(HYPERPARAMETERS)
n_shots = a3.n_shots(HYPERPARAMETERS)
primary_model = a3.selected_model_name(HYPERPARAMETERS)
secondary_model = a3.secondary_model_name(HYPERPARAMETERS)
prompt_modes = ["zero_shot", "few_shot", "cot"]

print({
    "shortlist_k": shortlist_k,
    "llm_temperature": a3.llm_temperature(HYPERPARAMETERS),
    "n_shots": n_shots,
    "max_new_tokens": a3.max_new_tokens_for("zero_shot"),
    "cot_max_new_tokens": a3.max_new_tokens_for("cot"),
    "primary_model": primary_model,
    "secondary_model": secondary_model,
})
""",
        tags=[GPU],
    )

    md(
        """
## §4.1 The shortlist, and proof it is the frozen one

`build_shortlist_ranking` runs the frozen Arm 1 configuration from `config.py`. The assertion then
checks rank 1 against §2.9's `arm1_val_predictions.csv`, item for item.

This guard exists because of a real failure: on an earlier run the NHS document directory was
absent, `load_class_doc` returned `""` for all 906 classes, and the frozen **QLAD** index silently
ran as **QLA**. Every number from that run was off-config and was discarded. Rank 1 is a sharp
fingerprint of the index that produced it, so a single disagreement stops the notebook before a
prompt is sent.
""",
        tags=[GPU],
    )

    code(
        """
set_seed()
shortlist_rankings, top_sim = a3.build_shortlist_ranking(fit, val, HYPERPARAMETERS, depth=shortlist_k)

check = a3.assert_reproduces_arm1(shortlist_rankings, arm1_val_predictions)
print(f"shortlist top-1 reproduces Arm 1 on all {check['n']} items")

gold = val["disease"].tolist()
arm1_top1 = [r[0] for r in shortlist_rankings]
in_shortlist = sum(g in r for g, r in zip(gold, shortlist_rankings)) / len(gold)
print(f"Arm 1 top-1 accuracy on these items : {sum(p == g for p, g in zip(arm1_top1, gold)) / len(gold):.3f}")
print(f"gold present in the shortlist       : {in_shortlist:.3f}  (the ceiling the LLM selects against)")
""",
        tags=[GPU],
    )

    md(
        """
## §4.2 Prompt budget — Table 2

*Report Table 2.* Three prompt conditions — `zero_shot`, `few_shot` (2 exemplars drawn from the
fit split only), and `cot`. Lengths are measured with the primary model's own tokeniser before any
condition runs, and the truncation rate at 512 tokens is printed rather than assumed.

`cot` gets 128 new tokens against 20 for the others: chain-of-thought has to emit reasoning *and*
a final answer, and an earlier run capped every condition at 8 new tokens, which made
chain-of-thought structurally impossible.
""",
        tags=[GPU],
    )

    code(
        """
set_seed()
examples = a3.build_examples(fit, n=n_shots, seed=SEED)
budget_tokeniser = AutoTokenizer.from_pretrained(primary_model)

prompts_by_mode = {}
budget_rows = []
for mode in prompt_modes:
    prompts = a3.build_prompts_for_condition(
        val, shortlist_rankings, mode, examples=examples if mode == "few_shot" else None
    )
    prompts_by_mode[mode] = prompts
    summary = a3.prompt_token_lengths(prompts, budget_tokeniser, max_length=512)
    budget_rows.append({
        "condition": mode,
        "min_tokens": summary["min"], "median_tokens": summary["median"],
        "mean_tokens": summary["mean"], "p95_tokens": summary["p95"],
        "max_tokens": summary["max"], "truncation_rate_512": summary["truncation_rate"],
    })

budget_df = pd.DataFrame(budget_rows)
budget_df.to_csv(ARTEFACTS_DIR / "arm3_prompt_budget.csv", index=False)
print(budget_df.to_string(index=False))

with open(ARTEFACTS_DIR / "arm3_prompts.txt", "w", encoding="utf-8") as f:
    f.write(a3.build_prompt_table(prompts_by_mode))

print()
print("--- one zero-shot prompt, as the model receives it ---")
print(a3.flatten_messages(prompts_by_mode["zero_shot"][0]))
""",
        tags=[GPU],
    )

    md(
        """
## §4.3 Condition × model grid — *gated*

Three prompt conditions on two generators. **The recorded run took 1,448 s (24.1 min) on a T4**,
summed over the six cells from `arm3_prompt_conditions.csv`; `cot` dominates that at 758 s + 395 s
because of its 128-token budget. With `RUN_FULL_SELECTION = False` the six cells are loaded from
`artefacts/precomputed/`, and the selection rules in §4.4 run live on them either way.
""",
        tags=[GPU],
    )

    code(
        """
if RUN_FULL_SELECTION:
    def run_grid(model_name):
        frames, rows = {}, []
        # generator_session frees the GPU on the exception path as well as the success one.
        # run_condition calls the pipe once per item, so a failure part-way through the grid
        # would otherwise leave a multi-GB generator resident and make a retry load a second
        # copy on top of it.
        with a3.generator_session(model_name, device=device) as (_, _, pipe):
            print(f"loaded {model_name} on {device}")
            for mode in prompt_modes:
                start = time.perf_counter()
                pred_df, metrics, _ = a3.run_condition(
                    fit, val, HYPERPARAMETERS, mode=mode, pipe=pipe,
                    examples=examples if mode == "few_shot" else None,
                    shortlist_depth=shortlist_k, model_name=model_name,
                    shortlist_rankings=shortlist_rankings, top_sim=top_sim,
                )
                metrics["wall_clock_sec"] = time.perf_counter() - start
                frames[mode] = pred_df
                rows.append(metrics)
                print(f"{mode:>10}: acc={metrics['accuracy']:.3f}  arm1={metrics['arm1_accuracy']:.3f}  "
                      f"fallback={metrics['fallback_rate']:.3f}  ({metrics['wall_clock_sec']:.0f}s)")
        return frames, rows

    set_seed()
    condition_frames, primary_rows = run_grid(primary_model)
    secondary_frames, secondary_rows = run_grid(secondary_model)

    condition_df = pd.DataFrame(primary_rows)
    ablation_df = pd.concat([condition_df, pd.DataFrame(secondary_rows)], ignore_index=True)
    pd.concat(list(condition_frames.values()) + list(secondary_frames.values()),
              ignore_index=True).to_csv(ARTEFACTS_DIR / "arm3_val_predictions.csv", index=False)
else:
    print("RUN_FULL_SELECTION = False -- loading the recorded grid from artefacts/precomputed/")
    all_preds = pd.read_csv(PRECOMPUTED_DIR / "arm3_val_predictions.csv")
    ablation_df = pd.read_csv(PRECOMPUTED_DIR / "arm3_prompt_conditions.csv")

    def frames_for(model_name):
        return {
            mode: all_preds[(all_preds["condition"] == mode)
                            & (all_preds["model_name"] == model_name)].reset_index(drop=True)
            for mode in prompt_modes
        }

    condition_frames = frames_for(primary_model)
    secondary_frames = frames_for(secondary_model)
    condition_df = ablation_df[ablation_df["model_name"] == primary_model].reset_index(drop=True)

ablation_df.to_csv(ARTEFACTS_DIR / "arm3_prompt_conditions.csv", index=False)
print()
print(ablation_df.to_string(index=False))
""",
        tags=[GPU],
    )

    md(
        """
### §4.3a Did the LLM earn its place?

`accuracy_minus_arm1` is the quantity that matters: the LLM re-ranks a shortlist whose top-1 is
already a prediction. McNemar pairs each condition against that top-1 over the same items.
""",
        tags=[GPU],
    )

    code(
        """
vs_arm1_df = a3.condition_vs_arm1_mcnemar(condition_frames)
vs_arm1_df.to_csv(ARTEFACTS_DIR / "arm3_vs_arm1_mcnemar.csv", index=False)
print(vs_arm1_df.to_string(index=False))
""",
        tags=[GPU],
    )

    md(
        """
## §4.4 Two-stage selection — always live

A 6-cell grid on a 200-item hold-out (SE ≈ 3.5pp) cannot support six-way selection, so the choice
is made in two stages, in a **pre-registered order** fixed before the run:

1. **Prompt condition**, on the primary generator alone, by pairwise McNemar. If no condition
   separates at p < 0.05 the comparison is reported as unresolved and `zero_shot` is kept on the
   declared prior of the simplest prompt — fewest tokens, no exemplar-selection confound, lowest
   inference cost. **This can retain a lower-scoring condition.**
2. **Model**, at the already-selected condition, by McNemar over the same items. If p ≥ 0.05 the
   in-domain clinical model is kept on the same prior as §3.3.

Unlike §3.3's encoder rule, this one *was* pre-registered before any condition was run. The
remaining grid cells are reported for transparency and select nothing.
""",
        tags=[GPU],
    )

    code(
        """
condition_mcnemar_df = a3.pairwise_condition_mcnemar(condition_frames)
selected_mode, condition_tie_break_fired = a3.select_prompt_mode(condition_df, condition_mcnemar_df)
condition_mcnemar_df.to_csv(ARTEFACTS_DIR / "arm3_condition_mcnemar.csv", index=False)

print("stage 1 -- prompt condition, on the primary generator:")
print(condition_mcnemar_df.to_string(index=False))
print()
print({"selected_mode": selected_mode, "tie_break_fired": condition_tie_break_fired})
if condition_tie_break_fired:
    smallest = condition_mcnemar_df.loc[condition_mcnemar_df["p_value"].idxmin()]
    print(f"UNRESOLVED: no condition separates at p < 0.05 (smallest p = {smallest['p_value']:.4f}, "
          f"{smallest['condition_a']} vs {smallest['condition_b']}).")
    print("zero_shot kept on the declared simplest-prompt prior.")
    best_acc_mode = condition_df.loc[condition_df["accuracy"].idxmax()]
    if best_acc_mode["condition"] != selected_mode:
        chosen_acc = condition_df.loc[condition_df["condition"] == selected_mode, "accuracy"].item()
        print(f"NOTE: this retains the LOWER-scoring condition ({selected_mode} {chosen_acc:.3f} vs "
              f"{best_acc_mode['condition']} {best_acc_mode['accuracy']:.3f}). The prior selected it.")
""",
        tags=[GPU],
    )

    code(
        """
model_mcnemar_df = a3.model_mcnemar(condition_frames[selected_mode], secondary_frames[selected_mode])
selected_generator, model_tie_break_fired = a3.select_model(model_mcnemar_df, HYPERPARAMETERS)
model_mcnemar_df.to_csv(ARTEFACTS_DIR / "arm3_model_mcnemar.csv", index=False)

print(f"stage 2 -- model, at the selected condition ({selected_mode}):")
print(model_mcnemar_df.to_string(index=False))
print()
print({"selected_generator": selected_generator, "tie_break_fired": model_tie_break_fired})
if model_tie_break_fired:
    print("UNRESOLVED: McNemar does not separate the generators; the in-domain clinical model is "
          "kept on the declared prior.")
else:
    print("RESOLVED on measured accuracy -- the clinical prior was available but never had to fire.")

assert selected_mode == HYPERPARAMETERS.prompt_mode, "prompt_mode differs from config.py"
assert selected_generator == HYPERPARAMETERS.arm3_model_name, "arm3_model_name differs from config.py"
print("\\nboth reproduce config.py")
""",
        tags=[GPU],
    )

    md(
        """
## §4.5 Run the frozen condition live

The selected cell — `zero_shot` on the frozen generator — re-run on validation now, so the
selected system is executed in this notebook rather than only loaded. The generator stays resident
for the test run in §5.5.
""",
        tags=[GPU],
    )

    code(
        """
set_seed()
# Loaded bare rather than through a3.generator_session: this generator has to stay resident
# across cells, through the test run in §5.4, and a `with` block cannot span notebook cells.
# It is freed explicitly at the end of §5.4.
_, arm3_model_obj, arm3_pipe = a3.load_generator(HYPERPARAMETERS.arm3_model_name, device=device)
print(f"loaded {HYPERPARAMETERS.arm3_model_name} on {device}")

start = time.perf_counter()
arm3_val_frozen, arm3_val_metrics, _ = a3.run_condition(
    fit, val, HYPERPARAMETERS,
    mode=HYPERPARAMETERS.prompt_mode, pipe=arm3_pipe, examples=examples,
    shortlist_depth=shortlist_k, model_name=HYPERPARAMETERS.arm3_model_name,
    shortlist_rankings=shortlist_rankings, top_sim=top_sim,
)
print(f"ran 200 validation items in {time.perf_counter() - start:.1f}s")
for key, value in arm3_val_metrics.items():
    print(f"  {key}: {value}")

recorded_acc = ablation_df.loc[
    (ablation_df["condition"] == HYPERPARAMETERS.prompt_mode)
    & (ablation_df["model_name"] == HYPERPARAMETERS.arm3_model_name), "accuracy"].item()
print(f"\\nlive run {arm3_val_metrics['accuracy']:.4f} | recorded {recorded_acc:.4f} | "
      f"delta {abs(arm3_val_metrics['accuracy'] - recorded_acc):.4f}")
""",
        tags=[GPU],
    )

    md(
        """
### §4.5a What the LLM did to the shortlist

*Report §3.3, §4.1.* Arm 3's headline accuracy blends three populations that behave completely
differently, and reporting it undivided makes the arm look uniformly weak rather than weak in one
diagnosable way. `arm1_accuracy` on each stratum is the counterfactual — what the shortlist alone
would have scored on those same items.

The fallback rate is **not** a parser defect: none of the fallback outputs match any of the 906
labels, so they are free-text hallucinations and the parser is behaving correctly. It is reported
as a finding rather than tuned away.
""",
        tags=[GPU],
    )

    code(
        """
arm3_val_decomposition = results.rerank_decomposition(condition_frames[selected_mode])
display(arm3_val_decomposition)

fallbacks = condition_frames[selected_mode][condition_frames[selected_mode]["fallback_fired"]]
label_universe = set(load_train()["disease"].unique())
matched = sum(str(r).strip() in label_universe for r in fallbacks["raw_output"])
print(f"\\nfallback outputs matching any of the 906 labels: {matched} of {len(fallbacks)} "
      f"-- the parser is correct, these are hallucinations")
""",
        tags=[GPU],
    )

    md(
        """
## §4.6 Cross-arm comparison on validation

*Report §3.2.* All three arms have now answered the same 200 validation items, so they can be
compared pairwise. This is the comparison §5.6 measures the optimism of each arm against, and it
is computed here, before any test question is read.
""",
        tags=[GPU],
    )

    code(
        """
from itertools import combinations

val_arms = {
    "arm1_tfidf_knn": arm1_val_predictions,
    "arm2_bio_clinicalbert": arm2_val_predictions,
    "arm3_llm_rerank": condition_frames[selected_mode],
}
val_gold = results.assert_query_aligned(val_arms)
print(f"query-aligned on {len(val_gold)} validation items across {len(val_arms)} arms\\n")

rows = []
for name, frame in val_arms.items():
    pred = frame["pred"].tolist()
    ci = evaluate.bootstrap_accuracy_ci(pred, val_gold, seed=SEED)
    rows.append({"arm": name, "n": len(val_gold),
                 "accuracy": sum(p == g for p, g in zip(pred, val_gold)) / len(val_gold),
                 "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]})
cross_arm_val = pd.DataFrame(rows)
cross_arm_val.to_csv(ARTEFACTS_DIR / "cross_arm_val_comparison.csv", index=False)
print(cross_arm_val.to_string(index=False))

mcnemar_rows = [
    {"system_a": a, "system_b": b,
     **evaluate.mcnemar_exact(val_arms[a]["pred"].tolist(), val_arms[b]["pred"].tolist(), val_gold)}
    for a, b in combinations(val_arms, 2)
]
cross_arm_val_mcnemar = pd.DataFrame(mcnemar_rows)
cross_arm_val_mcnemar.to_csv(ARTEFACTS_DIR / "cross_arm_val_mcnemar.csv", index=False)
print()
for row in cross_arm_val_mcnemar.itertuples(index=False):
    verdict = "RESOLVED" if row.p_value < 0.05 else "UNRESOLVED"
    print(f"{row.system_a:24s} vs {row.system_b:24s} p = {row.p_value:.4g}  -> {verdict}")

mark("4. Arm 3")
""",
        tags=[GPU],
    )


# --------------------------------------------------------------------------------------
# §5 — test run
# --------------------------------------------------------------------------------------

def section_5() -> None:
    md(
        """
# §5 The frozen test run

*Backs report §3.2 (performance comparison), §3.3 (error analysis), Figures 8–9.*

**This is the only section that evaluates on the test set, and it runs after every hyperparameter
is frozen.** Nothing here selects anything: no variant, no checkpoint, no prompt, no threshold.
Every choice was made on validation in §2–§4 and is read from `config.HYPERPARAMETERS`.

Three constraints govern it:

1. **`answer` is never an inference-time input.** The column is stripped from the shipped CSV and
   `results.assert_no_answer_column` re-checks at every entry point.
2. **Models are fit on `split_fit` only** — no refit on fit + val. The tested model is the exact
   model that was validated, so §5.6's optimism analysis describes one object. The forgone data is
   200 of 8,891 items (2.2%), negligible against a 3.5pp SE.
3. **No final system is nominated.** The brief asks for a comparison of algorithms, not the
   nomination of a winner, so all three arms are reported side by side.
"""
    )

    md(
        """
## §5.1 Freeze check

Confirm every hyperparameter this run depends on is actually frozen. A `None` here would mean a
choice was never made and the test run would be silently inventing one.

Test *questions and labels* have already been read once, in §1.4–§1.6, for the distributional
diagnostics declared there; those select nothing. This is the first point at which a test question
is scored, and it comes after the freeze.
"""
    )

    code(
        """
from amlh import results  # also imported in §4; repeated so §5 reads standalone

set_seed()

FROZEN_FIELDS = [
    "ngram_range", "min_df", "max_df", "sublinear_tf", "lemmatise",
    "index_variant", "index_scheme", "k_neighbors",
    "bert_model_name", "max_length", "learning_rate", "batch_size", "num_epochs",
    "shortlist_k", "llm_temperature", "prompt_mode", "n_shots",
    "arm3_model_name", "arm3_max_new_tokens",
]
frozen = {name: getattr(HYPERPARAMETERS, name) for name in FROZEN_FIELDS}
unset = [name for name, value in frozen.items() if value is None]
assert not unset, f"hyperparameters still unfrozen: {unset}"

print(f"SEED = {SEED}")
for name, value in frozen.items():
    print(f"  {name:24s} = {value!r}")
print(f"\\nall {len(frozen)} hyperparameters frozen -- the first test question is SCORED from the "
      f"next cell on (§1.4-§1.6 read test for the declared diagnostics only)")
"""
    )

    code(
        """
test = load_test()
results.assert_no_answer_column(test)

print(f"fit  = {len(fit):5d} rows, {fit.disease.nunique()} classes")
print(f"test = {len(test):5d} rows, {test.disease.nunique()} classes")
print(f"test columns: {list(test.columns)}  <- no `answer`")
print(f"test classes absent from split_fit: {len(set(test.disease) - set(fit.disease))}")
"""
    )

    md(
        """
## §5.2 Arm 1 on test

`require_doc_coverage` raises rather than degrading when the `D` component is missing — the guard
added after the QLA incident described in §4.1. The ranking uses the same `frozen_ranking` code
path the validation grid used, at the same hyperparameters, so the test and validation numbers
describe one system rather than two implementations that happen to agree.
"""
    )

    code(
        """
print(f"doc coverage: {features.require_doc_coverage(fit.disease.unique())}")

set_seed()
arm1_test = results.build_test_predictions(fit, test, HYPERPARAMETERS)
arm1_test.to_csv(ARTEFACTS_DIR / "arm1_test_predictions.csv", index=False)

print(f"Arm 1 test accuracy: {(arm1_test['pred'] == arm1_test['gold']).mean():.4f} "
      f"over {len(arm1_test)} items")
arm1_test[["question", "gold", "pred", "top_sim"]].head()
""",
        tags=[CPU_SLICE_END],
    )

    md(
        """
## §5.3 Arm 2 on test

The encoder is rebuilt from §3.4's state dict — the same weights, no retraining — then released
again so the generator has the GPU to itself.
""",
        tags=[GPU],
    )

    code(
        """
model = AutoModelForSequenceClassification.from_pretrained(
    HYPERPARAMETERS.bert_model_name, num_labels=len(label_to_id)
)
model.load_state_dict(state_dict)
model = model.to(device)

truncation = ab.truncation_rate(test["question"].tolist(), tokeniser, HYPERPARAMETERS.max_length)
print(f"test truncation rate at max_length={HYPERPARAMETERS.max_length}: {truncation:.4f}")

test_ranked = ab.predict_ranked(
    model, tokeniser, test["question"].tolist(), id_to_label,
    max_length=HYPERPARAMETERS.max_length, batch_size=HYPERPARAMETERS.batch_size,
    device=device, top_k=None,
)

arm2_test = pd.DataFrame({"question": test["question"], "gold": test["disease"]})
arm2_test["pred"] = [r[0] for r in test_ranked]
for i in range(results.SHORTLIST_DEPTH):
    arm2_test[f"top_{i + 1}"] = [r[i] if i < len(r) else None for r in test_ranked]
arm2_test.to_csv(ARTEFACTS_DIR / "arm2_test_predictions.csv", index=False)

print(f"Arm 2 test accuracy: {(arm2_test['pred'] == arm2_test['gold']).mean():.4f}")

del model
torch.cuda.empty_cache()
""",
        tags=[GPU],
    )

    md(
        """
## §5.4 Arm 3 on test

The shortlist is rebuilt on this runtime and asserted against §5.2's `arm1_test_predictions.csv`
item for item before a single prompt is sent — the same guard as §4.1, applied to the test
shortlist.
""",
        tags=[GPU],
    )

    code(
        """
set_seed()
features.require_doc_coverage(fit.disease.unique())

test_shortlists, test_top_sim = a3.build_shortlist_ranking(
    fit, test, HYPERPARAMETERS, depth=shortlist_k
)
results.assert_reproduces_arm1_test(test_shortlists)

gold_in_shortlist = sum(g in s for g, s in zip(test["disease"], test_shortlists)) / len(test)
print(f"shortlist depth {len(test_shortlists[0])} | gold present in {gold_in_shortlist:.3f} "
      f"of shortlists (Arm 3's ceiling)")
""",
        tags=[GPU],
    )

    code(
        """
set_seed()
start = time.perf_counter()

arm3_test, arm3_test_metrics, _ = a3.run_condition(
    fit, test, HYPERPARAMETERS,
    mode=HYPERPARAMETERS.prompt_mode, pipe=arm3_pipe, examples=examples,
    model_name=HYPERPARAMETERS.arm3_model_name,
    shortlist_rankings=test_shortlists, top_sim=test_top_sim,
)
arm3_test.to_csv(ARTEFACTS_DIR / "arm3_test_predictions.csv", index=False)

print(f"ran in {time.perf_counter() - start:.1f}s")
for key, value in arm3_test_metrics.items():
    print(f"  {key}: {value}")

# The generator held open since §4.5, released now that nothing else needs it.
del arm3_model_obj, arm3_pipe
torch.cuda.empty_cache()
print(f"\\ngenerator released | GPU allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
""",
        tags=[GPU],
    )

    md(
        """
## §5.5 Headline test results

*Report §3.2, Figure 9.* Accuracy is the headline metric the brief specifies, with a percentile
bootstrap 95% CI over resampled items. Top-5, macro-F1 and MRR ride along for the error analysis —
they explain *why* accuracy is capped and are not promoted to headline status.
"""
    )

    code(
        """
frames = results.load_available_arms()
missing = [arm for arm in results.TEST_PREDICTION_FILES if arm not in frames]
print("available:", list(frames))
if missing:
    print(f"MISSING: {missing} -- sections below report only the arms present.")

test_gold = results.assert_query_aligned(frames)
print(f"query-aligned on {len(test_gold)} test items")

test_scores = results.score_arms(frames)
test_scores.to_csv(ARTEFACTS_DIR / "arm_test_comparison.csv", index=False)
test_scores
"""
    )

    md(
        """
## §5.6 Pairwise McNemar, and validation → test optimism

Every arm answers the same 200 questions, so the comparisons are paired and the items two arms
agree on carry no evidence about which is better. That is why a difference of two accuracies is
never judged against a single-proportion SE here.

The optimism table pairs each arm's validation accuracy (§4.6) against its test accuracy. It is
measured **per arm** rather than assumed uniform — a retrieval arm and a fine-tuned encoder need
not be optimistic by the same amount, and if they are not, that is itself a result. This is a
diagnostic of the hold-out protocol computed after the test run; it selects nothing.
"""
    )

    code(
        """
test_mcnemar = results.pairwise_mcnemar(frames)
test_mcnemar.to_csv(ARTEFACTS_DIR / "arm_test_mcnemar.csv", index=False)

for row in test_mcnemar.itertuples(index=False):
    verdict = "RESOLVED" if row.p_value < 0.05 else "UNRESOLVED"
    print(f"{row.system_a:24s} vs {row.system_b:24s} p = {row.p_value:.4g}  -> {verdict}")
test_mcnemar
"""
    )

    code(
        """
gap = results.validation_test_gap(test_scores)
gap.to_csv(ARTEFACTS_DIR / "validation_test_gap.csv", index=False)
gap
"""
    )

    md(
        """
## §5.7 Error analysis

*Report §3.3.*

### §5.7a Most frequent confusions

`same_family` flags whether gold and prediction share a label prefix; `family_error_possible`
flags whether that row's gold label had **any** sibling in the 906-label space to be confused
with. Both are needed — a `same_family` column of all-False means nothing on its own, because it
cannot be told apart from a set of gold labels that had no siblings in the first place.
"""
    )

    code(
        """
label_space = pd.read_csv(PROJECT_ROOT / "data" / "patient_qa_classification_train.csv")["disease"].unique()

for arm, frame in frames.items():
    pairs = results.confusion_pairs(frame, top_n=10, label_space=label_space)
    print()
    print(f"=== {results.ARM_LABELS.get(arm, arm)} - top confusions ===")
    print(pairs.to_string(index=False))
    pairs.to_csv(ARTEFACTS_DIR / f"{arm}_test_confusions.csv", index=False)
"""
    )

    md(
        """
### §5.7b Family-internal error share — conditioned on being possible

Is the residual error *fine-grained* (confusing two conditions in the same family) or *coarse*
(missing the topic entirely)? An unconditioned share cannot answer it: a gold label whose prefix
family has only one member **cannot** be confused with a sibling, so it contributes a guaranteed
zero to the numerator while still inflating the denominator.

That distinction decides the reading. On validation 88 of 200 items have a gold label in a
multi-member family; on test only 18 do. `family_error_summary` reports the conditioned
denominator alongside the count and returns `None` rather than `0.0` when nothing was possible.
**Read `n_family_error_possible` before the share** — on test it is too small to support a rate,
which is why the report quotes this metric on validation only.
"""
    )

    code(
        """
family_errors = pd.DataFrame(
    [{"arm": arm, **results.family_error_summary(frame, label_space)} for arm, frame in frames.items()]
)
family_errors.to_csv(ARTEFACTS_DIR / "test_family_error_share.csv", index=False)

val_family_errors = pd.DataFrame(
    [{"arm": arm, **results.family_error_summary(frame, label_space)} for arm, frame in val_arms.items()]
)
val_family_errors.to_csv(ARTEFACTS_DIR / "val_family_error_share.csv", index=False)

print("TEST - denominator too small to support a rate:")
print(family_errors.to_string(index=False))
print()
print("VALIDATION - denominator supports a rate:")
print(val_family_errors.to_string(index=False))
""",
        tags=[GPU],
    )

    md(
        """
### §5.7c What the LLM did to the shortlist, on test

The §4.5a decomposition repeated on test. Read the strata, not the headline: the fallback stratum
is inert by construction (it returns Arm 1's top-1), so any net loss has to come from the items
the LLM actively re-ranked.
""",
        tags=[GPU],
    )

    code(
        """
if "arm3_llm_rerank" in frames:
    decomposition = results.rerank_decomposition(frames["arm3_llm_rerank"])
    decomposition.to_csv(ARTEFACTS_DIR / "arm3_test_decomposition.csv", index=False)
    display(decomposition)
else:
    print("Arm 3 test predictions absent.")
""",
        tags=[GPU],
    )

    md(
        """
### §5.7d Worked examples

The brief asks for examples of correct and incorrect predictions per method. Sampled with the
project seed so the report's examples are stable across reruns.
"""
    )

    code(
        """
worked = results.worked_examples(frames, n=4)
worked.to_csv(ARTEFACTS_DIR / "test_worked_examples.csv", index=False)
worked
"""
    )

    md(
        """
### §5.7e Figure 8 — confusion pairs

*Report Figure 8.* One panel per arm. A bar is annotated only where the gold label actually had a
sibling to be confused with, so the reader is not invited to infer a fine-grained failure from a
pair that could never have been one.
"""
    )

    code(
        """
fig, axes = plt.subplots(1, len(frames), figsize=(4.8 * len(frames), 3.8))
axes = axes if len(frames) > 1 else [axes]

for ax, (arm, frame) in zip(axes, frames.items()):
    pairs = results.confusion_pairs(frame, top_n=8, label_space=label_space)
    labels = [f"{g[:24]} -> {p[:24]}" for g, p in zip(pairs["gold"], pairs["pred"])]
    colours = ["#a8422f" if poss else "#4c72b0" for poss in pairs["family_error_possible"]]
    y = range(len(pairs))
    ax.barh(list(y), pairs["n"], color=colours, height=0.6)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("errors")
    ax.set_title(results.ARM_LABELS.get(arm, arm), fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.get_major_locator().set_params(integer=True)

fig.suptitle("Most frequent test-set confusions, one panel per arm", fontsize=10)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig8_confusions.png", dpi=200, bbox_inches="tight")
plt.show()
"""
    )

    md(
        """
### §5.7f Figure 9 — test accuracy with bootstrap CIs

*Report Figure 9.* Error bars are the percentile bootstrap 95% CIs from §5.5, not ±1 SE, so the
visual comparison matches the interval quoted in the text.
"""
    )

    code(
        """
fig, ax = plt.subplots(figsize=(7, 3.6))

order = test_scores.sort_values("accuracy", ascending=True)
y = range(len(order))
lower = order["accuracy"] - order["ci_low"]
upper = order["ci_high"] - order["accuracy"]

ax.barh(list(y), order["accuracy"], color="#4c72b0", height=0.55)
ax.errorbar(order["accuracy"], list(y), xerr=[lower, upper], fmt="none",
            ecolor="#22303f", capsize=4)
ax.set_yticks(list(y))
ax.set_yticklabels(order["label"])
ax.set_xlabel("Test accuracy (200 items, bootstrap 95% CI)")
ax.set_xlim(0, 1)
for i, value, hi in zip(y, order["accuracy"], order["ci_high"]):
    ax.text(hi + 0.025, i, f"{value:.3f}", va="center", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig9_test_accuracy.png", dpi=200)
plt.show()

mark("5. test run")
"""
    )


# --------------------------------------------------------------------------------------
# §6 — diagrams, inventory, download
# --------------------------------------------------------------------------------------

def section_6() -> None:
    md(
        """
# §6 Workflow diagrams, inventory and export

*Backs report Figures 4 and 5.*

The diagrams are drawn by `amlh.diagrams`, which reads every hyperparameter it prints from
`config.HYPERPARAMETERS` rather than from hard-coded text, so a diagram cannot silently drift out
of step with the frozen configuration.
"""
    )

    code(
        """
from amlh import diagrams

diagrams.draw_arm1_workflow()
diagrams.draw_arms23_workflow()
plt.show()
"""
    )

    md("## §6.1 Everything this notebook wrote")

    code(
        '''
artefact_names = sorted(p.name for p in ARTEFACTS_DIR.glob("*.csv"))
figure_names = sorted(p.name for p in FIGURES_DIR.glob("*.png"))

print(f"artefacts/ ({len(artefact_names)} CSVs)")
for name in artefact_names:
    print(f"  {name}")
print(f"\\nfigures/ ({len(figure_names)} PNGs)")
for name in figure_names:
    print(f"  {name}")
'''
    )

    md(
        """
## §6.2 Wall clock

Runtime of this run, by section. These are the figures quoted in the report's appendix note.
"""
    )

    code(
        """
print(f"{'section':28s} {'cumulative (min)':>18s} {'section (min)':>15s}")
previous = 0.0
for name, elapsed in SECTION_TIMES.items():
    print(f"{name:28s} {elapsed / 60:18.1f} {(elapsed - previous) / 60:15.1f}")
    previous = elapsed
print(f"\\ntotal: {(time.perf_counter() - NOTEBOOK_START) / 60:.1f} min "
      f"| RUN_FULL_SELECTION = {RUN_FULL_SELECTION}")
"""
    )

    md(
        """
## §6.3 Export

Bundles the artefacts and figures this run produced, so the outputs can be inspected outside the
runtime.
"""
    )

    code(
        """
export = Path("amlh_outputs.zip")
with zipfile.ZipFile(export, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for folder, prefix in ((ARTEFACTS_DIR, "artefacts"), (FIGURES_DIR, "figures")):
        for path in sorted(folder.glob("*")):
            if path.is_file():
                zf.write(path, arcname=f"{prefix}/{path.name}")

print(f"wrote {export.name} ({export.stat().st_size / 1e6:.1f} MB, "
      f"{len(zipfile.ZipFile(export).namelist())} members)")

if IN_COLAB:
    from google.colab import files
    files.download(str(export))
"""
    )


# --------------------------------------------------------------------------------------
# assembly and checks
# --------------------------------------------------------------------------------------

def build_notebook(cells: list[dict]) -> dict:
    # nbformat 4.5+ requires a unique id per cell. Derived from the position so a rebuild with
    # unchanged content produces an unchanged file.
    numbered = [dict(cell, id=f"cell-{i:03d}") for i, cell in enumerate(cells)]
    return {
        "cells": numbered,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": [], "toc_visible": True},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


ABSOLUTE_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/])|(?:/content/drive)|(?:/Users/)|(?:/home/)")


def check(notebook: dict) -> None:
    """Every module cell must equal its source byte for byte, and no cell may hardcode a path."""
    module_cells = [
        c for c in notebook["cells"]
        if c["cell_type"] == "code" and "".join(c["source"]).startswith("%%writefile")
    ]
    assert len(module_cells) == len(MODULES), f"{len(module_cells)} module cells, expected {len(MODULES)}"

    for (name, _), cell in zip(MODULES, module_cells):
        emitted = _module_body_from_cell("".join(cell["source"]))
        source = (SRC_DIR / f"{name}.py").read_text(encoding="utf-8").replace("\r\n", "\n")
        assert emitted.rstrip("\n") == source.rstrip("\n"), f"{name}.py does not match src/amlh/{name}.py"
    print(f"byte-equality: all {len(MODULES)} module cells match src/amlh/")

    offenders = []
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code" and "".join(cell["source"]).startswith("%%writefile"):
            continue  # module sources are checked above and contain no absolute paths
        hit = ABSOLUTE_PATH_RE.search("".join(cell["source"]))
        if hit:
            offenders.append((i, hit.group(0)))
    assert not offenders, f"absolute paths found: {offenders}"
    print("absolute-path scan: none found")

    n_code = sum(c["cell_type"] == "code" for c in notebook["cells"])
    n_md = sum(c["cell_type"] == "markdown" for c in notebook["cells"])
    print(f"cells: {len(notebook['cells'])} total ({n_code} code, {n_md} markdown)")


def cpu_slice(cells: list[dict]) -> list[dict]:
    """Cells runnable without a GPU, truncated after the Arm 1 test-run cell."""
    kept = []
    for cell in cells:
        tags = cell["metadata"].get("tags", [])
        if GPU in tags or COLAB_SETUP in tags:
            continue
        kept.append(cell)
        if CPU_SLICE_END in tags:
            break
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true",
                        help="also write a CPU-only slice for local verification")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="directory for the CPU slice (default: alongside the notebook)")
    args = parser.parse_args()

    for build in (section_0, section_1, section_2, section_3, section_4, section_5, section_6):
        build()

    notebook = build_notebook(_CELLS)
    check(notebook)

    SUBMISSION_DIR.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {OUTPUT.relative_to(PROJECT_ROOT).as_posix()} "
          f"({OUTPUT.stat().st_size / 1e3:.0f} kB)")

    if args.cpu_only:
        out_dir = args.out_dir or SUBMISSION_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        slice_nb = build_notebook(cpu_slice(_CELLS))
        path = out_dir / CPU_OUTPUT_NAME
        path.write_text(json.dumps(slice_nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {path} ({len(slice_nb['cells'])} cells, CPU verification slice)")


if __name__ == "__main__":
    sys.exit(main())
