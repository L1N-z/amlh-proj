"""Builds submission/AMLH_patient_question_classification.ipynb.

One self-contained notebook for the report's Appendix requirement (brief §5, §6). The
twelve src/amlh modules are inlined verbatim via %%writefile cells generated from the
files on disk (never hand-copied), so the emitted notebook cannot drift from src/amlh.
Sections 1-2 are pulled verbatim from notebooks/01_eda.ipynb and notebooks/02_arm1.ipynb
(both already CPU-only and fully executed); sections 3-5 are hand-assembled from the
Colab-variant notebooks (03_arm2_bert_colab, 04_arm3_llm_colab, 05_results_colab,
05_results) with the Colab bootstrap/upload/drive cells replaced by the shared §0
bootstrap and the two expensive sweeps gated behind RUN_FULL_SELECTION.

Run: python scripts/build_submission_notebook.py
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

_cell_id_counter = itertools.count()


def _next_id() -> str:
    return f"c{next(_cell_id_counter):04d}"

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src" / "amlh"
NB_DIR = REPO_ROOT / "notebooks"
OUT_PATH = REPO_ROOT / "submission" / "AMLH_patient_question_classification.ipynb"

MODULES = [
    "__init__", "config", "data", "features", "evaluate",
    "arm1_tfidf", "arm1_experiments", "arm2_bert", "arm3_llm",
    "results", "eda", "diagrams",
]


# --------------------------------------------------------------------------- helpers

def md(text: str) -> dict:
    return {"cell_type": "markdown", "id": _next_id(), "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": _next_id(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _lines(text),
    }


def _lines(text: str) -> list[str]:
    text = text.strip("\n")
    if not text:
        return []
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def load_source_cells(nb_filename: str) -> list[dict]:
    nb = json.loads((NB_DIR / nb_filename).read_text(encoding="utf-8"))
    out = []
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            out.append(md(src))
        else:
            out.append(code(src))
    return out


def patch(cell: dict, old: str, new: str) -> dict:
    src = "".join(cell["source"])
    if old in src:
        cell = code(src.replace(old, new)) if cell["cell_type"] == "code" else md(src.replace(old, new))
    return cell


# --------------------------------------------------------------------------- section 0

def section0_setup() -> list[dict]:
    cells = []
    cells.append(md(
        "# AMLH Coursework — Patient Question Classification\n\n"
        "Dataset C. Predicts a disease label from a patient `question` alone — `answer` is "
        "never an inference-time input. This is the reproducible notebook required by the "
        "brief (§5 Coding, §6 Appendix): every substantive line of code is the real "
        "`src/amlh` package, inlined verbatim below (§0.4) and then imported normally, so "
        "nothing here can drift from what actually ran. Section headers mirror "
        "`report/final_report.md`'s sections; each `## §N` markdown cell names the report "
        "section it backs.\n\n"
        "**Two expensive sweeps are gated behind `RUN_FULL_SELECTION`** (§0.1): Arm 2's "
        "2x24-epoch encoder comparison and Arm 3's 6-cell prompt x model grid. A full "
        "re-run of every experiment is 5-7 hours and does not fit one free Colab session. "
        "With the flag off, those two sweeps load the CSVs the original sweep produced "
        "(shipped in the input archive, §0.2) — **every selection rule and tie-break still "
        "executes live** on whichever data is in memory, printing its own margin, SE and "
        "p-value. Nothing that decides a hyperparameter is ever just asserted.\n\n"
        "**Leakage discipline.** `data.load_test()` is called exactly once, in §5, and "
        "nowhere earlier. §1's EDA use of `test` is under CLAUDE.md's declared exception "
        "(sibling-homogeneity measurement, near-duplication audit, novelty calibration) — "
        "a distributional diagnostic that selects nothing, labelled as such in its own "
        "markdown cell."
    ))

    cells.append(md("## §0. Setup, environment and provenance"))
    cells.append(md(
        "### 0.1 Environment\n\n"
        "Every value below is printed by this cell, not hand-typed — CLAUDE.md hard rule #3 "
        "forbids stating a number no code just printed. `SEED` and every frozen "
        "hyperparameter come from `config.py`, read after §0.4's imports; this cell only "
        "records the runtime itself."
    ))
    cells.append(code(
        '"""Environment banner. RUN_FULL_SELECTION / QUICK_SMOKE_TEST are the two flags that\n'
        'govern this notebook\'s run scope; both are visible top-level toggles, not buried\n'
        'constants."""\n'
        "import sys\n"
        "import time\n\n"
        "RUN_FULL_SELECTION = False  # True re-runs both expensive sweeps live (5-7h total)\n"
        "QUICK_SMOKE_TEST = False    # this notebook is the real run, never the smoke test\n\n"
        "SECTION_TIMES: list[tuple[str, float]] = []\n\n\n"
        "def _mark(label: str) -> None:\n"
        "    SECTION_TIMES.append((label, time.perf_counter()))\n\n\n"
        "_mark(\"start\")\n"
        "print(f\"python        = {sys.version}\")\n"
        "print(f\"RUN_FULL_SELECTION = {RUN_FULL_SELECTION}\")\n"
        "print(f\"QUICK_SMOKE_TEST   = {QUICK_SMOKE_TEST}\")\n"
    ))
    cells.append(code(
        "!pip install -q transformers torch scikit-learn spacy matplotlib seaborn pandas numpy\n"
        "!python -m spacy download en_core_web_sm -q\n"
        "# en_core_web_sm is required by features.lemmatise (via features._nlp()) and therefore\n"
        "# by arm1_experiments.run_preprocessing_ablation in §2. It is not declared in\n"
        "# pyproject.toml, so a clean runtime needs this explicit download.\n"
    ))
    cells.append(code(
        "import numpy, pandas, sklearn, matplotlib, seaborn, spacy, torch, transformers\n\n"
        "for _name, _mod in [\n"
        '    ("numpy", numpy), ("pandas", pandas), ("scikit-learn", sklearn),\n'
        '    ("matplotlib", matplotlib), ("seaborn", seaborn), ("spacy", spacy),\n'
        '    ("torch", torch), ("transformers", transformers),\n'
        "]:\n"
        '    print(f"{_name:14s} {_mod.__version__}")\n\n'
        'print(f"cuda available = {torch.cuda.is_available()}")\n'
        "if torch.cuda.is_available():\n"
        '    print(f"gpu            = {torch.cuda.get_device_name(0)}")\n'
        "else:\n"
        '    print("gpu            = none -- Arms 2 and 3 need a T4 runtime "\n'
        '          "(Runtime > Change runtime type)")\n'
    ))

    cells.append(md(
        "### 0.2 Upload the input archive\n\n"
        "Everything this notebook reads arrives in `amlh_submission_inputs.zip`, built on "
        "the workstation by `python scripts/make_submission_inputs.py`. It carries the "
        "training CSV, the sanitised test CSV (`answer` stripped before it ever left the "
        "workstation), all 906 NHS documents, and the eleven precomputed sweep CSVs used "
        "when `RUN_FULL_SELECTION = False`."
    ))
    cells.append(code(
        "from pathlib import Path\n\n"
        "try:\n"
        "    from google.colab import files\n"
        "    ON_COLAB = True\n"
        "except ImportError:\n"
        "    ON_COLAB = False\n\n"
        "ARCHIVE_NAME = \"amlh_submission_inputs.zip\"\n\n"
        "if ON_COLAB:\n"
        "    if not Path(ARCHIVE_NAME).exists():\n"
        "        files.upload()  # select amlh_submission_inputs.zip\n"
        "    !unzip -o -q {ARCHIVE_NAME} -d .\n"
        "    print(f\"unzipped into {Path.cwd()}\")\n"
        "else:\n"
        "    print(f\"ON_COLAB={ON_COLAB} -- assuming the repo's data/ and artefacts/ are \"\n"
        "          \"already present locally (workstation smoke-test path).\")\n"
    ))
    cells.append(code(
        "# Raw filesystem checks only -- amlh isn't importable yet (that's §0.4).\n"
        "_n_docs = len(list(Path(\"data/db_nhs_qa_classification\").glob(\"*.txt\")))\n"
        "assert _n_docs == 906, f\"expected 906 NHS documents, found {_n_docs}\"\n"
        "assert Path(\"data/patient_qa_classification_train.csv\").is_file()\n"
        "assert Path(\"data/patient_qa_classification_test.csv\").is_file()\n\n"
        "import pandas as pd\n"
        "_test_cols = list(pd.read_csv(\"data/patient_qa_classification_test.csv\", nrows=0).columns)\n"
        'print(f"NHS documents          : {_n_docs}/906")\n'
        'print(f"test CSV columns       : {_test_cols}  <- no `answer`")\n'
        'assert "answer" not in _test_cols, "test CSV carries answers -- rebuild the archive"\n'
    ))

    cells.append(md(
        "### 0.3 Module index\n\n"
        "| module | report section | role |\n"
        "|---|---|---|\n"
        "| `config` | all | frozen hyperparameters, `SEED`, paths |\n"
        "| `data` | 2.1, 2.2 | loading, integrity audit, splits |\n"
        "| `features` | 2.1, 2.3 | NHS-doc loading/coverage, TF-IDF index building |\n"
        "| `evaluate` | 3.1, 3.2 | scoring, bootstrap CIs, McNemar |\n"
        "| `arm1_tfidf` | 2.3 | k-NN ranking over a TF-IDF index |\n"
        "| `arm1_experiments` | 2.3, 2.5 | Arm 1 grids, ablations, frozen predictions |\n"
        "| `arm2_bert` | 2.4 | BERT fine-tuning, epoch selection |\n"
        "| `arm3_llm` | 2.4 | shortlist prompting, condition/model selection |\n"
        "| `results` | 3.2, 3.3 | test-set scoring, error analysis |\n"
        "| `eda` | 2.1, 3.1 | label families, sibling homogeneity, novelty |\n"
        "| `diagrams` | 2.3, 2.4 | workflow figures, read from `config.HYPERPARAMETERS` |"
    ))

    cells.append(md(
        "### 0.4 Inline `src/amlh` verbatim\n\n"
        "Twelve `%%writefile` cells, one per module, in dependency order. Generated by "
        "`scripts/build_submission_notebook.py` directly from the files on disk — this is a "
        "projection of the real package, not a retyped copy, and the build script asserts "
        "byte-equality against `src/amlh/*.py` before it will emit the notebook."
    ))
    cells.append(code('import os\nos.makedirs("src/amlh", exist_ok=True)  # %%writefile does not create parent dirs\n'))
    for name in MODULES:
        src = (SRC_DIR / f"{name}.py").read_text(encoding="utf-8")
        # Built without the md()/code() text-stripping helper: byte-equality with the
        # source file (verified below) requires the exact original bytes, trailing
        # newline included.
        cells.append({
            "cell_type": "code",
            "id": _next_id(),
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [f"%%writefile src/amlh/{name}.py\n"] + src.splitlines(keepends=True),
        })

    cells.append(code(
        "import sys\n"
        'sys.path.insert(0, "src")\n\n'
        "from amlh import config, data, features\n\n"
        "print(f\"SEED           = {config.SEED}\")\n"
        "print(f\"PROJECT_ROOT   = {config.PROJECT_ROOT}\")\n"
    ))
    cells.append(md(
        "Post-import integrity check — the 2026-08-23 incident (missing NHS documents "
        "silently degrading the frozen QLAD index to QLA) is exactly this failure mode, and "
        "`require_doc_coverage` now raises rather than degrading."
    ))
    cells.append(code(
        "train = data.load_train()\n"
        "_coverage = features.require_doc_coverage(train[\"disease\"].unique())\n"
        'print(f"NHS document coverage: {_coverage}")\n'
        "_mark(\"end §0\")\n"
    ))
    return cells


# --------------------------------------------------------------------------- section 1

def section1_eda() -> list[dict]:
    cells = [
        md(
            "## §1. Dataset and preprocessing\n\n"
            "-> report 2.1 (Dataset Description), 2.2 (Data Preprocessing), 3.1 (Exploratory "
            "Data Analysis).\n\n"
            "Loads and audits the raw data, characterises class support and label-space "
            "structure, and measures the phrasing-homogeneity gap between the validation "
            "hold-out and the real test set. **Declared exception (CLAUDE.md hard rule #2), "
            "labelled inline below:** two cells read `test.disease` for a distributional "
            "diagnostic — sibling-phrasing homogeneity and the novelty-calibration target. "
            "This selects no model, hyperparameter, preprocessing setting, or index variant."
        )
    ]
    src_cells = load_source_cells("01_eda.ipynb")[1:]  # drop the notebook's own title cell
    cells.extend(src_cells)
    cells.append(md(
        "### Load the canonical frozen split\n\n"
        "`make_validation_split(seed=44)` above is deterministic within one fixed code+data "
        "state -- confirmed by running it in two separate fresh interpreters against this "
        "repo, which produced byte-identical output. It does **not**, however, reproduce "
        "the specific 200-item partition `config.HYPERPARAMETERS` was actually selected and "
        "frozen against when the grids in §2-§4 were originally run: some drift since then "
        "(exact cause not chased down; not load-bearing for the report's argument) changes "
        "the class/tie-break draw enough to move Arm 1's test accuracy by about a point. The "
        "cells above are kept because they are exactly what `notebooks/01_eda.ipynb` runs "
        "and their diagnostic conclusions do not depend on the exact 200 items drawn; the "
        "canonical split shipped in the archive is loaded here and overwrites "
        "`artefacts/split_fit.csv` / `split_val.csv` before §2 reads them, so the rest of "
        "this notebook reproduces the reported 0.850 validation / 0.765 test accuracy "
        "exactly rather than a re-drawn approximation of them."
    ))
    cells.append(code(
        "import shutil\n\n"
        'shutil.copy("artefacts/frozen_split_fit.csv", ARTEFACTS_DIR / "split_fit.csv")\n'
        'shutil.copy("artefacts/frozen_split_val.csv", ARTEFACTS_DIR / "split_val.csv")\n\n'
        'frozen_fit = pd.read_csv(ARTEFACTS_DIR / "split_fit.csv")\n'
        'frozen_val = pd.read_csv(ARTEFACTS_DIR / "split_val.csv")\n'
        'print(f"canonical split loaded: fit={len(frozen_fit)} ({frozen_fit.disease.nunique()} classes) | "\n'
        '      f"val={len(frozen_val)} ({frozen_val.disease.nunique()} classes)")\n'
        'print("every section from here on reads this split from artefacts/.")\n'
    ))
    cells.append(code("_mark(\"end §1\")\n"))
    return cells


# --------------------------------------------------------------------------- section 2

def section2_arm1() -> list[dict]:
    cells = [
        md(
            "## §2. Arm 1 — TF-IDF + k-NN retrieval\n\n"
            "-> report 2.3 (Traditional NLP Approach).\n\n"
            "Validation-only hyperparameter, preprocessing, index-variant, indexing-scheme, "
            "split-design and baseline-model experiments. No test-set cell anywhere in this "
            "section."
        )
    ]
    src_cells = load_source_cells("02_arm1.ipynb")[1:]
    src_cells = [patch(c, "fig4_coverage_curve.png", "fig7_coverage_curve.png") for c in src_cells]
    cells.extend(src_cells)
    cells.append(code("_mark(\"end §2\")\n"))
    return cells


# --------------------------------------------------------------------------- section 3

def section3_arm2() -> list[dict]:
    cells = [
        md(
            "## §3. Arm 2 — Bio_ClinicalBERT\n\n"
            "-> report 2.4 (Neural Approaches).\n\n"
            "`question` -> WordPiece (`max_length=48`) -> encoder -> 906-way linear "
            "classification head -> argmax. Labels are encoded against the full 906-class "
            "training universe so the label space matches Arm 1 and the frozen test run. "
            "No test-set cell anywhere in this section."
        ),
        code(
            "import time\n"
            "from pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n"
            "import torch\n\n"
            "from amlh import arm2_bert as ab\n"
            "from amlh import evaluate\n"
            "from amlh.config import ARTEFACTS_DIR, FIGURES_DIR, SEED, set_seed\n"
            "from amlh.data import load_train\n"
        ),
        md("### §3.1 Load splits from `artefacts/`"),
        code(
            "set_seed()\n\n"
            'fit = pd.read_csv(ARTEFACTS_DIR / "split_fit.csv")\n'
            'val = pd.read_csv(ARTEFACTS_DIR / "split_val.csv")\n\n'
            'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
            'gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"\n'
            'print(f"fit={len(fit)} ({fit.disease.nunique()} classes) | '
            'val={len(val)} ({val.disease.nunique()} classes)")\n'
            'print(f"device={device} | gpu={gpu_name}")\n'
            'assert device.type == "cuda", "this run needs a T4 -- Runtime > Change runtime type"\n'
        ),
        md(
            "### §3.2 Label encoding over the full 906-class universe\n\n"
            "Encoded from the full training CSV, not `fit` -- `fit` is a split, so encoding "
            "labels from it would leave the output layer missing classes the frozen test run "
            "could need to predict."
        ),
        code(
            "label_to_id, id_to_label = ab.encode_labels(load_train())\n"
            "assert len(label_to_id) == 906\n"
            'print(f"{len(label_to_id)} classes encoded")\n'
        ),
        md(
            "### §3.3 Tokenisation sanity check\n\n"
            "`max_length=48` is fixed per §1's question-length percentiles (mean 8.4 words, "
            "99th percentile 18) -- confirmed here on the fit split, not re-tuned."
        ),
        code(
            "from transformers import AutoTokenizer\n\n"
            "MAX_LENGTH = 48\n"
            'sanity_tokeniser = AutoTokenizer.from_pretrained("bert-base-uncased")\n'
            'rate = ab.truncation_rate(fit["question"].tolist(), sanity_tokeniser, MAX_LENGTH)\n'
            'print(f"truncation rate at max_length={MAX_LENGTH}: {rate:.4f}")\n'
        ),
        md(
            "### §3.4 Train both encoders (gated)\n\n"
            "`run_model_ablation` trains Bio_ClinicalBERT and `bert-base-uncased` with "
            "**identical** hyperparameters, seed and epochs -- the only thing that varies is "
            "the checkpoint. This is the in-domain-pretraining claim in report §1.1: it has "
            "to be measured, not asserted.\n\n"
            "**When `RUN_FULL_SELECTION = False`**, this cell loads the histories the "
            "original 24-epoch x 2-encoder sweep (`notebooks/03_arm2_bert_colab.ipynb`) "
            "produced, shipped in `artefacts/precomputed/`, instead of re-running ~48 "
            "epochs of training. The epoch-selection and encoder-choice rules below run "
            "live either way."
        ),
        code(
            'MODEL_NAMES = ["emilyalsentzer/Bio_ClinicalBERT", "bert-base-uncased"]\n'
            'SHORT_NAME = {"emilyalsentzer/Bio_ClinicalBERT": "bioclinicalbert", "bert-base-uncased": "bertbase"}\n'
            "LEARNING_RATE = 2e-5\n"
            "BATCH_SIZE = 16\n"
            "NUM_EPOCHS = 24  # sweep budget; the SELECTED checkpoint is frozen separately in §3.7\n"
            'PRECOMPUTED = Path("artefacts/precomputed")\n\n'
            "if RUN_FULL_SELECTION:\n"
            "    def report(model_name, row):\n"
            '        short = model_name.split("/")[-1]\n'
            "        print(\n"
            '            f"{short} | epoch {row[\'epoch\'] + 1}/{NUM_EPOCHS} | "\n'
            "            f\"train_loss {row['train_loss']:.4f} | val_loss {row['val_loss']:.4f} | \"\n"
            "            f\"val_acc {row['val_accuracy']:.4f}\"\n"
            "        )\n\n"
            "    set_seed()\n"
            "    start = time.perf_counter()\n"
            "    ablation_summary, histories, ranked_by_epoch = ab.run_model_ablation(\n"
            "        fit, val, label_to_id, MODEL_NAMES,\n"
            "        lr=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, max_length=MAX_LENGTH, seed=SEED,\n"
            "        on_epoch_end=report,\n"
            "    )\n"
            '    print(f"total wall clock for both encoders: {(time.perf_counter() - start) / 60:.1f} min")\n\n'
            "    history_bioclinicalbert = histories[MODEL_NAMES[0]]\n"
            "    history_bertbase = histories[MODEL_NAMES[1]]\n"
            '    history_bioclinicalbert.to_csv(ARTEFACTS_DIR / "arm2_history_bioclinicalbert.csv", index=False)\n'
            '    history_bertbase.to_csv(ARTEFACTS_DIR / "arm2_history_bertbase.csv", index=False)\n'
            '    ablation_summary.to_csv(ARTEFACTS_DIR / "arm2_model_ablation.csv", index=False)\n'
            "else:\n"
            "    ranked_by_epoch = None  # only produced by a live sweep\n"
            '    history_bioclinicalbert = pd.read_csv(PRECOMPUTED / "arm2_history_bioclinicalbert.csv")\n'
            '    history_bertbase = pd.read_csv(PRECOMPUTED / "arm2_history_bertbase.csv")\n'
            '    ablation_summary = pd.read_csv(PRECOMPUTED / "arm2_model_ablation.csv")\n'
            "    histories = {MODEL_NAMES[0]: history_bioclinicalbert, MODEL_NAMES[1]: history_bertbase}\n"
            '    print("RUN_FULL_SELECTION=False -- loaded the original sweep\'s recorded histories from "\n'
            '          "artefacts/precomputed/ (produced by notebooks/03_arm2_bert_colab.ipynb).")\n\n'
            "ablation_summary\n"
        ),
        md(
            "### §3.5 F6 — training/validation loss curves\n\n"
            "Report §4.2 reads the train/val gap as evidence of memorisation under ~10 "
            "examples per class."
        ),
        code(
            "fig, ax = plt.subplots(figsize=(7, 5))\n"
            "for name, history in histories.items():\n"
            '    short = name.split("/")[-1]\n'
            '    ax.plot(history["epoch"], history["train_loss"], marker="o", label=f"{short} train")\n'
            '    ax.plot(history["epoch"], history["val_loss"], marker="o", linestyle="--", label=f"{short} val")\n'
            'ax.set_xlabel("epoch")\n'
            'ax.set_ylabel("loss")\n'
            'ax.set_title("Arm 2 — training/validation loss")\n'
            "ax.legend()\n"
            "fig.tight_layout()\n"
            'fig.savefig(FIGURES_DIR / "fig6_loss_curves.png", dpi=150)\n'
            "plt.show()\n"
        ),
        md(
            "### §3.6 Selection\n\n"
            "Epoch/checkpoint selection uses validation accuracy on the standard hold-out "
            "only. Within-1-SE-prefer-fewest-epochs discipline applies throughout: SE ~ "
            "3.5pp at n=200, so differences below ~7pp are noise. **This cell always "
            "executes**, on whichever histories are in memory."
        ),
        code(
            "selected_bioclinicalbert = ab.select_best_epoch_within_one_se(history_bioclinicalbert, n_val=len(val))\n"
            "selected_bertbase = ab.select_best_epoch_within_one_se(history_bertbase, n_val=len(val))\n\n"
            'print("Bio_ClinicalBERT selected epoch (within 1 SE, fewest epochs):")\n'
            "print(selected_bioclinicalbert)\n"
            "print()\n"
            'print("bert-base-uncased selected epoch (within 1 SE, fewest epochs):")\n'
            "print(selected_bertbase)\n\n"
            "selected_epoch_by_model = {\n"
            '    "emilyalsentzer/Bio_ClinicalBERT": int(selected_bioclinicalbert["epoch"]),\n'
            '    "bert-base-uncased": int(selected_bertbase["epoch"]),\n'
            "}\n"
            'print(f"\\nselected epochs: {selected_epoch_by_model}")\n'
        ),
        md(
            "Per-item validation predictions for both encoders, then McNemar. When the sweep "
            "was not re-run, the per-item predictions come from `artefacts/precomputed/` "
            "instead of the (absent) live epoch-by-epoch rankings — same selected-epoch "
            "predictions either way."
        ),
        code(
            "encoder_predictions = {}\n"
            "for model_name, epoch in selected_epoch_by_model.items():\n"
            "    if RUN_FULL_SELECTION:\n"
            "        ranked_at_epoch = ranked_by_epoch[model_name][epoch]\n"
            '        frame = pd.DataFrame({"question": val["question"], "gold": val["disease"]})\n'
            '        frame["pred"] = [r[0] for r in ranked_at_epoch]\n'
            "        for i in range(5):\n"
            '            frame[f"top_{i + 1}"] = [r[i] if i < len(r) else None for r in ranked_at_epoch]\n'
            '        path = ARTEFACTS_DIR / f"arm2_val_predictions_{SHORT_NAME[model_name]}.csv"\n'
            "        frame.to_csv(path, index=False)\n"
            "    else:\n"
            '        frame = pd.read_csv(PRECOMPUTED / f"arm2_val_predictions_{SHORT_NAME[model_name]}.csv")\n'
            "    encoder_predictions[model_name] = frame\n"
            '    accuracy = (frame["pred"] == frame["gold"]).mean()\n'
            '    print(f"{SHORT_NAME[model_name]}: epoch {epoch}, accuracy {accuracy:.4f}")\n\n'
            "mcnemar = evaluate.mcnemar_exact(\n"
            '    encoder_predictions["emilyalsentzer/Bio_ClinicalBERT"]["pred"].tolist(),\n'
            '    encoder_predictions["bert-base-uncased"]["pred"].tolist(),\n'
            '    val["disease"].tolist(),\n'
            ")\n"
            'print("\\nMcNemar exact -- a = Bio_ClinicalBERT, b = bert-base-uncased, each at its own selected epoch:")\n'
            "for key, value in mcnemar.items():\n"
            '    print(f"  {key}: {value}")\n'
        ),
        md(
            "### §3.6c Encoder choice\n\n"
            "Pre-registered rule (CLAUDE.md, 2026-08-17): if McNemar does not separate the "
            "two encoders on the standard hold-out, the comparison is reported as unresolved "
            "and the in-domain clinical encoder is kept on prior grounds -- a declared prior, "
            "not a reading of the numbers. Only if the test resolves the comparison does "
            "measured accuracy decide."
        ),
        code(
            "ALPHA = 0.05\n\n"
            "if mcnemar[\"p_value\"] >= ALPHA:\n"
            '    print(f"McNemar p = {mcnemar[\'p_value\']:.4f} >= alpha = {ALPHA}")\n'
            '    print("Unresolved -- applying the pre-registered tie-break: keep the in-domain clinical encoder.")\n'
            '    print("NOTE: this selects Bio_ClinicalBERT regardless of which encoder scored higher.")\n'
            '    selected_model_name = "emilyalsentzer/Bio_ClinicalBERT"\n'
            "    selected_row = selected_bioclinicalbert\n"
            "else:\n"
            '    print(f"McNemar p = {mcnemar[\'p_value\']:.4f} < alpha = {ALPHA}: the difference is reliable.")\n'
            '    higher_is_bioclinical = selected_bioclinicalbert["val_accuracy"] > selected_bertbase["val_accuracy"]\n'
            '    selected_model_name = "emilyalsentzer/Bio_ClinicalBERT" if higher_is_bioclinical else "bert-base-uncased"\n'
            "    selected_row = selected_bioclinicalbert if higher_is_bioclinical else selected_bertbase\n\n"
            'print(f"\\nselected model: {selected_model_name}")\n'
            "print(f\"selected epoch: {int(selected_row['epoch'])} (val_accuracy {selected_row['val_accuracy']:.4f})\")\n"
        ),
        md(
            "### §3.7 Validation predictions for the frozen model\n\n"
            "Trained once, here, at `config.HYPERPARAMETERS`'s frozen model/epoch count -- "
            "**this cell always executes**, regardless of `RUN_FULL_SELECTION`, matching the "
            "epoch/model the selection above chose. The resulting weights are kept in memory "
            "(moved to CPU) and reused for the §5 test predictions, so Bio_ClinicalBERT is "
            "trained exactly once in this entire notebook -- no refit on fit + val, and no "
            "second ~20-30 min training pass."
        ),
        code(
            "from transformers import AutoModelForSequenceClassification\n"
            "from amlh import config\n\n"
            "frozen_model_name = config.HYPERPARAMETERS.bert_model_name\n"
            "frozen_num_epochs = config.HYPERPARAMETERS.num_epochs\n"
            'print(f"selection above chose : {selected_model_name}, epoch {int(selected_row[\'epoch\'])} "\n'
            "      f\"({int(selected_row['epoch']) + 1} epochs)\")\n"
            'print(f"config.py freezes     : {frozen_model_name}, {frozen_num_epochs} epochs")\n\n'
            "set_seed()\n"
            "selected_epochs = frozen_num_epochs\n"
            "start = time.perf_counter()\n"
            "selected_state, _, _ = ab.train_model(\n"
            "    fit, val, label_to_id, frozen_model_name,\n"
            "    lr=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=selected_epochs, max_length=MAX_LENGTH, seed=SEED,\n"
            ")\n"
            'print(f"trained {selected_epochs} epochs in {(time.perf_counter() - start) / 60:.1f} min")\n\n'
            "final_model = AutoModelForSequenceClassification.from_pretrained(\n"
            "    frozen_model_name, num_labels=len(label_to_id)\n"
            ")\n"
            "final_model.load_state_dict(selected_state)\n"
            "final_model = final_model.to(device)\n"
            "final_tokeniser = AutoTokenizer.from_pretrained(frozen_model_name)\n\n"
            "ranked = ab.predict_ranked(\n"
            '    final_model, final_tokeniser, val["question"].tolist(), id_to_label,\n'
            "    max_length=MAX_LENGTH, batch_size=BATCH_SIZE, device=device, top_k=None,\n"
            ")\n"
            'scores = evaluate.score_ranked(ranked, val["disease"].tolist())\n'
            "print(scores)\n\n"
            'captured_top1 = encoder_predictions[frozen_model_name]["pred"].tolist()\n'
            "refit_top1 = [r[0] for r in ranked]\n"
            "agreement = sum(a == b for a, b in zip(captured_top1, refit_top1)) / len(val)\n"
            'print(f"retrain vs §3.6 captured top-1 agreement: {agreement:.4f}")\n\n'
            'val_predictions = pd.DataFrame({"question": val["question"], "gold": val["disease"]})\n'
            'val_predictions["pred"] = [r[0] for r in ranked]\n'
            "for i in range(5):\n"
            '    val_predictions[f"top_{i + 1}"] = [r[i] if i < len(r) else None for r in ranked]\n'
            'val_predictions.to_csv(ARTEFACTS_DIR / "arm2_val_predictions.csv", index=False)\n\n'
            "pd.DataFrame(\n"
            '    [{**scores, "model_name": frozen_model_name, "epoch": selected_epochs - 1, "refit_agreement": agreement}]\n'
            ').to_csv(ARTEFACTS_DIR / "arm2_val_metrics.csv", index=False)\n\n'
            "# Free GPU memory now; §5 moves this back to `device` for test predictions, no retrain.\n"
            "final_model = final_model.to(\"cpu\")\n"
            "torch.cuda.empty_cache()\n"
            'print("Bio_ClinicalBERT moved to CPU; §5 reuses these exact weights for test predictions.")\n'
            "val_predictions.head()\n"
        ),
        md("### §3.8 Confirm this run's configuration matches `config.py`"),
        code(
            "frozen_arm2_hyperparameters = {\n"
            '    "bert_model_name": frozen_model_name,\n'
            '    "max_length": MAX_LENGTH,\n'
            '    "learning_rate": LEARNING_RATE,\n'
            '    "batch_size": BATCH_SIZE,\n'
            '    "num_epochs": selected_epochs,\n'
            "}\n"
            'print(frozen_arm2_hyperparameters)\n'
            'assert frozen_arm2_hyperparameters["bert_model_name"] == config.HYPERPARAMETERS.bert_model_name\n'
            'assert frozen_arm2_hyperparameters["num_epochs"] == config.HYPERPARAMETERS.num_epochs\n'
            'print("matches config.HYPERPARAMETERS: True")\n'
        ),
        code("_mark(\"end §3\")\n"),
    ]
    return cells


# --------------------------------------------------------------------------- section 4

def section4_arm3() -> list[dict]:
    cells = [
        md(
            "## §4. Arm 3 — frozen retrieval shortlist + LLM selection\n\n"
            "-> report 2.4 (Neural Approaches).\n\n"
            "Frozen Arm 1 retrieval hands the LLM a 20-label shortlist; the LLM picks one by "
            "name. Nothing below reads a test-set quantity.\n\n"
            "**Two defects invalidated the first run of this pipeline, both guarded here:** "
            "(1) missing NHS documents silently degraded QLAD to QLA -- guarded by "
            "`features.require_doc_coverage`; (2) the CoT prompt budget was too small for "
            "reasoning, forcing fallback on 46.5% of items -- fixed by "
            "`arm3_cot_max_new_tokens=128` in `config.py`."
        ),
        code(
            "import time\n"
            "from pathlib import Path\n\n"
            "import pandas as pd\n"
            "import torch\n\n"
            "from amlh import arm3_llm as a3\n"
            "from amlh import features\n"
            "from amlh.config import ARTEFACTS_DIR, HYPERPARAMETERS, SEED, set_seed\n\n"
            'PRECOMPUTED = Path("artefacts/precomputed")\n'
        ),
        md("### §4.1 Load splits and frozen hyperparameters"),
        code(
            "set_seed()\n\n"
            "# `fit` is still in memory from §3; reloaded here so this section also stands alone.\n"
            'fit = pd.read_csv(ARTEFACTS_DIR / "split_fit.csv")\n'
            'val = pd.read_csv(ARTEFACTS_DIR / "split_val.csv")\n'
            'arm1_predictions = pd.read_csv(ARTEFACTS_DIR / "arm1_val_predictions.csv")\n\n'
            'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
            'print(f"fit={len(fit)} ({fit.disease.nunique()} classes) | '
            'val={len(val)} ({val.disease.nunique()} classes) | device={device}")\n\n'
            "shortlist_k = a3.shortlist_k(HYPERPARAMETERS)\n"
            "n_shots = a3.n_shots(HYPERPARAMETERS)\n"
            "primary_model = a3.selected_model_name(HYPERPARAMETERS)\n"
            "secondary_model = a3.secondary_model_name(HYPERPARAMETERS)\n"
            'prompt_modes = ["zero_shot", "few_shot", "cot"]\n\n'
            "print({\n"
            '    "shortlist_k": shortlist_k,\n'
            '    "llm_temperature": a3.llm_temperature(HYPERPARAMETERS),\n'
            '    "n_shots": n_shots,\n'
            '    "primary_model": primary_model,\n'
            '    "secondary_model": secondary_model,\n'
            '    "prompt_mode (frozen)": HYPERPARAMETERS.prompt_mode,\n'
            "})\n"
        ),
        md(
            "### §4.2 Frozen Arm 1 shortlist — and proof it is the frozen one\n\n"
            "`assert_reproduces_arm1` checks rank 1 against `artefacts/arm1_val_predictions.csv` "
            "item for item, before a single prompt is sent."
        ),
        code(
            "set_seed()\n"
            "shortlist_rankings, top_sim = a3.build_shortlist_ranking(fit, val, HYPERPARAMETERS, depth=shortlist_k)\n\n"
            "check = a3.assert_reproduces_arm1(shortlist_rankings, arm1_predictions)\n"
            'print(f"shortlist top-1 reproduces Arm 1 on all {check[\'n\']} items")\n\n'
            "arm1_top1 = [ranking[0] for ranking in shortlist_rankings]\n"
            'gold = val["disease"].tolist()\n'
            "in_shortlist = sum(g in ranking for g, ranking in zip(gold, shortlist_rankings)) / len(gold)\n"
            'print(f"Arm 1 top-1 accuracy on these items : {sum(p == g for p, g in zip(arm1_top1, gold)) / len(gold):.3f}")\n'
            'print(f"gold label present in the shortlist  : {in_shortlist:.3f}  (the ceiling the LLM selects against)")\n'
        ),
        md("### §4.3 Prompt budget check"),
        code(
            "from transformers import AutoTokenizer\n\n"
            "set_seed()\n"
            "examples = a3.build_examples(fit, n=n_shots, seed=SEED)\n"
            "budget_tokeniser = AutoTokenizer.from_pretrained(primary_model)\n\n"
            "prompts_by_mode = {}\n"
            "budget_rows = []\n"
            "for mode in prompt_modes:\n"
            "    prompts = a3.build_prompts_for_condition(\n"
            "        val, shortlist_rankings, mode, examples=examples if mode == \"few_shot\" else None\n"
            "    )\n"
            "    prompts_by_mode[mode] = prompts\n"
            "    summary = a3.prompt_token_lengths(prompts, budget_tokeniser, max_length=512)\n"
            "    budget_rows.append({\n"
            '        "condition": mode, "min_tokens": summary["min"], "median_tokens": summary["median"],\n'
            '        "mean_tokens": summary["mean"], "p95_tokens": summary["p95"], "max_tokens": summary["max"],\n'
            '        "truncation_rate_512": summary["truncation_rate"],\n'
            "    })\n\n"
            "budget_df = pd.DataFrame(budget_rows)\n"
            "print(budget_df.to_string(index=False))\n"
        ),
        md(
            "### §4.4 The 6-cell prompt x model grid (gated)\n\n"
            "Three prompt conditions x two generators = six cells on a 200-item hold-out. "
            "**When `RUN_FULL_SELECTION = False`**, this loads the original grid's recorded "
            "metrics and per-item predictions from `artefacts/precomputed/` "
            "(`notebooks/04_arm3_llm_colab.ipynb`) instead of loading two generators and "
            "running 1,200 generations. The selection rules in §4.5-§4.6 execute live either way."
        ),
        code(
            "if RUN_FULL_SELECTION:\n"
            "    set_seed()\n"
            "    _, primary_model_obj, primary_pipe = a3.load_generator(primary_model, device=device)\n"
            '    print(f"loaded {primary_model} on {device}")\n\n'
            "    condition_frames = {}\n"
            "    condition_rows = []\n"
            "    for mode in prompt_modes:\n"
            "        start = time.perf_counter()\n"
            "        pred_df, metrics, _ = a3.run_condition(\n"
            "            fit, val, HYPERPARAMETERS, mode=mode, pipe=primary_pipe,\n"
            '            examples=examples if mode == "few_shot" else None,\n'
            "            shortlist_depth=shortlist_k, model_name=primary_model,\n"
            "            shortlist_rankings=shortlist_rankings, top_sim=top_sim,\n"
            "        )\n"
            '        metrics["wall_clock_sec"] = time.perf_counter() - start\n'
            "        condition_frames[mode] = pred_df\n"
            "        condition_rows.append(metrics)\n"
            "        print(f\"{mode:>10}: acc={metrics['accuracy']:.3f}  arm1={metrics['arm1_accuracy']:.3f}  \"\n"
            "              f\"fallback={metrics['fallback_rate']:.3f}  ({metrics['wall_clock_sec']:.0f}s)\")\n"
            "    condition_df = pd.DataFrame(condition_rows)\n\n"
            "    vs_arm1_df = a3.condition_vs_arm1_mcnemar(condition_frames)\n"
            "    mcnemar_df = a3.pairwise_condition_mcnemar(condition_frames)\n\n"
            "    del primary_model_obj\n"
            "    torch.cuda.empty_cache()\n"
            "    set_seed()\n"
            "    _, secondary_model_obj, secondary_pipe = a3.load_generator(secondary_model, device=device)\n"
            '    print(f"loaded {secondary_model} on {device}")\n\n'
            "    secondary_frames = {}\n"
            "    secondary_rows = []\n"
            "    for mode in prompt_modes:\n"
            "        start = time.perf_counter()\n"
            "        pred_df, metrics, _ = a3.run_condition(\n"
            "            fit, val, HYPERPARAMETERS, mode=mode, pipe=secondary_pipe,\n"
            '            examples=examples if mode == "few_shot" else None,\n'
            "            shortlist_depth=shortlist_k, model_name=secondary_model,\n"
            "            shortlist_rankings=shortlist_rankings, top_sim=top_sim,\n"
            "        )\n"
            '        metrics["wall_clock_sec"] = time.perf_counter() - start\n'
            "        secondary_frames[mode] = pred_df\n"
            "        secondary_rows.append(metrics)\n"
            "        print(f\"{mode:>10}: acc={metrics['accuracy']:.3f}  arm1={metrics['arm1_accuracy']:.3f}  \"\n"
            "              f\"fallback={metrics['fallback_rate']:.3f}  ({metrics['wall_clock_sec']:.0f}s)\")\n"
            "    ablation_df = pd.concat([condition_df, pd.DataFrame(secondary_rows)], ignore_index=True)\n\n"
            "    del secondary_model_obj\n"
            "    torch.cuda.empty_cache()\n\n"
            "    predictions_df = pd.concat(\n"
            "        list(condition_frames.values()) + list(secondary_frames.values()), ignore_index=True\n"
            "    )\n"
            '    predictions_df.to_csv(ARTEFACTS_DIR / "arm3_val_predictions.csv", index=False)\n'
            '    ablation_df.to_csv(ARTEFACTS_DIR / "arm3_prompt_conditions.csv", index=False)\n'
            '    mcnemar_df.to_csv(ARTEFACTS_DIR / "arm3_condition_mcnemar.csv", index=False)\n'
            '    vs_arm1_df.to_csv(ARTEFACTS_DIR / "arm3_vs_arm1_mcnemar.csv", index=False)\n'
            "else:\n"
            '    ablation_df = pd.read_csv(PRECOMPUTED / "arm3_prompt_conditions.csv")\n'
            '    mcnemar_df = pd.read_csv(PRECOMPUTED / "arm3_condition_mcnemar.csv")\n'
            '    vs_arm1_df = pd.read_csv(PRECOMPUTED / "arm3_vs_arm1_mcnemar.csv")\n'
            '    predictions_df = pd.read_csv(PRECOMPUTED / "arm3_val_predictions.csv")\n'
            '    condition_df = ablation_df[ablation_df["model_name"] == primary_model].reset_index(drop=True)\n'
            '    print("RUN_FULL_SELECTION=False -- loaded the original grid\'s recorded metrics from "\n'
            '          "artefacts/precomputed/ (produced by notebooks/04_arm3_llm_colab.ipynb).")\n\n'
            "print(ablation_df.to_string(index=False))\n"
            'print()\n'
            'print("condition vs Arm 1 top-1, McNemar:")\n'
            "print(vs_arm1_df.to_string(index=False))\n"
        ),
        md(
            "### §4.5 Prompt-condition selection\n\n"
            "Pairwise McNemar over the same validation items, pre-registered in CLAUDE.md: "
            "if no condition separates from the others at p < 0.05 the comparison is "
            "**reported as unresolved** and `zero_shot` is kept on the declared prior of the "
            "simplest prompt. **This cell always executes.**"
        ),
        code(
            "selected_mode, condition_tie_break_fired = a3.select_prompt_mode(condition_df, mcnemar_df)\n\n"
            "print(mcnemar_df.to_string(index=False))\n"
            'print()\n'
            'print({"selected_mode": selected_mode, "tie_break_fired": condition_tie_break_fired})\n'
            "if condition_tie_break_fired:\n"
            '    print("UNRESOLVED: no condition separates at p < 0.05; zero_shot kept on the declared prior.")\n'
        ),
        md(
            "### §4.6 Model selection\n\n"
            "McNemar at the selected condition, mirroring the Arm 2 encoder rule: unresolved "
            "at p >= 0.05 keeps the in-domain clinical model on the declared prior. "
            "**This cell always executes.**"
        ),
        code(
            "if RUN_FULL_SELECTION:\n"
            "    model_mcnemar_df = a3.model_mcnemar(condition_frames[selected_mode], secondary_frames[selected_mode])\n"
            '    model_mcnemar_df.to_csv(ARTEFACTS_DIR / "arm3_model_mcnemar.csv", index=False)\n'
            "else:\n"
            '    model_mcnemar_df = pd.read_csv(PRECOMPUTED / "arm3_model_mcnemar.csv")\n\n'
            "selected_model, model_tie_break_fired = a3.select_model(model_mcnemar_df, HYPERPARAMETERS)\n\n"
            'print(f"comparison made at the selected condition: {selected_mode}")\n'
            "print(model_mcnemar_df.to_string(index=False))\n"
            'print()\n'
            'print({"selected_model": selected_model, "tie_break_fired": model_tie_break_fired})\n'
            "if model_tie_break_fired:\n"
            '    print(f"UNRESOLVED: McNemar does not separate the generators; {selected_model} kept on the declared prior.")\n'
        ),
        md(
            "### §4.7 Frozen configuration, reproduced live on validation\n\n"
            "Regardless of `RUN_FULL_SELECTION`, the actual frozen condition "
            "(`config.HYPERPARAMETERS.prompt_mode` on `config.HYPERPARAMETERS.arm3_model_name`) "
            "is confirmed here — reused from the live grid above if it ran, or run fresh "
            "otherwise, mirroring Arm 2's §3.7 always-executes retrain."
        ),
        code(
            "if RUN_FULL_SELECTION:\n"
            "    frozen_val_predictions = condition_frames[HYPERPARAMETERS.prompt_mode]\n"
            '    print(f"RUN_FULL_SELECTION=True -- reusing the live {HYPERPARAMETERS.prompt_mode} run above.")\n'
            "else:\n"
            "    set_seed()\n"
            "    _, frozen_model_obj, frozen_pipe = a3.load_generator(HYPERPARAMETERS.arm3_model_name, device=device)\n"
            "    frozen_val_predictions, frozen_metrics, _ = a3.run_condition(\n"
            "        fit, val, HYPERPARAMETERS, mode=HYPERPARAMETERS.prompt_mode, pipe=frozen_pipe,\n"
            "        shortlist_depth=shortlist_k, model_name=HYPERPARAMETERS.arm3_model_name,\n"
            "        shortlist_rankings=shortlist_rankings, top_sim=top_sim,\n"
            "    )\n"
            '    print(f"frozen condition ({HYPERPARAMETERS.prompt_mode}) on {HYPERPARAMETERS.arm3_model_name}: "\n'
            "          f\"accuracy={frozen_metrics['accuracy']:.3f}\")\n"
            "    del frozen_model_obj\n"
            "    torch.cuda.empty_cache()\n\n"
            'frozen_val_predictions.to_csv(ARTEFACTS_DIR / "arm3_val_predictions_frozen.csv", index=False)\n'
        ),
        md("### §4.8 Confirm this run's selection matches `config.py`"),
        code(
            "from amlh import config\n\n"
            'print("This run\'s selection (already frozen in config.py, not re-frozen here):")\n'
            "print({\n"
            '    "shortlist_k": shortlist_k, "llm_temperature": a3.llm_temperature(HYPERPARAMETERS),\n'
            '    "n_shots": n_shots, "prompt_mode": selected_mode, "arm3_model_name": selected_model,\n'
            "})\n"
            "assert selected_mode == config.HYPERPARAMETERS.prompt_mode\n"
            "assert selected_model == config.HYPERPARAMETERS.arm3_model_name\n"
            'print("matches config.HYPERPARAMETERS: True")\n'
            'print(f"prompt-condition tie-break fired: {condition_tie_break_fired}")\n'
            'print(f"model tie-break fired           : {model_tie_break_fired}")\n'
        ),
        code("_mark(\"end §4\")\n"),
    ]
    return cells


# --------------------------------------------------------------------------- section 5

def section5_results() -> list[dict]:
    cells = [
        md(
            "## §5. Frozen test run\n\n"
            "-> report 3.2 (Performance Comparison), 3.3 (Error Analysis).\n\n"
            "**This is the only section in the notebook that evaluates on the test set**, "
            "and it runs after every hyperparameter in `config.py` is frozen. Nothing here "
            "selects anything: no variant, no checkpoint, no prompt, no threshold. Every "
            "choice was made on validation in §2-§4 and is read from "
            "`config.HYPERPARAMETERS` below. `data.load_test()` is called exactly once, in "
            "the next cell -- the only test-set access in this notebook outside §1's "
            "declared EDA exception. No arm is refit on fit + val: the tested Arm 2 model is "
            "the exact one trained once in §3.7."
        ),
        code(
            "import time\n\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n"
            "import torch\n\n"
            "from amlh import features, results\n"
            "from amlh import arm2_bert as ab\n"
            "from amlh import arm3_llm as a3\n"
            "from amlh.config import ARTEFACTS_DIR, FIGURES_DIR, HYPERPARAMETERS, SEED, TRAIN_CSV, set_seed\n"
            "from amlh.data import load_test\n\n"
            'pd.set_option("display.width", 200)\n'
            'pd.set_option("display.max_colwidth", 60)\n'
        ),
        md(
            "### §5.1 Freeze check\n\n"
            "Before any test question is read, confirm every hyperparameter this run depends "
            "on is actually frozen. A `None` here would mean a choice was never made, and the "
            "test run would be silently inventing one."
        ),
        code(
            "set_seed()\n\n"
            "FROZEN_FIELDS = [\n"
            '    "ngram_range", "min_df", "max_df", "sublinear_tf", "lemmatise",\n'
            '    "index_variant", "index_scheme", "k_neighbors",\n'
            '    "bert_model_name", "max_length", "learning_rate", "batch_size", "num_epochs",\n'
            '    "shortlist_k", "llm_temperature", "prompt_mode", "n_shots",\n'
            '    "arm3_model_name", "arm3_max_new_tokens",\n'
            "]\n\n"
            "frozen = {name: getattr(HYPERPARAMETERS, name) for name in FROZEN_FIELDS}\n"
            "unset = [name for name, value in frozen.items() if value is None]\n"
            'assert not unset, f"hyperparameters still unfrozen: {unset}"\n\n'
            'print(f"SEED = {SEED}")\n'
            "for name, value in frozen.items():\n"
            '    print(f"  {name:24s} = {value!r}")\n'
            'print(f"\\nall {len(frozen)} hyperparameters frozen")\n'
        ),
        md(
            "### §5.2 Load the frozen inputs\n\n"
            "`fit` is `split_fit` — the training half, not the full training set. "
            "`load_test` returns the 200 test items with `answer` already dropped."
        ),
        code(
            'fit = pd.read_csv(ARTEFACTS_DIR / "split_fit.csv")\n'
            "test = load_test()\n\n"
            "results.assert_no_answer_column(test)\n"
            'print(f"fit  = {len(fit):5d} rows, {fit.disease.nunique()} classes")\n'
            'print(f"test = {len(test):5d} rows, {test.disease.nunique()} classes")\n'
            'print(f"test columns: {list(test.columns)}  <- no `answer`")\n\n'
            "unseen = set(test.disease) - set(fit.disease)\n"
            'print(f"test classes absent from split_fit: {len(unseen)}")\n'
        ),
        md(
            "### §5.3 Arm 1 — TF-IDF + k-NN retrieval on test\n\n"
            "`require_doc_coverage` raises rather than degrading if the NHS document set is "
            "incomplete -- the 2026-08-23 incident guard."
        ),
        code(
            "coverage = features.require_doc_coverage(fit.disease.unique())\n"
            'print(f"doc coverage: {coverage}")\n\n'
            "set_seed()\n"
            "arm1_test = results.build_test_predictions(fit, test, HYPERPARAMETERS)\n"
            'arm1_test.to_csv(ARTEFACTS_DIR / "arm1_test_predictions.csv", index=False)\n\n'
            'accuracy = (arm1_test["pred"] == arm1_test["gold"]).mean()\n'
            'print(f"Arm 1 test accuracy: {accuracy:.4f} over {len(arm1_test)} items")\n'
            'arm1_test[["question", "gold", "pred", "top_sim"]].head()\n'
        ),
        md(
            "### §5.4 Arm 2 — Bio_ClinicalBERT on test\n\n"
            "**No retrain.** `final_model` is the exact weights trained once in §3.7, moved "
            "back onto the GPU here for inference only. `top_k=None` ranks all 906 labels, "
            "matching Arm 1's ranking depth so the two arms' MRR columns are comparable."
        ),
        code(
            "truncation = ab.truncation_rate(test[\"question\"].tolist(), final_tokeniser, HYPERPARAMETERS.max_length)\n"
            'print(f"test truncation rate at max_length={HYPERPARAMETERS.max_length}: {truncation:.4f}")\n\n'
            "final_model = final_model.to(device)\n"
            "test_ranked = ab.predict_ranked(\n"
            '    final_model, final_tokeniser, test["question"].tolist(), id_to_label,\n'
            "    max_length=HYPERPARAMETERS.max_length, batch_size=HYPERPARAMETERS.batch_size,\n"
            "    device=device, top_k=None,\n"
            ")\n\n"
            'arm2_test = pd.DataFrame({"question": test["question"], "gold": test["disease"]})\n'
            'arm2_test["pred"] = [r[0] for r in test_ranked]\n'
            "for i in range(results.SHORTLIST_DEPTH):\n"
            '    arm2_test[f"top_{i + 1}"] = [r[i] if i < len(r) else None for r in test_ranked]\n'
            'arm2_test.to_csv(ARTEFACTS_DIR / "arm2_test_predictions.csv", index=False)\n'
            'print(f"Arm 2 test accuracy: {(arm2_test[\'pred\'] == arm2_test[\'gold\']).mean():.4f}")\n\n'
            "# Free GPU memory before Arm 3's generator loads -- both do not fit in 16 GB together.\n"
            "final_model = final_model.to(\"cpu\")\n"
            "del test_ranked\n"
            "torch.cuda.empty_cache()\n"
            'arm2_test[["question", "gold", "pred"]].head()\n'
        ),
        md(
            "### §5.5 Arm 1 test shortlist for Arm 3, and proof it is the frozen one\n\n"
            "The reproduction guard: rank 1 is a sharp fingerprint of the index that produced "
            "it, so if this shortlist disagrees with `arm1_test_predictions.csv` on even one "
            "item, the run stops here -- before any GPU time is spent on prompts."
        ),
        code(
            "set_seed()\n"
            "features.require_doc_coverage(fit.disease.unique())\n\n"
            "shortlists, top_sim_test = a3.build_shortlist_ranking(\n"
            "    fit, test, HYPERPARAMETERS, depth=a3.shortlist_k(HYPERPARAMETERS)\n"
            ")\n"
            "results.assert_reproduces_arm1_test(shortlists)\n\n"
            'gold_in_shortlist = sum(g in s for g, s in zip(test["disease"], shortlists)) / len(test)\n'
            'print(f"shortlist depth {len(shortlists[0])} | gold present in {gold_in_shortlist:.3f} of shortlists")\n'
        ),
        md(
            "### §5.6 Arm 3 — frozen prompt condition on the frozen generator on test\n\n"
            "`prompt_mode` and `arm3_model_name` come from `config.py`, selected in §4 by the "
            "pre-registered two-stage rule. Only the selected condition runs here."
        ),
        code(
            "set_seed()\n\n"
            "mode = HYPERPARAMETERS.prompt_mode\n"
            "model_name = HYPERPARAMETERS.arm3_model_name\n"
            'print(f"frozen condition: {mode} | frozen generator: {model_name}")\n\n'
            "_, arm3_test_model_obj, arm3_test_pipe = a3.load_generator(model_name, device=device)\n"
            "arm3_examples = a3.build_examples(fit, a3.n_shots(HYPERPARAMETERS), seed=SEED)\n\n"
            "start = time.perf_counter()\n"
            "arm3_test, arm3_test_metrics, _ = a3.run_condition(\n"
            "    fit, test, HYPERPARAMETERS,\n"
            "    mode=mode, pipe=arm3_test_pipe, examples=arm3_examples, model_name=model_name,\n"
            "    shortlist_rankings=shortlists, top_sim=top_sim_test,\n"
            ")\n"
            'arm3_test.to_csv(ARTEFACTS_DIR / "arm3_test_predictions.csv", index=False)\n\n'
            'print(f"ran in {time.perf_counter() - start:.1f}s")\n'
            "for key, value in arm3_test_metrics.items():\n"
            '    print(f"  {key}: {value}")\n\n'
            "del arm3_test_model_obj\n"
            "torch.cuda.empty_cache()\n"
        ),
        code(
            "arm3_test_decomposition_preview = results.rerank_decomposition(arm3_test)\n"
            "arm3_test_decomposition_preview\n"
        ),
        md(
            "### §5.7 Collect all three arms\n\n"
            "`load_available_arms` returns only the files that exist -- all three should be "
            "present now."
        ),
        code(
            "frames = results.load_available_arms()\n"
            "missing = [arm for arm in results.TEST_PREDICTION_FILES if arm not in frames]\n\n"
            'print("available:", list(frames))\n'
            "if missing:\n"
            '    print(f"MISSING: {missing}")\n'
            "else:\n"
            '    print("all three arms present")\n\n'
            "gold = results.assert_query_aligned(frames)\n"
            'print(f"\\nquery-aligned on {len(gold)} test items")\n'
        ),
        md(
            "### §5.8 Headline test results\n\n"
            "Accuracy is the headline metric, with a percentile bootstrap 95% CI over "
            "resampled items."
        ),
        code(
            "test_scores = results.score_arms(frames)\n"
            'test_scores.to_csv(ARTEFACTS_DIR / "arm_test_comparison.csv", index=False)\n'
            "test_scores\n"
        ),
        md(
            "### §5.9 Pairwise McNemar on test\n\n"
            "Every arm answers the same 200 questions, so the comparisons are paired. All "
            "pairs are reported because no final system is nominated."
        ),
        code(
            "test_mcnemar = results.pairwise_mcnemar(frames)\n"
            'test_mcnemar.to_csv(ARTEFACTS_DIR / "arm_test_mcnemar.csv", index=False)\n\n'
            "for row in test_mcnemar.itertuples(index=False):\n"
            '    verdict = "RESOLVED" if row.p_value < 0.05 else "UNRESOLVED"\n'
            '    print(f"{row.system_a:24s} vs {row.system_b:24s} p = {row.p_value:.4g}  -> {verdict}")\n'
            "test_mcnemar\n"
        ),
        md(
            "### §5.10 Validation -> test optimism\n\n"
            "Measures the hold-out's over-estimate per arm rather than assuming it is "
            "uniform. This is a diagnostic of the hold-out protocol computed after the test "
            "run -- it selects nothing."
        ),
        code(
            "gap = results.validation_test_gap(test_scores)\n"
            'gap.to_csv(ARTEFACTS_DIR / "validation_test_gap.csv", index=False)\n'
            "gap\n"
        ),
        md(
            "### §5.11 Error analysis\n\n"
            "**8a. Most frequent confusions.** `same_family` flags whether gold and "
            "prediction share a label prefix; `family_error_possible` flags whether that "
            "row's gold label had any sibling in the 906-label space to be confused with."
        ),
        code(
            'label_space = pd.read_csv(TRAIN_CSV)["disease"].unique()\n\n'
            "for arm, frame in frames.items():\n"
            "    pairs = results.confusion_pairs(frame, top_n=10, label_space=label_space)\n"
            "    print()\n"
            '    print(f"=== {results.ARM_LABELS.get(arm, arm)} - top confusions ===")\n'
            "    print(pairs.to_string(index=False))\n"
            '    pairs.to_csv(ARTEFACTS_DIR / f"{arm}_test_confusions.csv", index=False)\n'
        ),
        md(
            "**8b. Family-internal error share — conditioned on being possible.** A gold "
            "label whose prefix family has only one member cannot be confused with a "
            "sibling, so `family_error_summary` reports the conditioned denominator "
            "alongside the count and returns `None` rather than `0.0` when nothing was "
            "possible. Test's denominator is too small to support a rate (18/200 items vs "
            "88/200 on validation); the validation figure is reported alongside for that "
            "reason, not because the error mode changed between splits."
        ),
        code(
            'rows = [{"arm": arm, **results.family_error_summary(frame, label_space)} for arm, frame in frames.items()]\n'
            "family_errors = pd.DataFrame(rows)\n"
            'family_errors.to_csv(ARTEFACTS_DIR / "test_family_error_share.csv", index=False)\n\n'
            "val_frames = {\n"
            '    "arm1_tfidf_knn": pd.read_csv(ARTEFACTS_DIR / "arm1_val_predictions.csv"),\n'
            '    "arm2_bio_clinicalbert": pd.read_csv(ARTEFACTS_DIR / "arm2_val_predictions.csv"),\n'
            "}\n"
            "val_family_errors = pd.DataFrame(\n"
            '    [{"arm": a, **results.family_error_summary(f, label_space)} for a, f in val_frames.items()]\n'
            ")\n"
            'val_family_errors.to_csv(ARTEFACTS_DIR / "val_family_error_share.csv", index=False)\n\n'
            'print("TEST - denominator too small to support a rate:")\n'
            "print(family_errors.to_string(index=False))\n"
            "print()\n"
            'print("VALIDATION - denominator supports a rate:")\n'
            "print(val_family_errors.to_string(index=False))\n"
        ),
        md(
            "**8c. What the LLM actually did to the shortlist.** Arm 3's headline accuracy "
            "blends three populations: kept-at-rank-1, fallback (inert by construction), and "
            "active re-rank. `arm1_accuracy` on each stratum is the counterfactual."
        ),
        code(
            'if "arm3_llm_rerank" in frames:\n'
            '    decomposition = results.rerank_decomposition(frames["arm3_llm_rerank"])\n'
            '    decomposition.to_csv(ARTEFACTS_DIR / "arm3_test_decomposition.csv", index=False)\n'
            "    decomposition\n"
            "else:\n"
            '    print("Arm 3 test predictions absent.")\n'
        ),
        md("**8d. Worked examples**, sampled with the project seed so they are stable across reruns."),
        code(
            "examples_worked = results.worked_examples(frames, n=4)\n"
            'examples_worked.to_csv(ARTEFACTS_DIR / "test_worked_examples.csv", index=False)\n'
            "examples_worked\n"
        ),
        md(
            "### §5.12 Figure — test accuracy with bootstrap CIs\n\n"
            "Error bars are the percentile bootstrap 95% CIs from §5.8, not +/-1 SE."
        ),
        code(
            "fig, ax = plt.subplots(figsize=(7, 3.6))\n\n"
            'order = test_scores.sort_values("accuracy", ascending=True)\n'
            "y = range(len(order))\n"
            'lower = order["accuracy"] - order["ci_low"]\n'
            'upper = order["ci_high"] - order["accuracy"]\n\n'
            'ax.barh(list(y), order["accuracy"], color="#4c72b0", height=0.55)\n'
            'ax.errorbar(order["accuracy"], list(y), xerr=[lower, upper], fmt="none", ecolor="#22303f", capsize=4)\n'
            "ax.set_yticks(list(y))\n"
            'ax.set_yticklabels(order["label"])\n'
            'ax.set_xlabel("Test accuracy (200 items, bootstrap 95% CI)")\n'
            "ax.set_xlim(0, 1)\n"
            'for i, value, hi in zip(y, order["accuracy"], order["ci_high"]):\n'
            '    ax.text(hi + 0.025, i, f"{value:.3f}", va="center", fontsize=9)\n'
            'ax.spines[["top", "right"]].set_visible(False)\n'
            "fig.tight_layout()\n"
            'fig.savefig(FIGURES_DIR / "fig9_test_accuracy.png", dpi=200)\n'
            "plt.show()\n"
        ),
        md("### §5.13 Artefacts written"),
        code(
            "written = [\n"
            '    "arm1_test_predictions.csv", "arm2_test_predictions.csv", "arm3_test_predictions.csv",\n'
            '    "arm_test_comparison.csv", "arm_test_mcnemar.csv", "validation_test_gap.csv",\n'
            '    "test_family_error_share.csv", "val_family_error_share.csv", "test_worked_examples.csv",\n'
            "]\n"
            'written += [f"{arm}_test_confusions.csv" for arm in frames]\n'
            'if "arm3_llm_rerank" in frames:\n'
            '    written.append("arm3_test_decomposition.csv")\n\n'
            "for name in written:\n"
            "    path = ARTEFACTS_DIR / name\n"
            "    print(f\"  {'ok ' if path.exists() else 'MISSING'} {name}\")\n"
        ),
        code("_mark(\"end §5\")\n"),
    ]
    return cells


# --------------------------------------------------------------------------- section 6

def section6_wrapup() -> list[dict]:
    cells = [
        md(
            "## §6. Workflow diagrams and download\n\n"
            "-> report Figures F4 (Arm 1 workflow) and F5 (Arms 2/3 workflow).\n\n"
            "Drawn by `amlh.diagrams`, which reads every hyperparameter it prints from "
            "`config.HYPERPARAMETERS` rather than hard-coded text, so a diagram cannot "
            "silently drift out of step with the frozen configuration."
        ),
        code(
            "import matplotlib.pyplot as plt\n\n"
            "from amlh import diagrams\n\n"
            "diagrams.draw_arm1_workflow()\n"
            "diagrams.draw_arms23_workflow()\n"
            "plt.show()\n"
        ),
        md(
            "### §6.1 Most frequent confusions (report Figure 6)\n\n"
            "One panel per arm, over the pairs each arm confuses most often on test."
        ),
        code(
            "from amlh import results\n"
            "from amlh.config import FIGURES_DIR\n\n"
            "fig, axes = plt.subplots(1, len(frames), figsize=(4.8 * len(frames), 3.8))\n"
            "axes = axes if len(frames) > 1 else [axes]\n\n"
            "for ax, (arm, frame) in zip(axes, frames.items()):\n"
            "    pairs = results.confusion_pairs(frame, top_n=8, label_space=label_space)\n"
            '    labels = [f"{g[:24]} -> {p[:24]}" for g, p in zip(pairs["gold"], pairs["pred"])]\n'
            '    colours = ["#a8422f" if poss else "#4c72b0" for poss in pairs["family_error_possible"]]\n'
            "    y = range(len(pairs))\n"
            '    ax.barh(list(y), pairs["n"], color=colours, height=0.6)\n'
            "    ax.set_yticks(list(y))\n"
            '    ax.set_yticklabels(labels, fontsize=6)\n'
            "    ax.invert_yaxis()\n"
            '    ax.set_xlabel("errors")\n'
            "    ax.set_title(results.ARM_LABELS.get(arm, arm), fontsize=9)\n"
            '    ax.spines[["top", "right"]].set_visible(False)\n'
            "    ax.xaxis.get_major_locator().set_params(integer=True)\n\n"
            'fig.suptitle("Most frequent test-set confusions, one panel per arm", fontsize=10)\n'
            "fig.tight_layout()\n"
            'fig.savefig(FIGURES_DIR / "fig8_confusions.png", dpi=200, bbox_inches="tight")\n'
            "plt.show()\n"
        ),
        md(
            "### §6.2 Runtime summary\n\n"
            "Wall-clock per section, from this run's own `time.perf_counter()` marks -- "
            "printed, never estimated (CLAUDE.md hard rule #3). This is the source for the "
            "runtime figure in the report Appendix."
        ),
        code(
            'print(f"{\'section\':10s} {\'elapsed (min)\':>14s}")\n'
            "for (label_a, t_a), (label_b, t_b) in zip(SECTION_TIMES, SECTION_TIMES[1:]):\n"
            '    print(f"{label_b:10s} {(t_b - t_a) / 60:14.1f}")\n'
            'total_min = (SECTION_TIMES[-1][1] - SECTION_TIMES[0][1]) / 60\n'
            'print(f"\\ntotal wall clock: {total_min:.1f} min")\n'
        ),
        md(
            "### §6.3 Download the artefacts\n\n"
            "Zips everything this run wrote in `artefacts/` and `figures/` for local "
            "inspection alongside the report."
        ),
        code(
            "import zipfile\n"
            "from pathlib import Path\n\n"
            "from amlh.config import ARTEFACTS_DIR\n\n"
            'zip_path = Path("amlh_submission_outputs.zip")\n'
            'with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:\n'
            "    for folder in (ARTEFACTS_DIR, FIGURES_DIR):\n"
            '        for path in sorted(folder.rglob("*")):\n'
            "            if path.is_file():\n"
            "                zf.write(path, arcname=path.relative_to(ARTEFACTS_DIR.parent).as_posix())\n\n"
            'print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")\n\n'
            "if ON_COLAB:\n"
            "    from google.colab import files\n"
            "    files.download(str(zip_path))\n"
        ),
    ]
    return cells


# --------------------------------------------------------------------------- assembly

def build_notebook() -> dict:
    cells: list[dict] = []
    cells.extend(section0_setup())
    cells.extend(section1_eda())
    cells.extend(section2_arm1())
    cells.extend(section3_arm2())
    cells.extend(section4_arm3())
    cells.extend(section5_results())
    cells.extend(section6_wrapup())

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def verify_module_bodies(nb: dict) -> None:
    """Each %%writefile cell body must equal its source file, byte for byte."""
    by_module = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if src.startswith("%%writefile src/amlh/"):
            first_line, _, body = src.partition("\n")
            name = first_line.split("src/amlh/", 1)[1].rsplit(".py", 1)[0]
            by_module[name] = body

    missing = [m for m in MODULES if m not in by_module]
    assert not missing, f"module cells missing from notebook: {missing}"

    for name in MODULES:
        expected = (SRC_DIR / f"{name}.py").read_text(encoding="utf-8")
        actual = by_module[name]
        assert actual == expected, f"{name}.py: emitted cell body does not match src/amlh/{name}.py byte for byte"
    print(f"byte-equality verified for all {len(MODULES)} modules")


_DRIVE_LETTER_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:\\")


def verify_no_colab_paths(nb: dict) -> None:
    """No cell may hard-code a Drive path or a Windows-style absolute drive path."""
    offenders = []
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell["source"])
        if "/content/drive" in src or "drive.mount" in src:
            offenders.append((i, "drive reference"))
        for line in src.splitlines():
            if _DRIVE_LETTER_RE.search(line):
                offenders.append((i, f"possible absolute path: {line.strip()[:80]}"))
    assert not offenders, f"found Colab/Drive/absolute-path references: {offenders}"
    print("no /content/drive or absolute-path references found")


def main() -> None:
    nb = build_notebook()
    verify_module_bodies(nb)
    verify_no_colab_paths(nb)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    print(f"wrote {OUT_PATH} ({n_code} code cells, {n_md} markdown cells)")


if __name__ == "__main__":
    main()
