"""Build `submission/amlh_submission_inputs.zip` — the one archive the appendix notebook asks for.

Same rationale as `make_results_colab_inputs.py`, applied to the submission notebook. That
notebook is self-contained in code — the twelve `src/amlh` modules are written out by
`%%writefile` cells — but it cannot carry the dataset, so every non-code input arrives here.

Two kinds of member:

- `data/` — the train CSV, the sanitised test CSV, and all 906 NHS documents. Without the
  documents the frozen QLAD index degrades to QLA in silence; that is the 2026-08-23 incident,
  and it is why the count is asserted here rather than trusted at runtime.
- `artefacts/precomputed/` — the recorded outputs of the two sweeps the notebook gates behind
  `RUN_FULL_SELECTION`. With the flag off, the notebook loads these and then re-runs the
  *selection rules* on them live. The sweeps are gated; the rules never are.

**The test CSV is written with `answer` stripped.** Hard rule #1 says test-side answers are never
an inference-time input; the strongest form of that guarantee is that the column never leaves this
machine, so no code on the runtime can read it even by mistake. `assert_test_csv_has_no_answer`
re-opens the written file and checks.

Archive members are stored at their project-relative paths, so on Colab

    !unzip -o amlh_submission_inputs.zip -d .

from `/content` drops each file exactly where `config.py` expects it.

Usage:
    python scripts/make_submission_inputs.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amlh.config import ARTEFACTS_DIR, NHS_DOCS_DIR, PROJECT_ROOT, TEST_CSV, TRAIN_CSV  # noqa: E402

SUBMISSION_DIR = PROJECT_ROOT / "submission"
OUTPUT_ZIP = SUBMISSION_DIR / "amlh_submission_inputs.zip"
SANITISED_TEST_NAME = "data/patient_qa_classification_test.csv"
PRECOMPUTED_PREFIX = "artefacts/precomputed"

EXPECTED_DOC_COUNT = 906

# Named one by one rather than globbed, so a file that goes missing is a loud failure here
# instead of a quiet accuracy drop on the runtime.
DATA_FILES: list[Path] = [
    TRAIN_CSV,  # the only text any arm is allowed to fit on, plus the 906-label universe
]

# The canonical splits, shipped rather than regenerated. `make_validation_split` is
# deterministic within an environment but does not reproduce these bit for bit on a current
# numpy/pandas: the algorithm is unchanged -- the committed split has exactly the structure the
# code produces, 102 classes with 98 taking two items and 4 taking one -- but the `rng.choice`
# draw differs from the environment they were made in. Regenerating them would move every
# result in the report, so the notebook loads these and reports the divergence rather than
# quietly computing different numbers.
SPLIT_FILES: list[Path] = [
    ARTEFACTS_DIR / "split_fit.csv",
    ARTEFACTS_DIR / "split_val.csv",
]

# The recorded sweeps. Each is loaded by a gated stage of the notebook, which then re-derives the
# selection from it. The comment on each line names the stage that reads it.
PRECOMPUTED_FILES: list[str] = [
    # 3.2 — the 24-epoch x 2-encoder sweep, and the epoch/encoder selection run on top of it
    "arm2_history_bioclinicalbert.csv",
    "arm2_history_bertbase.csv",
    "arm2_model_ablation.csv",
    "arm2_encoder_mcnemar.csv",
    "arm2_val_predictions_bioclinicalbert.csv",
    "arm2_val_predictions_bertbase.csv",
    "arm2_val_metrics.csv",  # 3.4 compares the in-notebook retrain against this
    # 4.3 — the 3-condition x 2-model prompt grid, and the two-stage selection run on top of it
    "arm3_prompt_budget.csv",
    "arm3_prompt_conditions.csv",
    "arm3_condition_mcnemar.csv",
    "arm3_model_mcnemar.csv",
    "arm3_vs_arm1_mcnemar.csv",
    "arm3_val_predictions.csv",
]


def sanitised_test_csv() -> str:
    """Return the test CSV as text with `answer` removed.

    Reads the raw file rather than `data.load_test` because this must also work if the
    loader's contract ever changes: the column is dropped here, explicitly, and asserted
    gone before the archive is closed.
    """
    test = pd.read_csv(TEST_CSV)
    if "question" not in test.columns or "disease" not in test.columns:
        raise ValueError(f"test CSV is missing question/disease: {list(test.columns)}")
    stripped = test.drop(columns=[c for c in test.columns if c == "answer"])
    if "answer" in stripped.columns:
        raise AssertionError("failed to strip `answer` from the test CSV")
    print(f"test CSV sanitised: {list(test.columns)} -> {list(stripped.columns)}")
    return stripped.to_csv(index=False)


def collect_data_files() -> list[Path]:
    missing = [path for path in DATA_FILES + SPLIT_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "cannot build the submission input zip, these are missing:\n  "
            + "\n  ".join(str(p.relative_to(PROJECT_ROOT)) for p in missing)
        )

    docs = sorted(NHS_DOCS_DIR.glob("*.txt"))
    if len(docs) != EXPECTED_DOC_COUNT:
        raise FileNotFoundError(
            f"expected {EXPECTED_DOC_COUNT} NHS documents under "
            f"{NHS_DOCS_DIR.relative_to(PROJECT_ROOT)}, found {len(docs)}. The frozen QLAD "
            "index cannot be reproduced without all of them."
        )
    return DATA_FILES + SPLIT_FILES + docs


def collect_precomputed() -> list[Path]:
    paths = [ARTEFACTS_DIR / name for name in PRECOMPUTED_FILES]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "the gated stages have no recorded sweep to load, these are missing:\n  "
            + "\n  ".join(str(p.relative_to(PROJECT_ROOT)) for p in missing)
        )
    return paths


def assert_test_csv_has_no_answer(zip_path: Path) -> None:
    """Re-open the finished archive and confirm the shipped test CSV carries no answers."""
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(SANITISED_TEST_NAME) as handle:
            shipped = pd.read_csv(handle)
    if "answer" in shipped.columns:
        raise AssertionError(f"{SANITISED_TEST_NAME} in the archive still has an `answer` column")
    print(f"verified in-archive: {SANITISED_TEST_NAME} columns = {list(shipped.columns)}, {len(shipped)} rows")


def main() -> None:
    data_files = collect_data_files()
    precomputed = collect_precomputed()
    test_text = sanitised_test_csv()

    SUBMISSION_DIR.mkdir(exist_ok=True)

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in data_files:
            zf.write(path, arcname=str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))
        zf.writestr(SANITISED_TEST_NAME, test_text)
        for path in precomputed:
            zf.write(path, arcname=f"{PRECOMPUTED_PREFIX}/{path.name}")

    names = zipfile.ZipFile(OUTPUT_ZIP).namelist()
    n_docs = sum(name.startswith("data/db_nhs_qa_classification/") for name in names)
    n_pre = sum(name.startswith(PRECOMPUTED_PREFIX + "/") for name in names)
    size_mb = OUTPUT_ZIP.stat().st_size / 1e6

    print(f"\nwrote {OUTPUT_ZIP.relative_to(PROJECT_ROOT).as_posix()} ({size_mb:.1f} MB, {len(names)} members)")
    for path in DATA_FILES + SPLIT_FILES:
        print(f"  {path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"  {SANITISED_TEST_NAME}  (answer column stripped)")
    print(f"  data/db_nhs_qa_classification/*.txt  ({n_docs} files)")
    print(f"  {PRECOMPUTED_PREFIX}/*.csv  ({n_pre} files)")

    assert n_docs == EXPECTED_DOC_COUNT, f"only {n_docs} documents made it into the archive"
    assert n_pre == len(PRECOMPUTED_FILES), f"only {n_pre} precomputed files made it into the archive"
    assert len(names) == (
        len(DATA_FILES) + len(SPLIT_FILES) + EXPECTED_DOC_COUNT + 1 + len(PRECOMPUTED_FILES)
    )
    assert_test_csv_has_no_answer(OUTPUT_ZIP)


if __name__ == "__main__":
    main()
