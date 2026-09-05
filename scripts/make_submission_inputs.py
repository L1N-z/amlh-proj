"""Build `amlh_submission_inputs.zip` — everything the submission notebook needs.

`submission/AMLH_patient_question_classification.ipynb` (built by
`scripts/build_submission_notebook.py`) regenerates `artefacts/split_fit.csv` and
`split_val.csv` itself in §1, so unlike the other Colab-input archives this one does not
need to ship them. What it does need: the raw training and (sanitised) test CSVs, all 906
NHS documents, and the twelve precomputed sweep CSVs §3/§4 load when
`RUN_FULL_SELECTION = False`.

**This archive carries the test CSV, written with `answer` stripped** — the same guarantee
`make_results_colab_inputs.py` makes, reused verbatim here via `sanitised_test_csv` and
`assert_test_csv_has_no_answer` so the two archives can't drift apart on this point.

Archive members are stored at their project-relative paths, so on Colab

    !unzip -o amlh_submission_inputs.zip -d .

from the repo root drops each file exactly where `config.py` expects it.

Usage:
    python scripts/make_submission_inputs.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from make_results_colab_inputs import assert_test_csv_has_no_answer, sanitised_test_csv  # noqa: E402

from amlh.config import ARTEFACTS_DIR, NHS_DOCS_DIR, PROJECT_ROOT, TRAIN_CSV  # noqa: E402

OUTPUT_ZIP = PROJECT_ROOT / "submission" / "amlh_submission_inputs.zip"
SANITISED_TEST_NAME = "data/patient_qa_classification_test.csv"
EXPECTED_DOC_COUNT = 906

# Named one by one rather than globbed, so a file that goes missing is a loud failure here
# instead of a quiet accuracy drop on the runtime.
REQUIRED_FILES: list[Path] = [
    TRAIN_CSV,
]

# The notebook's §1 regenerates a split via make_validation_split(seed=44) for parity with
# notebooks/01_eda.ipynb, and that call IS deterministic within one fixed code+data state
# (confirmed: two fresh interpreter runs against the current repo produce byte-identical
# output). It does NOT, however, reproduce the specific 200-item partition config.py's
# hyperparameters were originally selected and frozen against -- some drift since then
# (exact cause not chased down) changes the class/tie-breaking draw, which silently pulls
# Arm 1's val/test accuracy away from the reported 0.850/0.765. So the canonical split is
# shipped too, at distinct archive names, and the notebook overwrites the freshly-generated
# split_fit.csv/split_val.csv with these right after §1 to guarantee it reproduces the
# reported numbers rather than a re-drawn approximation of them.
FROZEN_SPLIT_SOURCES: dict[Path, str] = {
    ARTEFACTS_DIR / "split_fit.csv": "artefacts/frozen_split_fit.csv",
    ARTEFACTS_DIR / "split_val.csv": "artefacts/frozen_split_val.csv",
}

# source path -> archive member name, under artefacts/precomputed/. Named explicitly against
# what's actually on disk in colab_outputs/, not globbed.
PRECOMPUTED_SOURCES: dict[Path, str] = {
    PROJECT_ROOT / "colab_outputs/arm2/artefacts/arm2_history_bioclinicalbert.csv":
        "artefacts/precomputed/arm2_history_bioclinicalbert.csv",
    PROJECT_ROOT / "colab_outputs/arm2/artefacts/arm2_history_bertbase.csv":
        "artefacts/precomputed/arm2_history_bertbase.csv",
    PROJECT_ROOT / "colab_outputs/arm2/artefacts/arm2_model_ablation.csv":
        "artefacts/precomputed/arm2_model_ablation.csv",
    PROJECT_ROOT / "colab_outputs/arm2/artefacts/arm2_encoder_mcnemar.csv":
        "artefacts/precomputed/arm2_encoder_mcnemar.csv",
    PROJECT_ROOT / "colab_outputs/arm2/artefacts/arm2_val_predictions_bioclinicalbert.csv":
        "artefacts/precomputed/arm2_val_predictions_bioclinicalbert.csv",
    PROJECT_ROOT / "colab_outputs/arm2/artefacts/arm2_val_predictions_bertbase.csv":
        "artefacts/precomputed/arm2_val_predictions_bertbase.csv",
    PROJECT_ROOT / "colab_outputs/arm3/artefacts/arm3_prompt_conditions.csv":
        "artefacts/precomputed/arm3_prompt_conditions.csv",
    PROJECT_ROOT / "colab_outputs/arm3/artefacts/arm3_condition_mcnemar.csv":
        "artefacts/precomputed/arm3_condition_mcnemar.csv",
    PROJECT_ROOT / "colab_outputs/arm3/artefacts/arm3_model_mcnemar.csv":
        "artefacts/precomputed/arm3_model_mcnemar.csv",
    PROJECT_ROOT / "colab_outputs/arm3/artefacts/arm3_vs_arm1_mcnemar.csv":
        "artefacts/precomputed/arm3_vs_arm1_mcnemar.csv",
    PROJECT_ROOT / "colab_outputs/arm3/artefacts/arm3_prompt_budget.csv":
        "artefacts/precomputed/arm3_prompt_budget.csv",
    PROJECT_ROOT / "colab_outputs/arm3/artefacts/arm3_val_predictions.csv":
        "artefacts/precomputed/arm3_val_predictions.csv",
}


def collect_files() -> list[Path]:
    missing = [path for path in REQUIRED_FILES if not path.is_file()]
    missing += [path for path in PRECOMPUTED_SOURCES if not path.is_file()]
    missing += [path for path in FROZEN_SPLIT_SOURCES if not path.is_file()]
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
            "shortlist cannot be reproduced without all of them."
        )
    return REQUIRED_FILES + docs


def main() -> None:
    files = collect_files()
    test_text = sanitised_test_csv()

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, arcname=str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))
        zf.writestr(SANITISED_TEST_NAME, test_text)
        for src_path, arcname in PRECOMPUTED_SOURCES.items():
            zf.write(src_path, arcname=arcname)
        for src_path, arcname in FROZEN_SPLIT_SOURCES.items():
            zf.write(src_path, arcname=arcname)

    names = zipfile.ZipFile(OUTPUT_ZIP).namelist()
    n_docs = sum(name.startswith("data/db_nhs_qa_classification/") for name in names)
    n_precomputed = sum(name.startswith("artefacts/precomputed/") for name in names)
    n_frozen_split = sum(name.startswith("artefacts/frozen_split_") for name in names)
    size_mb = OUTPUT_ZIP.stat().st_size / 1e6

    print(f"\nwrote {OUTPUT_ZIP.name} ({size_mb:.1f} MB, {len(names)} members)")
    for path in REQUIRED_FILES:
        print(f"  {path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"  {SANITISED_TEST_NAME}  (answer column stripped)")
    print(f"  data/db_nhs_qa_classification/*.txt  ({n_docs} files)")
    print(f"  artefacts/precomputed/*.csv  ({n_precomputed} files)")
    print(f"  artefacts/frozen_split_*.csv  ({n_frozen_split} files)")

    assert n_docs == EXPECTED_DOC_COUNT, f"only {n_docs} documents made it into the archive"
    assert n_precomputed == len(PRECOMPUTED_SOURCES), (
        f"expected {len(PRECOMPUTED_SOURCES)} precomputed CSVs, found {n_precomputed}"
    )
    assert n_frozen_split == len(FROZEN_SPLIT_SOURCES), (
        f"expected {len(FROZEN_SPLIT_SOURCES)} frozen-split CSVs, found {n_frozen_split}"
    )
    assert len(names) == (
        len(REQUIRED_FILES) + EXPECTED_DOC_COUNT + 1 + len(PRECOMPUTED_SOURCES) + len(FROZEN_SPLIT_SOURCES)
    )
    assert_test_csv_has_no_answer(OUTPUT_ZIP)


if __name__ == "__main__":
    main()
