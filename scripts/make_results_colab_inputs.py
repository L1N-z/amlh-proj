"""Build `results_colab_inputs.zip` — everything the GPU half of the test run needs.

Same rationale as `make_arm3_colab_inputs.py`: the repo is cloned on Colab from GitHub,
but `.gitignore` excludes `data/` and all of `artefacts/`, so every input has to arrive
through this archive. The 2026-08-23 incident is what happens when one is missed — a zip
without the NHS documents degraded the frozen QLAD index to QLA in silence.

**This archive carries the test CSV, and it is written with `answer` stripped.** Hard rule
#1 says test-side answers are never an inference-time input; the strongest form of that
guarantee is that the column never leaves this machine, so no code on the runtime can read
it even by mistake. `assert_test_csv_has_no_answer` re-opens the written file and checks.

Archive members are stored at their project-relative paths, so on Colab

    !unzip -o results_colab_inputs.zip -d .

from the repo root drops each file exactly where `config.py` expects it.

Usage:
    python scripts/make_results_colab_inputs.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amlh.config import ARTEFACTS_DIR, NHS_DOCS_DIR, PROJECT_ROOT, TEST_CSV, TRAIN_CSV  # noqa: E402

OUTPUT_ZIP = PROJECT_ROOT / "results_colab_inputs.zip"
SANITISED_TEST_NAME = "data/patient_qa_classification_test.csv"

# Named one by one rather than globbed, so a file that goes missing is a loud failure here
# instead of a quiet accuracy drop on the runtime.
REQUIRED_FILES: list[Path] = [
    ARTEFACTS_DIR / "split_fit.csv",  # the only text either arm is allowed to fit on
    ARTEFACTS_DIR / "split_val.csv",  # not scored here; loaded for the shape/identity check
    ARTEFACTS_DIR / "arm1_test_predictions.csv",  # CPU-side Arm 1, for the reproduction assert
    TRAIN_CSV,  # full train CSV, for the label space and the shape check
]

EXPECTED_DOC_COUNT = 906


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


def collect_files() -> list[Path]:
    missing = [path for path in REQUIRED_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "cannot build the Colab input zip, these are missing:\n  "
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


def assert_test_csv_has_no_answer(zip_path: Path) -> None:
    """Re-open the finished archive and confirm the shipped test CSV carries no answers."""
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(SANITISED_TEST_NAME) as handle:
            shipped = pd.read_csv(handle)
    if "answer" in shipped.columns:
        raise AssertionError(f"{SANITISED_TEST_NAME} in the archive still has an `answer` column")
    print(f"verified in-archive: {SANITISED_TEST_NAME} columns = {list(shipped.columns)}, {len(shipped)} rows")


def main() -> None:
    files = collect_files()
    test_text = sanitised_test_csv()

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, arcname=str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))
        zf.writestr(SANITISED_TEST_NAME, test_text)

    names = zipfile.ZipFile(OUTPUT_ZIP).namelist()
    n_docs = sum(name.startswith("data/db_nhs_qa_classification/") for name in names)
    size_mb = OUTPUT_ZIP.stat().st_size / 1e6

    print(f"\nwrote {OUTPUT_ZIP.name} ({size_mb:.1f} MB, {len(names)} members)")
    for path in REQUIRED_FILES:
        print(f"  {path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"  {SANITISED_TEST_NAME}  (answer column stripped)")
    print(f"  data/db_nhs_qa_classification/*.txt  ({n_docs} files)")

    assert n_docs == EXPECTED_DOC_COUNT, f"only {n_docs} documents made it into the archive"
    assert len(names) == len(REQUIRED_FILES) + EXPECTED_DOC_COUNT + 1
    assert_test_csv_has_no_answer(OUTPUT_ZIP)


if __name__ == "__main__":
    main()
