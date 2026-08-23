"""Build `arm3_colab_inputs.zip` — everything Arm 3 needs on a Colab runtime.

The repo is cloned on Colab from GitHub, but `.gitignore` excludes `data/` and
all of `artefacts/` bar the `.gitkeep`. Every input therefore has to arrive
through this zip, and the first Arm 3 run proves what happens when one is
missed: `arm2_colab_inputs.zip` carried only the two split CSVs, so
`data/db_nhs_qa_classification/` was absent, `features.load_class_doc` returned
"" for all 906 classes, and the frozen QLAD index silently degraded to QLA.
Every shortlist in that run came from the wrong retriever.

Archive members are stored at their project-relative paths, so on Colab

    !unzip -o arm3_colab_inputs.zip -d .

from the repo root drops each file exactly where `config.py` expects it.

Usage:
    python scripts/make_arm3_colab_inputs.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amlh.config import ARTEFACTS_DIR, DATA_DIR, NHS_DOCS_DIR, PROJECT_ROOT, TRAIN_CSV  # noqa: E402

OUTPUT_ZIP = PROJECT_ROOT / "arm3_colab_inputs.zip"

# Named one by one rather than globbed, so a file that goes missing is a loud
# failure here instead of a quiet accuracy drop on the runtime.
REQUIRED_FILES: list[Path] = [
    ARTEFACTS_DIR / "split_fit.csv",  # the fit split; the only text Arm 1 indexes
    ARTEFACTS_DIR / "split_val.csv",  # the 200-item hold-out Arm 3 is scored on
    ARTEFACTS_DIR / "arm1_val_predictions.csv",  # frozen Arm 1 top-1, for the reproduction assert
    TRAIN_CSV,  # full train CSV, for the shape check at the top of the notebook
]

# The NHS class documents — the "D" in the frozen QLAD index variant.
EXPECTED_DOC_COUNT = 906


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
            f"{NHS_DOCS_DIR.relative_to(PROJECT_ROOT)}, found {len(docs)}. Arm 3's frozen QLAD "
            "shortlist cannot be reproduced without all of them."
        )
    return REQUIRED_FILES + docs


def main() -> None:
    files = collect_files()

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, arcname=str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    names = zipfile.ZipFile(OUTPUT_ZIP).namelist()
    n_docs = sum(name.startswith("data/db_nhs_qa_classification/") for name in names)
    size_mb = OUTPUT_ZIP.stat().st_size / 1e6

    print(f"wrote {OUTPUT_ZIP.name} ({size_mb:.1f} MB, {len(names)} members)")
    for path in REQUIRED_FILES:
        print(f"  {path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"  data/db_nhs_qa_classification/*.txt  ({n_docs} files)")

    assert n_docs == EXPECTED_DOC_COUNT, f"only {n_docs} documents made it into the archive"
    assert len(names) == len(REQUIRED_FILES) + EXPECTED_DOC_COUNT


if __name__ == "__main__":
    main()
