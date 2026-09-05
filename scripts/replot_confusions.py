"""Redraw `figures/fig8_confusions.png` from the persisted confusion artefacts.

The shipped figure carried "Figure F8 - most frequent confusions (red = a same-family
error was possible)" baked into the image. That number is an outline-era label the report
no longer uses, and the colour clause advertises a red that never appears: every
`family_error_possible` flag on the test split is False (`test_family_error_share.csv`
records `n_within_family = 0` for all three arms), so no bar is ever drawn red.

This redraws the same plot from `artefacts/<arm>_test_confusions.csv` — the frames
`05_results.ipynb` §9b already wrote — so no test-set prediction is recomputed and no
selection decision is touched. The colour rule is kept as-is; only the title changes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from amlh.config import ARTEFACTS_DIR, FIGURES_DIR
from amlh.results import ARM_LABELS

TOP_N = 8
"""Matches `05_results.ipynb` §9b's `confusion_pairs(..., top_n=8)`."""

SUPTITLE = "Most frequent test-set confusions, one panel per arm"


def main() -> None:
    arms = [a for a in ARM_LABELS if (ARTEFACTS_DIR / f"{a}_test_confusions.csv").exists()]
    if not arms:
        raise SystemExit("no *_test_confusions.csv artefacts found — run 05_results.ipynb first")

    fig, axes = plt.subplots(1, len(arms), figsize=(4.8 * len(arms), 3.8))
    axes = axes if len(arms) > 1 else [axes]

    for ax, arm in zip(axes, arms):
        pairs = pd.read_csv(ARTEFACTS_DIR / f"{arm}_test_confusions.csv").head(TOP_N)
        labels = [f"{g[:24]} -> {p[:24]}" for g, p in zip(pairs["gold"], pairs["pred"])]
        colours = ["#a8422f" if poss else "#4c72b0" for poss in pairs["family_error_possible"]]
        y = range(len(pairs))
        ax.barh(list(y), pairs["n"], color=colours, height=0.6)
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("errors")
        ax.set_title(ARM_LABELS.get(arm, arm), fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.get_major_locator().set_params(integer=True)

        n_possible = int(pairs["family_error_possible"].sum())
        print(f"  {arm}: {len(pairs)} pairs plotted, {n_possible} with a same-family error possible")

    fig.suptitle(SUPTITLE, fontsize=10)
    fig.tight_layout()
    out = FIGURES_DIR / "fig8_confusions.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
