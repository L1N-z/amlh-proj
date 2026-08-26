"""Matplotlib workflow diagrams for the report (figures F4 and F5).

The brief asks for a diagram illustrating the workflow of each method. Drawing them in
code rather than by hand keeps them in the same pipeline as every other figure: they are
version-controlled, they regenerate when the pipeline changes, and — most importantly —
every hyperparameter shown is read from `config.HYPERPARAMETERS` rather than typed in, so
a diagram cannot drift out of step with the frozen configuration the way a hand-drawn
image silently would.

Both diagrams are laid out on a unit grid in axes coordinates, so box positions read as
data rather than as magic pixel offsets.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from amlh.config import FIGURES_DIR, HYPERPARAMETERS

# Muted, print-safe, and still distinguishable in greyscale — the report may be marked on paper.
INPUT_FACE = "#e8eef5"
PROCESS_FACE = "#ffffff"
FROZEN_FACE = "#dce8dd"
OUTPUT_FACE = "#f3e6dd"
EDGE = "#33414f"
TEXT = "#16202b"
MUTED = "#5c6a78"
FLAG = "#a8422f"


def _box(ax, x, y, w, h, title, subtitle=None, face=PROCESS_FACE, fontsize=8.5):
    """One rounded node. `subtitle` carries the frozen hyperparameters, set smaller so the
    diagram reads as a flow first and a parameter table second."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            linewidth=0.9, edgecolor=EDGE, facecolor=face, zorder=2,
        )
    )
    cy = y + h / 2
    if subtitle:
        ax.text(x + w / 2, cy + h * 0.17, title, ha="center", va="center",
                fontsize=fontsize, color=TEXT, zorder=3)
        ax.text(x + w / 2, cy - h * 0.21, subtitle, ha="center", va="center",
                fontsize=fontsize - 1.7, color=MUTED, family="monospace", zorder=3)
    else:
        ax.text(x + w / 2, cy, title, ha="center", va="center",
                fontsize=fontsize, color=TEXT, zorder=3)


def _arrow(ax, start, end, linestyle="-", color=EDGE, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=11,
            linewidth=0.9, color=color, linestyle=linestyle,
            connectionstyle=f"arc3,rad={rad}", zorder=1,
        )
    )


def _canvas(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def draw_arm1_workflow(hp=HYPERPARAMETERS, save_to=None):
    """F4 — Arm 1: TF-IDF + k-NN retrieval, and the shortlist it hands to Arm 3."""
    fig, ax = _canvas((9.2, 3.6))

    ax.text(0.0, 0.99, "Figure F4 — Arm 1: TF-IDF retrieval over a QLAD class-blob index",
            ha="left", va="top", fontsize=10, color=TEXT)
    ax.text(0.0, 0.925,
            "Green = frozen in config.py. Training answers and NHS documents are indexed as "
            "class evidence; at prediction time the only input is the question.",
            ha="left", va="top", fontsize=7.2, color=MUTED)

    _box(ax, 0.01, 0.56, 0.175, 0.22, "split_fit", "8,691 questions", face=INPUT_FACE)
    _box(ax, 0.01, 0.17, 0.175, 0.22, "NHS documents", "906 x .txt", face=INPUT_FACE)

    _box(ax, 0.225, 0.36, 0.19, 0.26,
         f"Index — {hp.index_variant}", f"{hp.index_scheme}\n906 class blobs", face=FROZEN_FACE)
    _box(ax, 0.455, 0.36, 0.185, 0.26,
         "TF-IDF", f"ngram {hp.ngram_range}\nmin_df={hp.min_df}", face=FROZEN_FACE)
    _box(ax, 0.68, 0.36, 0.175, 0.26,
         "Cosine k-NN", f"k={hp.k_neighbors}", face=FROZEN_FACE)

    _box(ax, 0.885, 0.56, 0.105, 0.22, "Top-1", "prediction", face=OUTPUT_FACE)
    _box(ax, 0.885, 0.17, 0.105, 0.22, f"Top-{hp.shortlist_k}", "to Arm 3", face=OUTPUT_FACE)

    _arrow(ax, (0.185, 0.67), (0.225, 0.55))
    _arrow(ax, (0.185, 0.28), (0.225, 0.43))
    _arrow(ax, (0.415, 0.49), (0.455, 0.49))
    _arrow(ax, (0.640, 0.49), (0.680, 0.49))
    _arrow(ax, (0.855, 0.54), (0.885, 0.66))
    _arrow(ax, (0.855, 0.44), (0.885, 0.32), linestyle=(0, (3, 2)))

    _box(ax, 0.36, 0.03, 0.375, 0.09,
         "Test question — the only inference-time input", face=INPUT_FACE, fontsize=8)
    _arrow(ax, (0.548, 0.12), (0.548, 0.36))

    fig.tight_layout()
    if save_to is not False:
        path = save_to or FIGURES_DIR / "fig4_arm1_workflow.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path}")
    return fig


def draw_arms23_workflow(hp=HYPERPARAMETERS, save_to=None):
    """F5 — Arms 2 and 3 together, so the shared Arm 1 dependency is visible.

    Arm 3 is not an independent system: it re-ranks Arm 1's shortlist and returns Arm 1's
    own top-1 whenever parsing finds no shortlist label. Drawing it apart from Arm 1 would
    misrepresent what it is, and would hide why its accuracy tracks Arm 1's.
    """
    fig, ax = _canvas((9.2, 4.7))

    ax.text(0.0, 0.99, "Figure F5 — Arm 2 (fine-tuned encoder) and Arm 3 (shortlist + LLM)",
            ha="left", va="top", fontsize=10, color=TEXT)

    # --- Arm 2 lane ---
    ax.text(0.0, 0.845, "ARM 2", ha="left", va="center", fontsize=8,
            color=MUTED, family="monospace")
    _box(ax, 0.075, 0.760, 0.155, 0.17, "split_fit", "8,691 questions", face=INPUT_FACE)
    _box(ax, 0.265, 0.760, 0.185, 0.17,
         "WordPiece tokeniser", f"max_length={hp.max_length}", face=FROZEN_FACE)
    _box(ax, 0.485, 0.760, 0.215, 0.17,
         "Bio_ClinicalBERT", f"lr={hp.learning_rate:g}  bs={hp.batch_size}\n906-way head",
         face=FROZEN_FACE)
    _box(ax, 0.735, 0.760, 0.185, 0.17,
         "Fine-tune", f"epoch {hp.num_epochs} selected", face=FROZEN_FACE)
    _arrow(ax, (0.230, 0.845), (0.265, 0.845))
    _arrow(ax, (0.450, 0.845), (0.485, 0.845))
    _arrow(ax, (0.700, 0.845), (0.735, 0.845))

    # --- shared input ---
    _box(ax, 0.365, 0.550, 0.27, 0.11, "Test question", face=INPUT_FACE, fontsize=8.5)
    _arrow(ax, (0.560, 0.660), (0.827, 0.760), rad=-0.15)

    # --- Arm 3 lane ---
    ax.text(0.0, 0.400, "ARM 3", ha="left", va="center", fontsize=8,
            color=MUTED, family="monospace")
    _box(ax, 0.075, 0.315, 0.165, 0.17,
         "Arm 1 shortlist", f"frozen, top-{hp.shortlist_k}", face=FROZEN_FACE)
    _box(ax, 0.275, 0.315, 0.185, 0.17,
         f"{hp.prompt_mode} prompt",
         f"T={hp.llm_temperature:g}  new={hp.arm3_max_new_tokens}", face=FROZEN_FACE)
    _box(ax, 0.495, 0.315, 0.175, 0.17, "MediPhi-Guidelines", "generate", face=FROZEN_FACE)
    _box(ax, 0.705, 0.315, 0.155, 0.17, "Parse to a\nshortlist label")

    _arrow(ax, (0.240, 0.400), (0.275, 0.400))
    _arrow(ax, (0.460, 0.400), (0.495, 0.400))
    _arrow(ax, (0.670, 0.400), (0.705, 0.400))
    _arrow(ax, (0.440, 0.550), (0.320, 0.485), rad=0.15)

    _box(ax, 0.885, 0.550, 0.105, 0.17, "Prediction", face=OUTPUT_FACE, fontsize=8.5)
    _arrow(ax, (0.920, 0.760), (0.920, 0.722))
    _arrow(ax, (0.860, 0.400), (0.920, 0.550), rad=-0.2)

    # The fallback: a parse failure returns Arm 1's top-1 unchanged.
    _arrow(ax, (0.782, 0.315), (0.158, 0.315), linestyle=(0, (3, 2)), color=FLAG, rad=-0.22)
    ax.text(0.470, 0.168, "no shortlist label matched: return Arm 1 top-1 unchanged",
            ha="center", va="center", fontsize=7.2, color=FLAG)

    ax.text(0.0, 0.065,
            "Green = frozen in config.py. Dashed red = the fallback path. Arm 3 is not "
            "independent of Arm 1 — it re-ranks Arm 1's shortlist and returns Arm 1's own\n"
            "top-1 whenever parsing finds no candidate, which is why its accuracy is anchored "
            "to Arm 1's.",
            ha="left", va="top", fontsize=7.2, color=MUTED)

    fig.tight_layout()
    if save_to is not False:
        path = save_to or FIGURES_DIR / "fig5_arms23_workflow.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path}")
    return fig
