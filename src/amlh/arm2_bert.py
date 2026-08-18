"""Fine-tuning orchestration for Arm 2 (BERT classifier).

`question` -> WordPiece -> encoder -> 906-way linear head -> argmax. Follows the
manual epoch-loop pattern from `reference/NLP3-BERT_for_document_classification-solution.ipynb`
(`transformers` + `torch` only, no `Trainer`/`accelerate`/`datasets`). Orchestration
functions here return DataFrames/tuples and never print or plot — notebooks do that.
"""

import time
from typing import Any, Callable

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from amlh import config
from amlh.evaluate import score_ranked


def encode_labels(train_df: pd.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    """label_to_id / id_to_label over the full disease universe, sorted for determinism.

    Must be called on the full training set, not a validation-time subset, so the
    output layer covers every class the frozen test run could need to predict."""
    labels = sorted(train_df["disease"].unique())
    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    return label_to_id, id_to_label


def tokenise(texts: list[str], tokeniser, max_length: int) -> dict[str, torch.Tensor]:
    """Pad/truncate to a fixed length so all batches share tensor shape."""
    return tokeniser(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def truncation_rate(texts: list[str], tokeniser, max_length: int) -> float:
    """Fraction of texts whose untruncated WordPiece length exceeds max_length."""
    lengths = [len(ids) for ids in tokeniser(texts, truncation=False)["input_ids"]]
    return sum(n > max_length for n in lengths) / len(lengths)


def make_loader(
    encodings: dict[str, torch.Tensor],
    labels: list[int],
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader:
    """num_workers=0 always (rule #7 — Windows spawn hangs a notebook otherwise).
    Shuffling is seeded via an explicit Generator rather than global RNG state, so
    each call is independently reproducible."""
    dataset = TensorDataset(
        encodings["input_ids"], encodings["attention_mask"], torch.tensor(labels)
    )
    if shuffle:
        if seed is None:
            raise ValueError("seed is required when shuffle=True")
        generator = torch.Generator().manual_seed(seed)
        sampler = RandomSampler(dataset, generator=generator)
    else:
        sampler = SequentialSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=0)


def _run_epoch(model, loader: DataLoader, optimizer, device, train: bool) -> float:
    """Shared batching/forward logic for the train and eval passes. Both pass
    `labels=` so eval-mode also yields a loss (needed to log val_loss every epoch,
    not just at the end, for report figure F6)."""
    model.train() if train else model.eval()
    total_loss = 0.0
    n = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            if train:
                model.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            n += len(labels)
    return total_loss / n


def predict_ranked(
    model,
    tokeniser,
    questions: list[str],
    id_to_label: dict[int, str],
    max_length: int,
    batch_size: int,
    device,
    top_k: int | None = 5,
) -> list[list[str]]:
    """Ranked (best-first) label predictions, order-aligned to `questions`.

    `top_k=None` ranks every label. Pass it wherever MRR is reported: a
    truncated ranking silently scores MRR@k instead, and Arm 1 ranks all
    classes, so a top-5 Arm 2 ranking would not be comparable to it.
    """
    encodings = tokenise(questions, tokeniser, max_length)
    dummy_labels = [0] * len(questions)
    loader = make_loader(encodings, dummy_labels, batch_size, shuffle=False)

    model.eval()
    ranked: list[list[str]] = []
    k = len(id_to_label) if top_k is None else min(top_k, len(id_to_label))
    with torch.no_grad():
        for input_ids, attention_mask, _ in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            top_ids = torch.topk(logits, k=k, dim=1).indices.cpu().tolist()
            ranked.extend([[id_to_label[i] for i in row] for row in top_ids])
    return ranked


def train_model(
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_to_id: dict[str, int],
    model_name: str,
    lr: float,
    batch_size: int,
    epochs: int,
    max_length: int,
    seed: int,
    model=None,
    tokeniser=None,
) -> tuple[dict, pd.DataFrame, dict[int, list[list[str]]]]:
    """Manual AdamW training loop. Returns (best_state_dict, history_df,
    ranked_by_epoch) where history_df has one row per epoch: epoch, train_loss,
    val_loss, val_accuracy. Best epoch is selected by val_accuracy (ties keep the
    earlier/fewer-epoch state).

    `ranked_by_epoch[epoch]` is the top-5 validation ranking that epoch's model
    produced — already computed to score the epoch, so keeping it is free. It
    means per-item predictions at *any* epoch can be recovered afterwards without
    retraining, which is what McNemar over the encoder ablation needs: the
    within-1-SE selected epoch is not known until training has finished.

    `model`/`tokeniser` are a test-only injection seam: pass pre-built objects to
    skip `from_pretrained` (e.g. a tiny random-init model, no network). Production
    and notebook calls omit them."""
    config.set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if tokeniser is None:
        tokeniser = AutoTokenizer.from_pretrained(model_name)
    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(label_to_id)
        )
    model = model.to(device)

    fit_labels = fit_df["disease"].map(label_to_id).tolist()
    val_labels = val_df["disease"].map(label_to_id).tolist()
    fit_encodings = tokenise(fit_df["question"].tolist(), tokeniser, max_length)
    val_encodings = tokenise(val_df["question"].tolist(), tokeniser, max_length)

    fit_loader = make_loader(fit_encodings, fit_labels, batch_size, shuffle=True, seed=seed)
    val_loader = make_loader(val_encodings, val_labels, batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=lr)

    id_to_label = {i: label for label, i in label_to_id.items()}
    val_gold = val_df["disease"].tolist()

    rows = []
    ranked_by_epoch: dict[int, list[list[str]]] = {}
    best_state = None
    best_val_accuracy = -1.0
    for epoch in range(epochs):
        train_loss = _run_epoch(model, fit_loader, optimizer, device, train=True)
        val_loss = _run_epoch(model, val_loader, optimizer, device, train=False)
        ranked = predict_ranked(
            model, tokeniser, val_df["question"].tolist(), id_to_label, max_length, batch_size, device
        )
        val_accuracy = score_ranked(ranked, val_gold)["accuracy"]
        rows.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_accuracy}
        )
        ranked_by_epoch[epoch] = ranked
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return best_state, pd.DataFrame(rows), ranked_by_epoch


def measure_run(fn: Callable, *args, **kwargs) -> tuple[Any, float, float]:
    """Wraps a call, returning (result, wall_clock_s, peak_memory_mb). peak_memory_mb
    is nan when CUDA is unavailable — never a fabricated 0 (rule #3)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    wall_clock_s = time.perf_counter() - start
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6
    else:
        peak_memory_mb = float("nan")
    return result, wall_clock_s, peak_memory_mb


def run_model_ablation(
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_to_id: dict[str, int],
    model_names: list[str],
    lr: float,
    batch_size: int,
    epochs: int,
    max_length: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, dict[int, list[list[str]]]]]:
    """Trains each encoder in `model_names` with identical hyperparameters via
    `train_model` (no duplicated training logic), re-seeding before each run.
    Returns a summary row per encoder, each encoder's full history_df, and each
    encoder's per-epoch validation rankings — the last so the encoders can be
    compared per-item (McNemar) at whichever epoch selection later picks, without
    retraining either of them."""
    summary_rows = []
    histories = {}
    ranked_by_epoch_by_model = {}
    for model_name in model_names:
        config.set_seed(seed)
        (best_state, history_df, ranked_by_epoch), wall_clock_s, peak_memory_mb = measure_run(
            train_model,
            fit_df,
            val_df,
            label_to_id,
            model_name,
            lr,
            batch_size,
            epochs,
            max_length,
            seed,
        )
        best_row = history_df.loc[history_df["val_accuracy"].idxmax()]
        histories[model_name] = history_df
        ranked_by_epoch_by_model[model_name] = ranked_by_epoch
        summary_rows.append(
            {
                "model_name": model_name,
                "best_epoch": int(best_row["epoch"]),
                "val_accuracy": best_row["val_accuracy"],
                "val_loss": best_row["val_loss"],
                "wall_clock_s": wall_clock_s,
                "peak_memory_mb": peak_memory_mb,
            }
        )
    return pd.DataFrame(summary_rows), histories, ranked_by_epoch_by_model


def select_best_epoch_within_one_se(history_df: pd.DataFrame, n_val: int = 200) -> pd.Series:
    """Among epochs within 1 SE of the top val_accuracy, return the one with the
    fewest epochs trained — guards against picking the raw argmax of a noisy
    200-item hold-out, and prefers the cheaper checkpoint when accuracy ties."""
    import math

    best_acc = history_df["val_accuracy"].max()
    se = math.sqrt(best_acc * (1 - best_acc) / n_val)
    within = history_df[history_df["val_accuracy"] >= best_acc - se]
    return within.sort_values("epoch").iloc[0]
