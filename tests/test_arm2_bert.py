"""CPU-only smoke tests for Arm 2. No `from_pretrained` download anywhere in this
file — the tokeniser is built from an in-memory vocab, the model is a tiny
random-init BertConfig. Run behaviour (loss/accuracy trends), not convergence,
is asserted, matching test_arm1_experiments.py's smoke-test style."""

import math

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from amlh import arm2_bert as ab  # noqa: E402

VOCAB = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "hello", "world", "cough", "fever", "throat", "sore", "ache",
]


@pytest.fixture
def tiny_tokeniser(tmp_path):
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text("\n".join(VOCAB), encoding="utf-8")
    from transformers import BertTokenizer

    return BertTokenizer(vocab_file=str(vocab_file))


@pytest.fixture
def tiny_bert():
    from transformers import BertConfig, BertForSequenceClassification

    bert_config = BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=16,
        num_labels=4,
    )
    return BertForSequenceClassification(bert_config)


@pytest.fixture
def label_to_id():
    return {"disease_a": 0, "disease_b": 1, "disease_c": 2, "disease_d": 3}


@pytest.fixture
def tiny_frame():
    fit = pd.DataFrame(
        {
            "question": [
                "hello cough fever",
                "sore throat ache",
                "cough fever world",
                "hello throat ache",
            ],
            "disease": ["disease_a", "disease_b", "disease_c", "disease_d"],
        }
    )
    val = pd.DataFrame(
        {
            "question": ["hello cough", "sore ache"],
            "disease": ["disease_a", "disease_b"],
        }
    )
    return fit, val


def test_encode_labels_covers_full_universe(train):
    label_to_id, id_to_label = ab.encode_labels(train)
    n_classes = train["disease"].nunique()
    assert len(label_to_id) == n_classes
    assert len(id_to_label) == n_classes
    assert set(train["disease"].unique()) == set(label_to_id)
    assert all(id_to_label[i] == label for label, i in label_to_id.items())


def test_encode_labels_deterministic(train):
    first, _ = ab.encode_labels(train)
    second, _ = ab.encode_labels(train)
    assert first == second


def test_tokenise_shapes(tiny_tokeniser):
    encodings = ab.tokenise(["hello world", "cough"], tiny_tokeniser, max_length=8)
    assert encodings["input_ids"].shape == (2, 8)
    assert encodings["attention_mask"].shape == (2, 8)
    assert encodings["input_ids"].dtype == torch.long


def test_truncation_rate_bounds(tiny_tokeniser):
    long_text = "hello world hello world hello world hello world"
    short_text = "hello"
    rate = ab.truncation_rate([long_text, short_text], tiny_tokeniser, max_length=5)
    assert rate == 0.5
    rate_generous = ab.truncation_rate([long_text, short_text], tiny_tokeniser, max_length=64)
    assert rate_generous == 0.0


def test_make_loader_no_workers(tiny_tokeniser):
    encodings = ab.tokenise(["hello", "world", "cough", "fever"], tiny_tokeniser, max_length=8)
    loader = ab.make_loader(encodings, [0, 1, 2, 3], batch_size=2, shuffle=False)
    assert loader.num_workers == 0
    labels_tensor = next(iter(loader))[2]
    assert labels_tensor.dtype == torch.long


def test_make_loader_shuffle_seeded_reproducible(tiny_tokeniser):
    texts = ["hello", "world", "cough", "fever", "throat", "sore", "ache", "hello world"]
    labels = list(range(8))
    encodings = ab.tokenise(texts, tiny_tokeniser, max_length=8)

    def order(seed):
        loader = ab.make_loader(encodings, labels, batch_size=1, shuffle=True, seed=seed)
        return [batch[2].item() for batch in loader]

    assert order(44) == order(44)
    assert order(44) != order(1)


def test_run_epoch_returns_finite_loss(tiny_bert, tiny_tokeniser):
    device = torch.device("cpu")
    tiny_bert.to(device)
    texts = ["hello cough", "sore ache"]
    labels = [0, 1]
    encodings = ab.tokenise(texts, tiny_tokeniser, max_length=8)
    loader = ab.make_loader(encodings, labels, batch_size=2, shuffle=False)
    optimizer = torch.optim.AdamW(tiny_bert.parameters(), lr=1e-4)
    loss = ab._run_epoch(tiny_bert, loader, optimizer, device, train=True)
    assert math.isfinite(loss)


def test_train_model_history_shape(tiny_bert, tiny_tokeniser, label_to_id, tiny_frame):
    fit_df, val_df = tiny_frame
    best_state, history_df, ranked_by_epoch = ab.train_model(
        fit_df,
        val_df,
        label_to_id,
        model_name="unused-with-injected-model",
        lr=1e-4,
        batch_size=2,
        epochs=2,
        max_length=8,
        seed=44,
        model=tiny_bert,
        tokeniser=tiny_tokeniser,
    )
    assert {"epoch", "train_loss", "val_loss", "val_accuracy"} <= set(history_df.columns)
    assert len(history_df) == 2
    assert isinstance(best_state, dict)
    assert len(best_state) > 0
    assert isinstance(ranked_by_epoch, dict)
    assert set(ranked_by_epoch) == {0, 1}


def test_train_model_captures_predictions_matching_history(
    tiny_bert, tiny_tokeniser, label_to_id, tiny_frame
):
    """The captured per-epoch rankings must be the ones that produced the
    accuracies in history_df — otherwise McNemar would compare a different
    checkpoint from the one selection ranked."""
    fit_df, val_df = tiny_frame
    _, history_df, ranked_by_epoch = ab.train_model(
        fit_df,
        val_df,
        label_to_id,
        model_name="unused-with-injected-model",
        lr=1e-4,
        batch_size=2,
        epochs=2,
        max_length=8,
        seed=44,
        model=tiny_bert,
        tokeniser=tiny_tokeniser,
    )
    gold = val_df["disease"].tolist()
    for epoch, ranked in ranked_by_epoch.items():
        assert len(ranked) == len(val_df)
        recomputed = sum(r[0] == g for r, g in zip(ranked, gold)) / len(gold)
        logged = history_df.loc[history_df["epoch"] == epoch, "val_accuracy"].iloc[0]
        assert recomputed == pytest.approx(logged)


def test_train_model_calls_on_epoch_end_per_epoch(tiny_bert, tiny_tokeniser, label_to_id, tiny_frame):
    fit_df, val_df = tiny_frame
    calls = []
    ab.train_model(
        fit_df,
        val_df,
        label_to_id,
        model_name="unused-with-injected-model",
        lr=1e-4,
        batch_size=2,
        epochs=2,
        max_length=8,
        seed=44,
        model=tiny_bert,
        tokeniser=tiny_tokeniser,
        on_epoch_end=calls.append,
    )
    assert len(calls) == 2
    for row in calls:
        assert {"epoch", "train_loss", "val_loss", "val_accuracy"} <= set(row)


def test_train_model_rejects_unmapped_label(tiny_bert, tiny_tokeniser, label_to_id, tiny_frame):
    fit_df, val_df = tiny_frame
    val_df = val_df.copy()
    val_df.loc[0, "disease"] = "unmapped_disease"
    with pytest.raises(ValueError):
        ab.train_model(
            fit_df,
            val_df,
            label_to_id,
            model_name="unused-with-injected-model",
            lr=1e-4,
            batch_size=2,
            epochs=1,
            max_length=8,
            seed=44,
            model=tiny_bert,
            tokeniser=tiny_tokeniser,
        )


def test_predict_ranked_shape_and_labels(tiny_bert, tiny_tokeniser, label_to_id):
    id_to_label = {i: label for label, i in label_to_id.items()}
    device = torch.device("cpu")
    tiny_bert.to(device)
    ranked = ab.predict_ranked(
        tiny_bert, tiny_tokeniser, ["hello cough", "sore ache", "world"], id_to_label,
        max_length=8, batch_size=2, device=device,
    )
    assert len(ranked) == 3
    for row in ranked:
        assert len(row) == min(5, len(label_to_id))
        assert all(label in id_to_label.values() for label in row)


def test_predict_ranked_full_depth_when_top_k_none(tiny_bert, tiny_tokeniser, label_to_id):
    """top_k=None must rank every label — MRR over a truncated ranking is MRR@k,
    which would not be comparable to Arm 1's full-depth ranking."""
    id_to_label = {i: label for label, i in label_to_id.items()}
    device = torch.device("cpu")
    tiny_bert.to(device)
    ranked = ab.predict_ranked(
        tiny_bert, tiny_tokeniser, ["hello cough", "sore ache"], id_to_label,
        max_length=8, batch_size=2, device=device, top_k=None,
    )
    for row in ranked:
        assert len(row) == len(label_to_id)
        assert len(set(row)) == len(label_to_id)


def test_run_model_ablation_reuses_train_model(monkeypatch, label_to_id, tiny_frame):
    fit_df, val_df = tiny_frame
    calls = []

    def stub_train_model(fit_df, val_df, label_to_id, model_name, lr, batch_size, epochs, max_length, seed, on_epoch_end=None):
        calls.append(
            {"model_name": model_name, "lr": lr, "batch_size": batch_size, "epochs": epochs, "seed": seed}
        )
        history_df = pd.DataFrame(
            [
                {"epoch": 0, "train_loss": 1.0, "val_loss": 1.1, "val_accuracy": 0.5},
                {"epoch": 1, "train_loss": 0.5, "val_loss": 0.9, "val_accuracy": 0.6},
            ]
        )
        ranked_by_epoch = {0: [["disease_a"], ["disease_b"]], 1: [["disease_b"], ["disease_b"]]}
        return {"dummy": "state"}, history_df, ranked_by_epoch

    monkeypatch.setattr(ab, "train_model", stub_train_model)

    summary_df, histories, ranked_by_epoch_by_model = ab.run_model_ablation(
        fit_df, val_df, label_to_id, ["model-a", "model-b"],
        lr=1e-4, batch_size=2, epochs=2, max_length=8, seed=44,
    )

    assert len(calls) == 2
    assert [c["model_name"] for c in calls] == ["model-a", "model-b"]
    shared = [{k: v for k, v in c.items() if k != "model_name"} for c in calls]
    assert shared[0] == shared[1]
    assert set(summary_df["model_name"]) == {"model-a", "model-b"}
    assert set(histories) == {"model-a", "model-b"}
    assert set(ranked_by_epoch_by_model) == {"model-a", "model-b"}
    assert set(ranked_by_epoch_by_model["model-a"]) == {0, 1}


def test_measure_run_reports_wall_clock():
    import time

    result, wall_clock_s, peak_memory_mb = ab.measure_run(lambda: time.sleep(0.01) or "done")
    assert result == "done"
    assert wall_clock_s > 0
    if not torch.cuda.is_available():
        assert math.isnan(peak_memory_mb)
