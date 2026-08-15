# Prompt for the next Claude Code session — Arm 2 (BERT)

Copy everything below the line into a fresh Claude Code session in this repo.

---

Implement Arm 2 of the AMLH coursework: a fine-tuned BERT classifier over the 906 disease
labels. Read `CLAUDE.md` first and treat its hard rules as binding — especially rule #2 (no
test-set cell in notebooks 02–04), rule #3 (never state a number that code did not just print),
rule #4 (`SEED = 44`, re-seed before every stochastic stage) and rule #7 (`num_workers=0`).

**Plan before implementing.** Show me the module API and the notebook section list, and wait for
my approval before writing code — this step freezes four hyperparameters into `config.py`, so it
touches the evaluation boundary.

## Deliverables

1. `src/amlh/arm2_bert.py` — all logic.
2. `notebooks/03_arm2_bert.ipynb` — a thin notebook that imports from `src/amlh`, displays
   results, and persists to `artefacts/` and `figures/`. It must run standalone after a kernel
   restart by loading `artefacts/split_fit.csv` and `artefacts/split_val.csv`.
3. `tests/test_arm2_bert.py` — CPU-only, no model download in the test path.
4. Arm 2 fields frozen in `src/amlh/config.py`.

## What Arm 2 is

`question` → WordPiece tokenisation (`max_length=48`, pad + truncate) → encoder →
906-way linear classification head → argmax. `answer` is never an input. Labels are encoded
against the **full 906-class training universe**, not the classes present in the fit split, so
that the label space matches Arm 1 and the frozen test run.

## Design constraints, all non-negotiable

- **`max_length=48`.** Justified in `01_eda` by the question-length percentiles (mean 8.4 words,
  99th pct 18) and recorded in report §2.2. Do not use 512.
- **`num_workers=0`** in every DataLoader. Windows spawns rather than forks; anything else hangs.
- **Fit on the fit split only.** The tokeniser is pretrained (no fitting), but the label encoder
  and the model see `split_fit.csv` only. Never `fit`/`fit_transform` on validation.
- **Re-seed before every stochastic stage** with `config.set_seed()` — model init, DataLoader
  shuffling, and each ablation run.
- **Target the free Colab T4, 16 GB.** Prefer a batch size that fits without gradient
  accumulation at `max_length=48`; report peak memory.
- Stack: `transformers` + `torch` only, following `reference/NLP3-BERT_for_document_classification-solution.ipynb`.
  Write the manual epoch loop with `AdamW` as that practical does — do not introduce
  `Trainer`, `accelerate`, `datasets`, or anything the module did not teach.

## Module API — `src/amlh/arm2_bert.py`

Mirror the existing style in `arm1_experiments.py`: small tested primitives, orchestration
functions that return `pd.DataFrame`, docstrings that state *why*, no printing inside functions.
Reuse `evaluate.score_ranked` for metrics so Arm 2 is scored identically to Arm 1 — it takes
`ranked: list[list[str]]` (labels best-first) and `gold: list[str]`, and returns accuracy,
acc@5, macro-F1 and MRR. Return ranked label lists, not raw logits, from the prediction path.

Suggested surface (argue for a different one if you prefer):

- `encode_labels(train_df) -> (label_to_id, id_to_label)` over all 906 classes.
- `tokenise(texts, tokeniser, max_length)` → input_ids / attention_mask tensors.
- `make_loader(...)` — `num_workers=0`, shuffling seeded.
- `train_model(fit_df, val_df, model_name, lr, batch_size, epochs, max_length, seed)` →
  `(best_state_dict, history_df)` where `history_df` has one row per epoch with
  `epoch, train_loss, val_loss, val_accuracy`. **Both losses are required** — F6 in the report
  is the training/validation loss curve and it is explicitly marked as required by the brief.
- `predict_ranked(model, tokeniser, questions, id_to_label, ...) -> list[list[str]]`.
- `run_model_ablation(...)` → one row per encoder, scored with `evaluate.score_ranked`.

## Notebook sections

1. Load splits from `artefacts/`; `set_seed()`; print device and GPU name.
2. Label encoding over the 906-class universe; assert the count is 906.
3. Tokenisation sanity check: print the truncation rate at `max_length=48` on the fit split —
   it should be near zero, and that number belongs in report §2.2.
4. Train Bio_ClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`). Persist `history_df` to
   `artefacts/arm2_history_bioclinicalbert.csv`.
5. **Ablation vs `bert-base-uncased`**, same hyperparameters, same seed, same epochs — the only
   thing that varies is the checkpoint. Persist `artefacts/arm2_model_ablation.csv` and
   `artefacts/arm2_history_bertbase.csv`. This is the in-domain-pretraining claim in report §1.1;
   it has to be measured, not asserted.
6. **F6 — training/validation loss curves**, both encoders, saved to
   `figures/fig6_loss_curves.png`. Report §4.2 reads the train/val gap as evidence of
   memorisation under ~10 examples per class, so plot both curves for both models.
7. Validation predictions for the selected model persisted to
   `artefacts/arm2_val_predictions.csv` (columns: `question`, `gold`, `pred`, and the top-5
   labels) — `05_results` and the error analysis in §3.3 both need them, and McNemar against Arm 1
   needs per-item predictions aligned to the same validation order.
8. Freeze: print the selected `bert_model_name`, `learning_rate`, `batch_size`, `num_epochs`,
   then edit `config.py` as a source change.

## Selection rule

Epoch/checkpoint selection uses **validation accuracy on the standard hold-out** — `CLAUDE.md`
states explicitly that the shift-aware protocol governs Arm 1 index selection only, and that Arm 2
continues to use the standard hold-out. Do not apply the tie-break rule here.

Apply the same **within-1-SE-prefer-simplest** discipline used throughout Arm 1: validation has
200 items, SE ≈ 3.5pp, so differences below ~7pp are noise. If Bio_ClinicalBERT and
`bert-base-uncased` land within 1 SE of each other, say so and report it as an unresolved
comparison rather than declaring a winner — and prefer the fewest epochs within 1 SE of the best.

## Reporting back

Run the notebook and show me the real printed output. State wall-clock and peak GPU memory —
report §4.1 compares approaches on measured cost, not adjectives. Run `pytest tests/` before you
finish. Do not write any test-set cell.
