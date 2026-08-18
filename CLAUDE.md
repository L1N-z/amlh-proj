# AMLH Coursework — Patient Question Classification

UCL Applied Machine Learning for Health, NLP Dataset C. Predict a disease label from a patient
question. 906 classes, 8,891 training questions, 200 test questions over 102 classes.

Deliverables: a 2,000–2,500 word report and reproducible code. Graded on preprocessing (15),
algorithm design and implementation (40), results (25), discussion (10), introduction (10).

## Hard rules — do not violate these

1. **`answer` is never an inference-time input.** The only input at prediction time is
   `question`. Training-side answers MAY be indexed as class evidence; test-side answers must
   never be read by any code path.
2. **The test set informs no selection decision.** Model, hyperparameter, preprocessing,
   index-variant and prompt choices are made on validation only. Test *predictions* are
   generated once, in `05_results.ipynb`, after every hyperparameter is frozen in `config.py`.

   **One declared exception:** `01_eda.ipynb` uses test questions and labels for distributional
   diagnostics only — sibling-homogeneity measurement, near-duplication audit, and the novelty
   target for calibration. These characterise the validation–test relationship and select
   nothing. Any new use of test data outside `05_results.ipynb` requires my approval first.

   Never add a test-set *evaluation* cell to notebooks 02–04.

   **Declared exception #2 (dated 2026-08-09):** `scripts/measure_protocol_ranking.py` computes
   `test_acc` for the Arm 1 index-variant/scheme grid under both the standard and the hard
   (shift-aware) validation split, to measure which protocol better predicts test ranking
   (Spearman rho). This is a protocol diagnostic, not model selection: the index-variant/scheme
   choice is made from `std_val_acc`/`hard_val_acc` alone via the tie-break rule below, never
   from `test_acc`. Test-set access is confined to this one script; notebooks 02–04 remain free
   of any test-set cell.

   **Index-variant selection tie-break rule** (pre-registered before viewing this comparison on
   real test data; scope clarified 2026-08-15, see below): the standard hold-out decides wherever
   it discriminates. Where it does *not* — the candidates fall within 1 SE of each other, so the
   ranking is noise — the hard (shift-aware) hold-out breaks the tie, because §3.1 establishes
   the test distribution resembles the lexeme-absent stratum, not the lexeme-present majority the
   standard split is drawn from. A protocol cannot overturn a comparison it does not itself
   resolve at better than 1 SE. This governs Arm 1 index-variant and index-scheme selection; Arm 2
   epoch/checkpoint selection continues to use standard hold-out.

   **Decision recorded 2026-08-15** (applied in `02_arm1.ipynb` §4a/§5 and frozen in `config.py`):

   - **`index_variant = "QLAD"`** — the standard hold-out ties all four variants at 0.820
     (SE 0.027), so it does not discriminate. The tie-break fires: the hard hold-out ranks QLAD
     first at 0.328, by 0.065 over the runner-up against its own SE of 0.023.
   - **`index_scheme = "class_blob"`** — both protocols rank it first, each by more than its own
     SE (standard 0.850 by 0.120, SE 0.025; hard 0.398 by 0.073, SE 0.024). They agree, so the
     tie-break never fires.
   - **`ngram_range = (1, 2)`** — a consequence of the variant switch, not an independent choice.
     The variant-specific re-check (notebook §4b) re-tunes the vectoriser on QLAD blobs, which are
     NHS prose rather than short questions; the step-2 optimum tuned on variant Q did not transfer.

   All three are computed in the notebook from validation accuracies alone and printed with their
   margins and SEs; none reads `test_acc`.

   **Arm 2 encoder tie-break rule (recorded 2026-08-17, formalised after the comparison was
   seen — disclose this timing in the report).** Bio_ClinicalBERT and `bert-base-uncased` are
   compared on the standard hold-out with **McNemar's exact test** over the discordant pairs, not
   by judging an accuracy difference against a single-proportion SE: both encoders predict the
   same 200 items, so the comparison is paired and the items they agree on carry no evidence.
   If McNemar returns p ≥ 0.05 the comparison is **reported as unresolved** and Bio_ClinicalBERT
   is kept on the declared prior that an in-domain clinical encoder is the appropriate default
   for a clinical task. **This can and does retain the lower-scoring encoder** — that is the
   intended behaviour of a prior, and the report must say so rather than imply accuracy chose it.
   Only when McNemar resolves the comparison does measured accuracy decide.

   Provenance, to be stated in the report: an earlier implementation applied this preference as a
   hardcoded fallback in a notebook cell, gated on a hand-rolled single-proportion SE rather than
   a paired test, and it was never written down as a rule. It is recorded here now, with the
   instrument corrected to McNemar. Unlike the Arm 1 tie-break above, it was **not** pre-registered
   before the numbers were seen.

   Epoch/checkpoint selection *within* each encoder is unchanged: standard hold-out,
   within-1-SE-prefer-fewest-epochs.
3. **Never state a number that was not printed by code you just ran.** No estimated, recalled or
   plausible metrics — in code comments, in notebook markdown, or in chat.
4. **`SEED = 44`** from `config.py`. Re-seed before every stochastic stage.
5. **Fit vectorisers and models on training data only.** Never `fit` or `fit_transform` on
   validation or test text.
6. Notebooks exchange state through `artefacts/`, never through in-memory variables.
7. Keep num_workers=0 in the BERT DataLoaders. Windows spawns rather than forks, so num_workers>0 inside a notebook will hang or error.

## Architecture

- `src/amlh/` — all logic. Notebooks import from here and contain no substantive code.
- `notebooks/01_eda` (CPU) → `02_arm1` (CPU) → `03_arm2_bert` (GPU) → `04_arm3_llm` (GPU/API)
  → `05_results` (frozen test run).
- `artefacts/` — splits, grids, predictions, metrics. `figures/` — report figures.
- Each notebook must run standalone after a kernel restart by loading from `artefacts/`.

## Stack

Match the course practicals. scikit-learn, pandas, numpy, matplotlib, seaborn, spaCy,
transformers, torch. Do not introduce dependencies the module did not teach (no FAISS, no
sentence-transformers, no LangChain) unless asked — the marker expects course-aligned methods.

Target environment: free Google Colab T4. Keep everything within 16 GB GPU memory.

## Dataset facts (verified — do not re-derive or contradict)

- Class support is **near-uniform**: min 4, median 10, max 20, sd 0.98. **Not long-tailed.**
- 355/906 labels share a prefix family (`baby_` 53, `pregnancy_` 24, `social_` 23,
  `cosmetic_` 20, `contraception_` 14). Fine-grained, partly overlapping labels.
- 85 identical question strings map to more than one disease — an irreducible error floor.
- Train–test near-duplication is negligible (0.5% above 0.9 cosine).
- Question length: mean 8.4 words, 99th percentile 18. BERT `max_length=48`.
- Six labels break the naming convention (`Bronchitis`, `Bronchiolitis`, `Laryngitis`,
  `Pneumonia`, `Tonsillitis`, `Multiple_sclerosis`). Resolve NHS document filenames by
  lowercasing.
- NHS `.txt` documents carry ~22% boilerplate (nav text, Alamy credit URLs, review dates).
  Strip before indexing.

## The validation caveat (state this in any results discussion)

Validation is a single stratified hold-out of 200 items / 102 classes. It **over-estimates test
accuracy by roughly 35pp** (~0.77 vs ~0.40 for Arm 1). Cause: the ~10 questions per disease were
generated in one pass, so a held-out question is a phrasing sibling of those left in training
(cosine 0.572) while a test question is not (0.391). Cross-validation would not fix this — siblings
remain inside every fold.

Validation has 200 items, so SE ≈ 3.5pp. **Differences below ~7pp are noise.** Do not describe a
configuration as "better" on a sub-7pp margin without a paired test.

## Metrics

**Accuracy is the headline metric**, as the brief specifies. Top-5 and MRR appear only in error
analysis, to explain why accuracy is capped. Report bootstrap 95% CIs and McNemar's exact test
for every system comparison.

## Working style

- Plan before implementing. Show the plan; wait for approval on anything touching splits,
  leakage boundaries or evaluation.
- Small, verifiable steps. Run the code and show real output.
- Run `pytest tests/` after changes to `data.py`, `features.py` or `evaluate.py`.
- Prefer editing `src/*.py` over editing `.ipynb` directly.
- If a project fact is corrected twice, add it to this file.