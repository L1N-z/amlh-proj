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

    **Arm 3 prompt-condition tie-break (pre-registered 2026-08-19, before any condition was run).**
    The three prompt conditions are compared pairwise on the standard hold-out with **McNemar's
    exact test**, over the same 200 validation items. If no condition separates from the others at
    p ≥ 0.05, the comparison is **reported as unresolved** and `zero_shot` is selected on the
    declared prior of the simplest prompt — fewest tokens, no exemplar-selection confound, lowest
    inference cost. **This can and does retain a lower-scoring condition** — that is the intended
    behaviour of a prior, and the report must say so rather than imply accuracy chose it. Unlike
    the Arm 2 encoder rule, this one was pre-registered before the numbers were seen.

    **Arm 3 selection order and model tie-break (pre-registered 2026-08-23, before the
    corrected run was executed).** Arm 3 now compares three prompt conditions on two
    generators. A 6-cell grid on a 200-item hold-out (SE ≈ 3.5pp) cannot support six-way
    selection, so the choice is made in two stages, in this order:

    1. **Prompt condition** is chosen on the **primary** generator alone, by the pairwise
       McNemar rule above (unresolved → `zero_shot`).
    2. **Model** is chosen at that already-selected condition, by McNemar over the same 200
       items. If p ≥ 0.05 the comparison is **reported as unresolved** and
       `microsoft/MediPhi-Guidelines` is kept on the declared prior that an in-domain
       clinical model is the appropriate default for a clinical task — the same prior, and
       the same instrument, as the Arm 2 encoder rule. **This can and does retain the
       lower-scoring model**; the report must say so rather than imply accuracy chose it.

    The remaining cells of the grid are reported for transparency and select nothing.
    `google/flan-t5-large` is the secondary/ablation generator.

    **Decision recorded 2026-08-23** (from the corrected Colab run; frozen in `config.py`).
    Both guards held: the run's splits are byte-identical to the canonical ones and
    `arm1_accuracy` reads 0.850 on every row, so the QLAD index was the frozen one.

    - **`prompt_mode = "zero_shot"`** — UNRESOLVED, not won. All three pairwise McNemar tests on
      the primary generator return p ≥ 0.05 (smallest: cot vs few_shot, p = 0.5114), so the
      pre-registered tie-break fires and the simplest-prompt prior is kept. This **retains the
      lower-scoring condition** (zero_shot 0.720 vs few_shot 0.730).
    - **`arm3_model_name = "microsoft/MediPhi-Guidelines"`** — RESOLVED on measured accuracy.
      McNemar at the selected condition: 37 / 20 discordant, p = 0.0331 < 0.05, accuracy 0.720 vs
      0.635. The clinical prior never had to fire. Do not describe this as a tie-break outcome.

    **Arm 3 headline result — Arm 3 loses to the Arm 1 shortlist it was handed.** 0.720 vs 0.850,
    McNemar p = 2.4e-05 (6 rescues, 32 breakages). Mechanism, from `arm3_val_predictions.csv`:
    of 200 items the LLM keeps rank 1 on 118 (0.966, identical to Arm 1 on those items), falls
    back on 32 (0.750, inert by construction — the fallback returns Arm 1's top-1), and actively
    re-ranks 50, scoring 0.120 where Arm 1 scored 0.640 on those same items. **All 32 losses come
    from active re-ranks; none from the fallback.** Intervention precision is 6/50.

    The 16% fallback rate is **not** a parser defect — checked, and 0 of 32 fallback outputs match
    any of the 906 labels, so they are free-text hallucinations and `parse_diagnosis_name` is
    correct. Do not retune the parser post-hoc; report the rate as a finding.

## Known limitation — `sublinear_tf` was never tie-broken (observed 2026-08-25, decided 2026-08-26)

`report/tables/table5_ablations.md` records `sublinear_tf=True` at **0.530** on the shift-aware
hold-out against the frozen `False` at **0.398** — a 13.3pp gap, more than five times that
split's SE — while the standard hold-out separates them by 2.0pp (0.870 vs 0.850), inside its own
~2.5pp SE.

That is exactly the pattern the pre-registered tie-break was written for: the standard protocol
does not discriminate, the shift-aware one does. But the rule's scope is stated as governing
**index-variant and index-scheme selection only**, and `sublinear_tf` is a vectoriser parameter,
so the rule was not violated — it simply never applied.

One nuance to keep straight when writing about this. The tie-break *rule* was pre-registered
(2026-08-09), but the *scope sentence* is marked "scope clarified 2026-08-15" — the same date the
index-variant decision was recorded, i.e. it was written down as the rule was being applied. There
is a defensible reason the vectoriser sat outside it (the §4b re-tune is recorded as "a consequence
of the variant switch, not an independent choice", so it was treated as mechanical rather than as a
selection needing a rule), but `sublinear_tf` came from the step-2 grid and was not part of that
re-tune. It therefore falls in a genuine gap, not a principled exclusion. Do not claim in the
report that the vectoriser was deliberately excluded.

### Decision (2026-08-26): leave it, and report it as a limitation

**`sublinear_tf` stays `False`. Nothing is re-run, and no further test-set quantity is read for
it.** This was chosen deliberately over two alternatives, both of which were considered and
rejected:

- **Switch to `True` and re-run the test.** Rejected as illegitimate. The evidence for `True` is
  validation-side, but the impulse to look again came *after* the test result was known, and
  whether the same look would have happened had test come back at 0.90 is unanswerable. That is a
  garden-of-forking-paths selection, and it breaks hard rule #2 outright.
- **Measure `True` on test as a declared post-hoc diagnostic**, extending the precedent of
  Declared exception #2 (which already computes test accuracy for the index-variant grid as a
  *protocol diagnostic, not model selection*). Defensible in principle, and it would have tested
  the report's central methodological claim directly. Rejected on asymmetry: once measured it must
  be reported whichever way it falls, a confirming result complicates the story ("why did you not
  ship the better system?") for modest gain in a 10-mark Discussion section, and a refuting result
  would damage the report's strongest methodological point. The report is also already over its
  word cap.

The reasoning to carry into the report: **following a pre-registered rule even when a post-hoc
look suggests a higher score was available is what pre-registration is for.** Frame it as
discipline exercised, not as an oversight discovered — the choice not to act is the point. Say
plainly that the shift-aware split predicts `True` would have generalised better, and that this
is the first thing to change in any repeat of the work.

## Test-run scope (decided 2026-08-23, before `05_results.ipynb` was written)

All three arms generate test predictions in `05_results.ipynb`, so the report can give the
per-method error analysis the brief asks for. **No "final deployed system" is selected.** The
brief asks for a comparison of algorithms, not the nomination of a winner, so every arm's test
accuracy is reported side by side and no post-hoc selection rule is invented to break the
Arm 1 / Arm 2 validation tie at 0.850.

That tie is exact, and is the reason no rule could have been written honestly:
`cross_arm_val_mcnemar.csv` records 15 items only Arm 1 gets right, 15 only Arm 2 gets right,
30 discordant, **p = 1.000**. The arms are differently-wrong, not redundantly-right.

**Models are fit on `split_fit` only for the test run — no refit on fit + val.** The tested model
is then the exact model that was validated, so the validation→test comparison describes one
object and the ~35pp optimism analysis stays clean. The forgone data is 200 of 8,891 items
(2.2%), negligible against a 3.5pp SE. Do not "improve" this by refitting on the full training
set later.

**Arm 2 is retrained in the Colab results notebook, never loaded from a Drive checkpoint.** The
Arm 2 run recorded `refit_agreement = 1.0`, so a seed-44 retrain at the frozen 15 epochs
reproduces the validated model exactly, and the notebook stays reproducible for a marker who has
no access to anyone's Drive.

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

## Colab runs — inputs must be complete or the frozen config is not what ran

`.gitignore` excludes `data/` and all of `artefacts/`, so a Colab clone has **neither**.
Every input has to arrive through an uploaded zip, built by
`python scripts/make_arm3_colab_inputs.py` and unzipped at the repo root
(`!unzip -o arm3_colab_inputs.zip -d .`). The archive names its members explicitly:
`artefacts/split_fit.csv`, `artefacts/split_val.csv`, `artefacts/arm1_val_predictions.csv`,
`data/patient_qa_classification_train.csv`, and all 906 `data/db_nhs_qa_classification/*.txt`.

**Incident, 2026-08-23.** The first Arm 3 Colab run uploaded `arm2_colab_inputs.zip`, which
holds only the two split CSVs. The NHS documents were absent, `load_class_doc` returned `""`
for all 906 classes, and `features.build_index` dropped the `D` component without
complaining — so the frozen **QLAD** index silently ran as **QLA**. Confirmed by exact
reproduction: the run's 200 shortlist top-1 labels and 200 top-sims match QLA at 1.000 and
QLAD at 0.915/0.000. Every Arm 3 number from that run is off-config and was discarded.

Two guards now make this impossible to repeat, and neither may be removed:

- `features.require_doc_coverage`, called by both index builders whenever the variant
  contains `D`, raises rather than degrading.
- `arm3_llm.assert_reproduces_arm1` checks the shortlist's top-1 against
  `artefacts/arm1_val_predictions.csv` item for item, before any prompt is sent.

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

Validation is a single stratified hold-out of 200 items / 102 classes, and it over-estimates test
accuracy. Cause: the ~10 questions per disease were generated in one pass, so a held-out question
is a phrasing sibling of those left in training (cosine 0.572) while a test question is not
(0.391). Cross-validation would not fix this — siblings remain inside every fold.

**Measured optimism, all three arms, from the completed frozen test run (2026-08-25).**
Arm 1 ran on CPU in `05_results.ipynb`; Arms 2 and 3 in `05_results_colab.ipynb`. The Arm 3
reproduction guard passed (its `arm1_pred` matches `arm1_test_predictions.csv` item for item)
and the Arm 2 retrain reproduced the selected model exactly (`retrain val accuracy: 0.8500 |
recorded: 0.8500 | delta: 0.0000`, printed in §3b).

| Arm | validation | test | optimism | macro-F1 (test) |
|---|---|---|---|---|
| Arm 1 — TF-IDF + k-NN | 0.850 | **0.765** | **8.5pp** | 0.563 |
| Arm 2 — Bio_ClinicalBERT | 0.850 | **0.615** | **23.5pp** | 0.354 |
| Arm 3 — shortlist + MediPhi | 0.720 | **0.720** | **0.0pp** | 0.494 |

Arm 1 test acc@5 = 0.925, acc@10 = 0.945, acc@20 = 0.975 (Arm 3's test shortlist ceiling).

**The validation ranking inverts on test, and this is the report's headline.** Every comparison
the hold-out resolved, the test set reverses or dissolves:

| Pair | Validation | Test |
|---|---|---|
| Arm 1 vs Arm 2 | 15/15, p = 1.000 — unresolved | 37/7, **p = 5.3e-06, Arm 1 wins** |
| Arm 1 vs Arm 3 | 32/6, p = 2.4e-05, Arm 1 wins | 20/11, **p = 0.150 — unresolved** |
| Arm 2 vs Arm 3 | 33/7, p = 4.2e-05, Arm 2 wins | 11/32, **p = 0.0019, Arm 3 wins** |

The per-arm optimism explains it: the encoder has only training questions to learn from and is
nearly three times as optimistic as the retriever, whose QLAD index also carries NHS prose that
owes nothing to sibling phrasing.

**Arm 3's 0.0pp optimism is arithmetic, not a finding.** 81% of its test items are kept-at-rank-1
or fallback — Arm 1's own answer — so Arm 3 tracks Arm 1 minus an intervention penalty, and the
penalty shrank (0.130 → 0.045) because the retriever it overrides got worse. The exact
0.720 = 0.720 equality is coincidence; say so rather than making a story of it.

Arm 3 test decomposition: kept rank 1 n=120 (0.958, delta 0), fell back n=42 (0.429, delta 0,
inert as designed), moved off rank 1 n=38 (0.289 against Arm 1's 0.526). Intervention precision
rose from 6/50 = 0.12 on validation to 11/38 = 0.29 on test, still net-negative.

Three-arm agreement on test: 107 all-correct, 34 all-wrong; uniquely correct 11 / 2 / 6 for
Arms 1 / 2 / 3; oracle ceiling 0.830. Arm 2 contributes 2 unique items.

**The within-family error metric is only reportable on validation.** Conditioned on such an error
being *possible* (gold's prefix family having more than one member in the 906-label space):
validation has 88/200 such items and Arm 1 makes 18 within-family errors out of 25 possible
(0.72, Arm 2: 10/21 = 0.48); test has only 18/200, and Arm 1's denominator is a **single error**.
The unconditioned test figure reads 0% for every arm, which is a fact about the test sample's
class composition, not about the models. `results.family_error_summary` enforces the conditioning
and returns `None` rather than `0.0` when nothing was possible. Do not write that "the error mode
changed between splits".

**The earlier "~0.77 vs ~0.40, roughly 35pp" figure is superseded and must not be repeated.** It
described a pre-freeze Arm 1 — index variant `Q`, the question-only index, before the QLAD /
class_blob / ngram (1,2) switch. It is not the frozen system and never was. The real optimism of
the frozen system is 8.5pp, a quarter of the retired figure.

Do not attribute the improvement to any single lever as though it were measured: what *is*
measured is 0.765 for the frozen config, alongside `report_outline.md`'s recorded span of
0.440–0.700 in test accuracy across the four class_blob variants at the pre-§4b vectoriser. The
direction is consistent with the index switch carrying it, but no isolating ablation was run at
the frozen vectoriser, so state it as consistent-with, not as demonstrated.

Anything in `report_outline.md` or `report/report_draft.md` still quoting ~0.40 or 35pp needs
rewriting before submission — §2.5 and §4.2 of the outline are both built on the retired number.

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