## Introduction

### 1.1 Literature Review

Natural language processing has become invaluable for many clinical tasks, from Electronic Health Records (EHR) phenotyping to patient communication systems. Patient question classification supports medical information retrieval, automated patient triage and clinical decision support.

Historically, healthcare NLP relied on rule-based methods or classical machine learning, with TF–IDF vectorisation a traditional and durable baseline, paired with distance-based retrieval: questions are represented as sparse TF-IDF vectors and classified by their most similar labels. TF-IDF baselines remain robust and computationally efficient despite struggling with semantic variation, synonyms and novel vocabulary.

Deep learning captures context-aware embeddings, representing specialised terminology better than lexical methods: BERT models fine-tuned on clinical corpora — BioBERT, ClinicalBERT — achieve state-of-the-art clinical text classification. Large language models have since enabled in-context learning, classifying purely through prompting, with zero-shot, few-shot and chain-of-thought the dominant strategies for supplying context without any parameter update (Schulhoff et al., 2024). Fine-tuned encoders nonetheless beat zero-shot LLMs on biomedical classification (Chen et al., 2025), motivating retrieval-augmented generation — prepending retrieved clinical text as context — to close that gap without further training while reducing hallucination.

### 1.2 Motivation and Rationale

This coursework's primary aim is to evaluate traditional lexical retrieval against deep neural approaches on a 906-way medical condition classification task, using accuracy and F1, and considering implications for digital health safety, transparency and compute cost.

Three arms implement this comparison end to end: Arm 1, TF–IDF vectorisation with cosine k-nearest-neighbour retrieval over a class-level index; Arm 2, Bio_ClinicalBERT fine-tuned as a 906-way classifier; and Arm 3, the same TF–IDF retrieval shortlisting candidates for an LLM to re-rank by prompting. We hypothesise retrieval will be a strong baseline because class support is uniform at ~10 examples, limiting the encoder's advantage over it by that same sparsity.

The contribution goes beyond running three models against each other: a near-duplication audit ruling out leakage, a sibling-homogeneity diagnosis explaining why hold-out validation over-estimates test accuracy, and paired McNemar testing for every head-to-head comparison, rather than judging accuracy gaps against a single-proportion standard error.

## Methodology

### 2.1 Dataset Description

The dataset was synthetically generated using ChatGPT from the NHS UK website's patient-information section, released as part of the OpenGPT dataset after manual human validation.

It covers 906 conditions with multiple Q&A pairs each; training spans all 906 across 8,891 questions. The provided **test set** — read once, in §3's frozen run — covers 200 questions over 102 of those diseases, all present in training. It is distinct from the internally constructed **validation split** (§2.5), drawn from training alone and matching the test set's size and class count by design. Neither has duplicates or nulls, and 87.9% of classes hold exactly 10 pairs — near-uniform, needing no resampling. Questions are short: 8.4 words on average in training, 7.4 in test.

### 2.2 Data Preprocessing

Text was converted by two routes: TfidfVectorizer for retrieval, the Bio_ClinicalBERT WordPiece tokeniser for the neural arm. Only the question field is input; `reference_url` was dropped since it maps directly onto the disease label. Labels were preserved case-sensitively, since NHS filenames match label casing. Table 1 lists every setting searched against its default; `min_df=1` avoids discarding rare class-identifying terms.

Stop-word removal and lemmatisation were tested as ablations and rejected. Removal risks a large share of an 8.4-word question, and because the text is lowercased, two of the 155 clinical acronyms in the training questions — AS (ankylosing spondylitis) and ME (myalgic encephalomyelitis) — appear in scikit-learn's stop-word list and would be silently destroyed. IDF already down-weights frequent terms regardless.

Questions were capped at `max_length=48` rather than BERT's default 512: the longest needs only 44 tokens including special tokens, attention cost is quadratic in length, and truncation at 48 measures 0%.

NHS documents were cleaned before indexing, since under QLAD they are scored at retrieval time: navigation text, image-credit links and "page last reviewed" footers made up 4.96% of characters, and stripping them cut the terms present in over 95% of documents from 22 to 13.

**Table 1. Preprocessing and index-construction ablations.**

| **Stage** | **Parameter** | **Value used** | **Library default** | **Changed** | **Source** |
| --- | --- | --- | --- | --- | --- |
| Case folding | lowercase | True | True | No | default |
| Tokenisation | token_pattern | (?u)\b\w\w+\b | (?u)\b\w\w+\b | No | default |
| N-grams | ngram_range | (1, 2) | (1, 1) | Yes | grid (variant-specific re-tune, §2.3) |
| Stop words | stop_words | None | None | No | grid |
| DF floor | min_df | 1 | 1 | No | grid |
| DF ceiling | max_df | 1.0 | 1.0 | No | fixed |
| Term weighting | sublinear_tf | False | False | No | grid |
| IDF | use_idf | True | True | No | default |
| IDF smoothing | smooth_idf | True | True | No | default |
| Normalisation | norm | l2 | l2 | No | default |
| Lemmatisation | spaCy model / enabled | en_core_web_sm / disabled | n/a (not a TfidfVectorizer parameter) | No | ablation |
| BERT tokenisation | WordPiece: model / max_length/padding/truncation | Bio_ClinicalBERT / 48, pad to max_length, truncate | model_max_length=512 (BERT-family default) | Yes | grid (§2.2 length percentiles; §2.4 model selection) |

Source indicates how each value was arrived at: grid search, ablation (Table X), fixed a priori, or the library default.

**Table X. Step-2 preprocessing and index-variant grid (pre-freeze; superseded by the frozen configuration below).**

Run at the step-2 vectoriser (`Q`, ngram (1,1)), *before* §2.3 moved the frozen configuration to `QLAD` and forced the re-tune (notebook §4b) to `(1,2)`. Rows are deltas around the step-2 baseline (0.785), not the frozen system, which scores 0.850 validation / 0.765 test (§3.2). No ablation was re-run at the frozen vectoriser, so the QLAD/class_blob improvement shown is consistent with the evidence, not demonstrated by it.

| **Variant** | **Hold-out acc** | **Shift-aware acc** | **Δ vs step-2 baseline (shift-aware)** | **Vocab size** |
| --- | --- | --- | --- | --- |
| Step-2 baseline (`Q`, ngram (1,1)) | 0.785 | 0.180 | 0.000 (baseline) | 4257 |
| ngram_range=(1,2) | 0.745 | 0.205 | +0.025 | 29181 |
| min_df=2 | 0.490 | 0.077 | -0.102 | 2052 |
| sublinear_tf=True | 0.790 | 0.225 | +0.045 | 4257 |
| stop_words='english' | 0.765 | 0.182 | +0.003 | 4056 |
| Lemmatisation enabled (en_core_web_sm) | 0.810 | 0.215 | +0.035 | 3472 |
| NHS documents uncleaned | 0.785 | 0.180 | +0.000 | 4257 |
| index_variant=QL | 0.795 | 0.172 | -0.008 | 4333 |
| index_variant=QLA | 0.770 | 0.263 | +0.083 | 10924 |
| index_variant=QLAD | 0.835 | 0.328 | +0.148 | 16671 |

### 2.3 Traditional NLP Approach

TF–IDF vectorisation feeds a cosine k-nearest-neighbour classifier: the top-k neighbours in a class-level index vote, weighted by similarity, for the predicted label (Figure F4). At the frozen k=1 this collapses to nearest-neighbour, but the same weighted ranking backs Arm 3's depth-20 shortlist, letting one very close neighbour outrank several weak ones.

Four index variants were tested: question text only (Q), +label text (QL), +training answers (QLA), +NHS documents (QLAD). Training-side answers and documents are indexed only as class evidence — prediction-time input remains the test question alone — so this is knowledge-base construction, not leakage.

LinearSVC, Random Forest and Logistic Regression were also fit directly on TF-IDF vectors as supervised baselines, scoring 0.815, 0.765 and 0.635 — none beating retrieval's 0.850, so no reason to prefer a learned boundary over neighbour matching.

Selection outcomes are in §3.2. QLAD's NHS-prose index text forced a vectoriser re-tune, selecting `ngram_range=(1,2)`, `min_df=1` and no stop-word removal. Shortlist depth was fixed at 20, where acc@k plateaus; the ceiling bounding Arm 3 is the test figure of 0.975, not the validation 0.995.

### 2.4 Neural Approaches

#### Arm 2: fine-tuned BERT classifier

Bio_ClinicalBERT was fine-tuned end-to-end with a 906-way head (AdamW, `learning_rate=2e-5`, `batch_size=16`, `max_length=48`; Figure F5). Learning rate and batch size were held at defaults across both encoders rather than swept: at ~3.5pp SE a 200-item hold-out cannot resolve such a grid. Epoch selection uses a within-1-SE-prefer-fewest-epochs rule over the training and validation curves of Figure F6: epoch 22 reaches 0.870 but epoch 14 already reaches 0.850, so the cheaper checkpoint is kept.

Bio_ClinicalBERT (0.850) was compared against `bert-base-uncased` (0.875) by McNemar: 6 items only the former got right, 11 only the latter, p=0.332. Unresolved, so Bio_ClinicalBERT is retained on the declared prior that an in-domain clinical encoder is the default for a clinical task — not because it scored higher; it did not. This prior was formalised after the comparison had been seen, unlike Arm 1's pre-registered tie-break.

#### Arm 3: LLM shortlist re-ranking

Arm 3 reuses the frozen Arm 1 index to shortlist 20 candidate labels per question and prompts an LLM to pick one, under three conditions: zero-shot (candidates only), few-shot (plus two worked examples), and chain-of-thought (plus a reasoning instruction). The reply is matched back to a candidate by name; an unmatched reply falls back to Arm 1's top-1 — the system's prediction without the LLM — rather than a random guess.

Two generators are compared at temperature 0: primary `microsoft/MediPhi-Guidelines` and secondary `google/flan-t5-large`; prompts are in Table 2. A 6-cell grid on 200 items cannot support six-way selection, so choice runs in two pre-registered McNemar stages: prompt condition on the primary generator alone (unresolved → simplest-prompt prior), then model at that condition (unresolved → clinical prior). Outcomes are in §3.2.

### 2.5 Experimental Protocol

Stratified splitting via `train_test_split` is infeasible: it requires one held-out example per class, but the 906-class label space exceeds the 200-item validation size, and an unstratified split leaves class coverage to chance (177 classes under seed 42). We instead allocate a fixed quota across a sampled subset of classes, mirroring the test set's size and class count while keeping at least two training examples per class.

At n=200 the single-proportion SE is ≈3.5pp, so differences below ~7pp are indistinguishable from noise; every comparison is instead checked with McNemar's exact test, since paired predictions on the same items carry more information than two independent proportions. Accuracy is reported with bootstrap 95% CIs, and compute cost in Table 3.

A second, shift-aware protocol breaks ties the standard hold-out cannot resolve: a hold-out drawn only from training questions sharing no word with their own label (1,423 of 8,891), because §3.1 shows the test distribution resembles this lexeme-absent stratum far more than the standard split's lexeme-present majority. The standard hold-out decides wherever it discriminates by more than its own SE; only where it does not does the shift-aware split break the tie. This was pre-registered and scoped to Arm 1 index selection only — a scoping whose cost §4.2 discusses.

Every model, hyperparameter and prompt decision was made on validation alone, and test predictions were generated once, after all were frozen.

## Results

### 3.1 Exploratory Data Analysis

Class support, question length and label-family structure are summarised in Figure F1, and the TF–IDF feature space in Figure F3. Near-duplication and sibling homogeneity use different representations, matched to what each detects: bigrams for duplicate phrasing, unigrams for the within-class comparison. Median per-test-question maximum similarity to any training question is 0.485, with only 0.5% above the 0.9 near-duplicate threshold, so the task is not solvable by memorisation.

Training questions are lexical siblings (Figure F2): the nearest same-class sibling scores 0.572, while a test question's nearest own-class training question scores only 0.391 — and its nearest *other*-class question scores 0.510, closer by 0.119. Consistently, 84.0% of training questions contain a word from their own disease label against only 5.5% of test questions.

The consequence is direct: for 59.5% of test questions a wrong-class training question sits closer than any own-class one, against 33.8% under training leave-one-out. A nearest-neighbour rule is structurally disadvantaged on the test distribution, and any validation estimate drawn from siblings will be optimistic — motivating the shift-aware protocol (§2.5).

### 3.2 Performance Comparison

| System | Validation | Test | 95% CI (test) | Optimism | Macro-F1 (test) |
| --- | --- | --- | --- | --- | --- |
| Arm 1 (TF–IDF k-NN) | 0.850 | **0.765** | [0.705, 0.820] | 8.5pp | 0.563 |
| Arm 2 (Bio_ClinicalBERT) | 0.850 | **0.615** | [0.545, 0.680] | 23.5pp | 0.354 |
| Arm 3 (shortlist + LLM) | 0.720 | **0.720** | [0.660, 0.780] | 0.0pp | 0.494 |

Selection outcomes: all four index variants tied at 0.820 on the standard hold-out, so the tie-break deferred to the shift-aware split, ranking QLAD (0.328) over QLA (0.263). Arm 3's prompt conditions were indistinguishable (smallest p=0.511), so zero-shot was kept on the simplicity prior; the generator comparison did resolve, MediPhi beating flan-t5-large 0.720 to 0.635 (p=0.033).

**The validation ranking then inverts on test** (Figure F9):

| Pair | Validation | Test |
| --- | --- | --- |
| Arm 1 vs Arm 2 | 15/15, p=1.000, unresolved | 37/7, **p=5.3e-06, Arm 1 wins** |
| Arm 1 vs Arm 3 | 32/6, p=2.4e-05, Arm 1 wins | 20/11, **p=0.150, unresolved** |
| Arm 2 vs Arm 3 | 33/7, p=4.2e-05, Arm 2 wins | 11/32, **p=0.0019, Arm 3 wins** |

Every comparison the hold-out resolved, the test set reversed or dissolved. Per-arm optimism explains why: the encoder learns only from training questions and is nearly three times as optimistic as the retriever, whose QLAD index also carries NHS prose owing nothing to sibling phrasing — the arm most dependent on paraphrase collapsed hardest when the paraphrase disappeared. Arm 3's 0.0pp gap is arithmetic, not a finding: 81% of its test items are answered by Arm 1's own top-1 (§3.3), so it tracks Arm 1 minus an intervention penalty.

Arms 1 and 2 tying on validation while disagreeing on 15% of items means they are differently-wrong, not redundantly-right. A three-arm oracle would reach 0.830 — but Arm 2 is uniquely correct on only 2 of 200 test items, so a simple ensemble would likely underperform Arm 1 alone.

### 3.3 Error Analysis

Macro-F1 falls further than accuracy for every arm (Arm 1 0.725 → 0.563), so test errors concentrate in the sparsest classes rather than spreading evenly.

Fine-grained confusion dominates where it can be measured — but only conditioned on such an error being *possible*, i.e. the gold label having a prefix sibling among the 906, since a label without one adds a guaranteed zero to the denominator. On validation, where 88 of 200 items have a sibling, Arm 1 makes 18 within-family errors of 25 possible (72%) and Arm 2 makes 10 of 21. The metric is **not** reportable on test, where only 18 items have a sibling at all, leaving Arm 1 a denominator of one; the unconditioned 0% every arm scores there describes the sample's class composition, not the models (Figure F8).

Arm 3 decomposes cleanly. Of 200 test items it keeps Arm 1's rank-1 label on 120 (0.958, unchanged), falls back to it on 42 (0.429, inert by construction), and overrides it on 38 — scoring 0.289 where Arm 1 scored 0.526, rescuing 11 and breaking 20. Intervention precision rose from 6-in-50 on validation to 11-in-38 on test but stays net-negative. The LLM had the gold label in 97.5% of test shortlists and still failed to beat the retriever: the failure is overconfidence, not indecision, since every net loss came from an override and none from the fallback.

## Discussion

### 4.1 Comparison of Approaches

Traditional retrieval beats both neural arms on test at a fraction of the compute (Table 3): Arm 1 needs no GPU and no training step, while Arm 3's inference cost is incurred *on top of* Arm 1, since it cannot run without it.

Near-uniform support at ~10 examples per class explains the outcome: too little per-class data for fine-tuning to carve 906 classes apart, while an 8.4-word question's lexical overlap with its class evidence remains strong signal. The encoder compensated by memorising phrasing shared among sibling questions, which is exactly why it lost most when that phrasing disappeared — and validation gave no hint of it.

### 4.2 Impact of Design Choices

The index variant was Arm 1's largest lever, and the shift-aware protocol is what identified it: the standard hold-out could not separate the four variants at all. That is the clearest lesson here — when a hold-out is drawn from paraphrase siblings, a shift-aware split is the better guide to generalisation, and the test results bear it out. No isolating ablation was re-run at the frozen vectoriser, so this remains consistent with the evidence rather than demonstrated by it.

The lesson has an unclaimed corollary. The ablations show `sublinear_tf=True` at 0.530 on the shift-aware split against the frozen 0.398 — over five standard errors — while the standard hold-out separates them by 2pp, inside its own SE. The tie-break was scoped to index selection, so it never reached the vectoriser grid; scoped wider it would have chosen `True`. Because it surfaced only after the test set had been read, it is left unchanged by choice, not oversight: acting on it would be exactly the test-informed selection pre-registration exists to prevent — and it is the first thing to change in a repeat.

The rules differ in provenance: Arm 1's and Arm 3's were pre-registered before the numbers were seen, Arm 2's only after — disclosed rather than smoothed over. Both the Arm 2 and Arm 3 priors deliberately retained the lower-scoring option, the intended behaviour of a declared prior rather than a hidden accuracy claim.

### 4.3 Implications for Healthcare

85 identical question strings map to more than one disease — an irreducible error floor. A triage system here should abstain rather than guess: validation accuracy climbs from 0.850 to 0.945 once only the most-confident 36.5% of items are answered (Figure F7). Arm 3 shows the same principle accidentally — its only component that never hurt accuracy was the fallback that declined to re-rank. With the sharp degradation once phrasing shifts, patient-facing classifiers must be validated on genuinely independent data and deployed with a route to human review.
