## Introduction

### 1.1 Literature Review

Natural language processing (NLP) has become invaluable for many clinical tasks, from Electronic Health Records (EHR) phenotyping to patient communication systems. Patient question classification supports medical information retrieval, automated patient triage and clinical decision support.

Historically, healthcare NLP relied on rule-based methods or classical machine learning (ML), with TF–IDF vectorisation a traditional baseline, paired with distance-based retrieval: questions are represented as sparse TF-IDF vectors and classified by most similar labels. TF-IDF baselines remain robust and computationally efficient despite struggling with semantic variation, synonyms and novel vocabulary.

Deep learning captures context-aware embeddings, representing specialised terminology better than lexical methods: BERT models fine-tuned on clinical corpora — BioBERT, ClinicalBERT — achieve state-of-the-art clinical text classification. Large language models (LLMs) have since enabled in-context learning, classifying purely through prompting, with zero-shot, few-shot and chain-of-thought strategies for supplying context without any parameter update (Schulhoff et al., 2024). Fine-tuned encoders nonetheless beat zero-shot LLMs on biomedical classification (Chen et al., 2025), motivating retrieval-augmented generation (RAG), which adds clinical text as context to improve accuracy and reduce hallucination.

### 1.2 Motivation and Rationale

The primary aim is to evaluate traditional lexical retrieval against deep neural approaches on a 906-class medical condition classification task, using accuracy and F1, and considering implications for digital health safety, transparency and compute cost.

Three arms implement this comparison: 
- Arm 1, TF–IDF vectorisation with cosine k-nearest-neighbour retrieval over a class-level index; 
- Arm 2, Bio_ClinicalBERT fine-tuned as a 906-way classifier; 
- Arm 3, TF–IDF retrieval shortlisting candidates for an LLM to re-rank by prompting. Retrieval is expected to be a strong baseline because class distribution is uniform at 10 examples for most classes, limiting the encoder's advantage over it by that same sparsity.

## Methodology

### 2.1 Dataset Description

The dataset was synthetically generated using ChatGPT from the NHS UK website's patient-information section, released as part of the OpenGPT dataset after manual human validation.

It covers 906 conditions with multiple Q&A pairs each, across 8,891 questions. The provided test set covers 200 questions over 102 of those conditions, all present in training. It is distinct from the internally constructed validation split, taken from training and matching the test set's size and class count. Neither has duplicates or nulls, and is almost uniformly distributed with 87.9% of classes holding exactly 10 pairs, requiring no resampling. Questions are short, averaging 8.4 words in training, 7.4 in test.

### 2.2 Data Preprocessing

Text was converted using TfidfVectorizer for retrieval and Bio_ClinicalBERT WordPiece tokeniser for the neural arm. Only the question field is inputted and `reference_url` was dropped since it maps directly to the disease label. Case sensitive labels were preserved since NHS filenames match label casing. Table 1 lists every setting searched against its default; `min_df=1` avoids discarding rare class-identifying terms.

Stop-word removal and lemmatisation were tested as ablations and rejected. Removal isn't safe due to the short question length (8.4), and because the text is lowercased, two of the 155 clinical acronyms in the training questions — AS (ankylosing spondylitis) and ME (myalgic encephalomyelitis) appear in scikit-learn's stop-word list and would be discarded, IDF already down-weights frequent terms.

Questions were capped at `max_length=48` rather than BERT's default 512 because the longest is 44 tokens including special tokens, and truncation at 48 measures 0%, allowing to safely reduce the computational costs.

NHS documents were cleaned before indexing, since for QLAD they are scored at retrieval time. Navigation text, image-credit links and "page last reviewed" footers made up 4.96% of characters, and stripping them reduced the terms present in over 95% of documents from 22 to 13.

**Table 1: Preprocessing and index-construction ablations.**

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

_Source indicates how each value was chosen: grid search, ablation (Table 2)._

**Table 2: Step-2 preprocessing and index-variant grid (pre-freeze; superseded by the frozen configuration below).**

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

*This grid was run before the configuration was re-tuned and frozen, so the accuracy differences shown (Δ column) are measured against the step-2 baseline (0.785), not against the final frozen system. Because the grid wasn't re-run at the final vectoriser settings, the QLAD/class-blob improvement it shows is suggestive rather than confirmed.*

### 2.3 Traditional NLP Approach

TF–IDF vectorisation is followed by a cosine k-nearest-neighbour classifier, where the predicted label is voted for by top-k (k=1) neighbours and weighed by similarity (Figure F4). The same weighted ranking informs Arm 3's depth-20 shortlist, letting one very close neighbour outrank several weak ones.

**Figure F4: Arm 1 — TF-IDF retrieval over a QLAD class-blob index.** *Pipeline diagram: `split_fit` questions and the 906 NHS documents are indexed under the frozen QLAD/class_blob/ngram(1,2) configuration; the test question is the only inference-time input, producing a top-1 prediction and, for Arm 3, a top-20 shortlist. Green blocks mark settings frozen in `config.py`.* *[Insert `figures/fig4_arm1_workflow.png` here.]*

Four index variants were tested: 
+ question text only (Q);
+ +label text (QL); 
+ +training answers (QLA); 
+ +NHS documents (QLAD). Training-side answers and documents are indexed only as class evidence and prediction-time input remains the test question alone, so this is knowledge-base construction, not leakage.

LinearSVC, Random Forest and Logistic Regression were fit directly on TF-IDF vectors as supervised baselines, scoring 0.815, 0.765 and 0.635. They did not beat retrieval's 0.850, therefore neighbour matching is preferred.

NHS-prose index text forced a vectoriser re-tune, selecting `ngram_range=(1,2)`, `min_df=1` and no stop-word removal. Shortlist depth was fixed at 20 and the ceiling bounding Arm 3 is the test figure of 0.975, instead of the 0.995 validation.

### 2.4 Neural Approaches

#### Arm 2: fine-tuned BERT classifier

Bio_ClinicalBERT was fine-tuned with a 906-way head (AdamW, `learning_rate=2e-5`, `batch_size=16`, `max_length=48`) (Figure F5). Learning rate and batch size were set at defaults across both encoders since at 3.5pp SE a 200-item hold-out cannot resolve such a grid. Epoch selection selects the fewest epochs within 1 standard error (SE) over the training and validation curves (Figure F6). Epoch 22 reaches 0.870 but epoch 14 already reaches 0.850, so the cheaper checkpoint was kept.

**Figure F5: Arm 2 (fine-tuned encoder) and Arm 3 (shortlist + LLM).** *Pipeline diagram covering both neural arms: Arm 2 tokenises and fine-tunes Bio_ClinicalBERT end-to-end; Arm 3 reuses Arm 1's frozen top-20 shortlist, prompts the LLM under the selected condition, and falls back to Arm 1's own top-1 (dashed red path) whenever the reply cannot be parsed to a shortlist label — so Arm 3's accuracy is anchored to Arm 1's. Green blocks mark settings frozen in `config.py`.* *[Insert `figures/fig5_arms23_workflow.png` here.]*

**Figure F6: Arm 2 training/validation loss.** *Cross-entropy loss by epoch for Bio_ClinicalBERT and `bert-base-uncased`, showing both curves flattening well before the selected epoch-15 checkpoint.* *[Insert `figures/fig6_loss_curves.png` here.]*

Bio_ClinicalBERT (0.850) was compared against `bert-base-uncased` (0.875) by McNemar: 6 items only the first got right, 11 only the latter, p=0.332. The comparison has failed to resolve the selection, so Bio_ClinicalBERT was kept as the domain-specific model.

#### Arm 3: LLM shortlist re-ranking

Arm 3 reuses the frozen Arm 1 index to select 20 candidate labels per question and prompts an LLM to pick one (pipeline in Figure F5), under three conditions:
- zero-shot (candidates only);
- few-shot (plus two worked examples); 
- and chain-of-thought (COT) (with a reasoning instruction). 

**Table 3: Prompt templates and token budgets for the three Arm 3 conditions (zero-shot, few-shot, chain-of-thought).** *[Insert table content from `report/tables/table2_prompts.md` here.]*

The reply was matched with a candidate by name and Arm 1's top-1 was used as a safe fallback if no match was found.

Two generators were compared at temperature 0: primary `microsoft/MediPhi-Guidelines` and secondary `google/flan-t5-large` (Table 3). Two pre-registered McNemar stages were used to choose between them: prompt condition on the primary generator alone, then model at that condition.{EXPLAIN THIS OR REWRITE}
The prompt condition and the generator model were chosen in two separate steps, fixed in advance rather than tested as all six combinations at once. First, the three prompt conditions were compared against each other using only the primary generator, and McNemar's test picked the best one. Then, with that prompt condition held fixed, the primary and secondary generators were compared against each other, again with McNemar's test, to choose the model. Testing in two fixed steps avoids the extra false-positive risk of running all six pairwise comparisons at once.

### 2.5 Experimental Protocol

Stratified splitting by `train_test_split` is infeasible: it requires one held-out example per class, but the 906-class label space exceeds the 200-item validation size, and an unstratified split can result in non-uniform class representation (177 classes under seed 42). Therefore quota-based sampling was used, mirroring the test set's size while keeping at least two training examples per class.

At n=200 the single-proportion SE is ≈3.5pp, so differences below ~7pp are considered as noise. Every comparison was instead checked with McNemar's test, which only uses the questions where the systems disagreed, since paired predictions on the same items carry more information than two independent proportions. Accuracy is reported with bootstrap 95% CIs (Table 4).

A shift-aware protocol was used to resolve ties which the standard hold-out failed. A hold-out was drawn only from training questions sharing no words with their own label (1,423 of 8,891), because the test distribution more closely resembles this lexeme-absent sample. The standard hold-out decided between candidates if the gap between them exceeded one SE, otherwise the shift-aware hold-out's ranking was used to break the tie. This was used for Arm 1 index selection only, as discussed in 4.2.

Validation informed every model, hyperparameter, and prompt decision, and test predictions were generated only once after freezing.

## Results

### 3.1 Exploratory Data Analysis

Class support, question length and label-family structure are summarised in Figure 1, and the TF–IDF feature space in Figure 3. Near-duplication and sibling homogeneity use different representations, bigrams detecting duplicate phrasing, unigrams the within-class comparison. Median per-test-question maximum similarity to any training question is 0.485, with only 0.5% above the 0.9 near-duplicate threshold, rejecting near-duplication as an explanation for the results.

**Figure 1: Dataset composition and length/duplication diagnostics.** *(a) Training questions per class are near-uniform (min 4, median 10, max 20) except a small rare-class tail. (b) Train vs. test question-length distributions in words. (c) The twelve largest label-prefix families. (d) Maximum cosine similarity from each test question to any training question, with the 0.9 near-duplicate threshold marked.* *[Insert `figures/fig1_eda.png` here.]*

Training questions are lexical siblings (Figure 2), where the nearest same-class sibling scores 0.572. The test question's nearest own-class training question scores only 0.391 and its nearest other-class question scores 0.510 (0.119 difference). 84.0% of training questions, but only 5.5% of test questions, contain a word from their own disease label.

**Figure 2: Sibling phrasing homogeneity explains the val-test gap.** *Cosine-similarity distributions: each training question to its nearest same-class sibling, each test question to its nearest own-class training question, and each test question to its nearest other-class training question — the basis for the shift-aware validation protocol (§2.5).* *[Insert `figures/fig2_novelty.png` here.]*

**Figure 3: PCA of TF-IDF question vectors, 10 largest classes.** *Two-dimensional PCA projection of TF-IDF question vectors, illustrating within-class lexical spread and between-class overlap in the retrieval feature space.* *[Insert `figures/fig3_pca.png` here.]*

Consequently, for 59.5% of test questions a wrong-class training question is closer than any own-class one. Therefore, nearest-neighbour rule is ineffective for the test distribution, and any validation estimate drawn from siblings will be optimistic, motivating the shift-aware protocol.

### 3.2 Performance Comparison

**Table 4: Validation and test accuracy by arm, with bootstrap 95% confidence intervals, optimism, and macro-F1.**

| System | Validation | Test | 95% CI (test) | Optimism | Macro-F1 (test) |
| --- | --- | --- | --- | --- | --- |
| Arm 1 (TF–IDF k-NN) | 0.850 | **0.765** | [0.705, 0.820] | 8.5pp | 0.563 |
| Arm 2 (Bio_ClinicalBERT) | 0.850 | **0.615** | [0.545, 0.680] | 23.5pp | 0.354 |
| Arm 3 (shortlist + LLM) | 0.720 | **0.720** | [0.660, 0.780] | 0.0pp | 0.494 |

All four index variants scored 0.820 on the standard hold-out, so shift-aware split was the tie-breaker, ranking QLAD (0.328) over QLA (0.263). Arm 3's prompt conditions were indistinguishable (smallest p=0.511), so zero-shot was chosen for simplicity. In the generator comparison via McNemar's test MediPhi (0.720 validation accuracy) beat flan-t5-large (0.635) (p=0.033).

**The validation ranking then inverts on test** (Figure F9):

**Figure F9: Test accuracy by arm.** *Test-set accuracy (200 items) with bootstrap 95% confidence intervals for each arm — the test-side counterpart to Table 4.* *[Insert `figures/fig9_test_accuracy.png` here.]*

**Table 5: Pairwise McNemar comparisons between arms, validation vs. test.**

| Pair | Validation | Test |
| --- | --- | --- |
| Arm 1 vs Arm 2 | 15/15, p=1.000, unresolved | 37/7, **p=5.3e-06, Arm 1 wins** |
| Arm 1 vs Arm 3 | 32/6, p=2.4e-05, Arm 1 wins | 20/11, **p=0.150, unresolved** |
| Arm 2 vs Arm 3 | 33/7, p=4.2e-05, Arm 2 wins | 11/32, **p=0.0019, Arm 3 wins** |

Every ranking that seemed reliable on the hold-out was reversed on the test set. Arm 2 learned to recognise the phrasing of training questions rather than the disease, so it performed well on questions with similar phrasing and struggled when test questions used different wording. Arm 1's TF-IDF index was less influenced because it's built from NHS description text, not paraphrases. This is also why Arm 3 closely tracks Arm 1, since the LLM agrees with Arm 1's top candidate in 81% cases.

Arm 1 and Arm 2 reached identical validation accuracy but disagreed on 15% of test items, suggesting they make mistakes on different questions. However, combining them would likely underperform Arm 1 alone, because Arm 2 is only uniquely correct on 2 of the 200 test items. Since Arm 1 is correct on almost every item where the two disagree, merging Arm 2's answers would add more mistakes than corrections.

### 3.3 Error Analysis

Macro-F1, the unweighted average of each class's F1-score, is lower than accuracy for every arm on test (Arm 1 falls from 0.725 to 0.563), suggesting errors are not spread evenly.

Many of the 906 disease labels share a text prefix (siblings). Of the 88 validation questions that had a true label sibling, Arm 1 made 25 errors, and 18 of them (72%) were within-family. Arm 2 made 21 errors on the same 88 questions, 10(48%) within-family errors.
This cannot be reported for the test set as too few (18 of the 200) test questions have a sibling.

**Figure F8: Most frequent confusions (red = a same-family error was possible).** *The most frequent test-set misclassifications by count, one panel per arm. None of the shown top confusions happen to be within-family pairs, consistent with the test set's sparse sibling coverage noted above.* *[Insert `figures/fig8_confusions.png` here.]*

Arm 3 uses the same 20-candidate shortlist as Arm 1 (built by TF–IDF retrieval of candidates, adding them to the prompt, then generating an answer). Out of 200 test questions the LLM kept Arm 1's top choice for 120 (accuracy 0.958, unchanged), fell back to Arm 1's top choice for 42, because its own output was not a valid candidate (accuracy 0.429, also unchanged), and chose a different candidate to Arm 1's on 38 (an override), where accuracy fell from 0.526 (Arm 1) 0.289 (Arm 3).

On those 38 override questions, Arm 1 was originally correct on 20 and wrong on 18, so overriding fixed 11 previously wrong answers but broke 20 previously correct ones.

The true label appeared in the LLM's 20-candidate list on 97.5% of test questions, so the LLM was rarely missing the right answer. Its weaker accuracy on override questions suggests its confidence in choosing the wrong candidate.

## Discussion

### 4.1 Comparison of Approaches

Traditional retrieval beats both neural arms on test at a fraction of the compute (Table 6): Arm 1 needs no GPU and no training step, while Arm 3's inference cost is in addition to Arm 1, since it builds up on it.

**Table 6: Computational resources per arm (hardware, wall-clock time, what was measured).** *[Insert table content from `report/tables/table3_resources.md` here.]*

Near-uniform distribution at ~10 examples per class is too little per-class data for fine-tuning to learn separate decision boundaries across 906 classes. Retrieval exploits a direct lexical queue where 84.0% of training questions contain a word from their own disease label, so TF-IDF retrieval use that overlap directly instead of learning it implicitly from class examples. The encoder seemed to compensate by memorising phrasing shared among sibling questions, hence it underperformed when the shared phrasing disappeared, and validation gave no hint of it.

### 4.2 Impact of Design Choices

The index variant was Arm 1's largest lever, and the shift-aware protocol is what identified it: the standard hold-out could not separate the four variants at all. That is the clearest lesson here — when a hold-out is drawn from paraphrase siblings, a shift-aware split is the better guide to generalisation, and the test results bear it out. No isolating ablation was re-run at the frozen vectoriser, so this remains consistent with the evidence rather than demonstrated by it.

The lesson has an unclaimed corollary. The ablations (Table 7) show `sublinear_tf=True` at 0.530 on the shift-aware split against the frozen 0.398 — over five standard errors — while the standard hold-out separates them by 2pp, inside its own SE. The tie-break was scoped to index selection, so it never reached the vectoriser grid; scoped wider it would have chosen `True`. Because this surfaced only after the test set had been read, it is left unchanged by choice rather than oversight: acting on it would be exactly the test-informed selection that pre-registering a rule exists to prevent. It is the first thing to change in a repeat.

**Table 7: Frozen-configuration vectoriser and index ablations — standard vs. shift-aware hold-out accuracy.** *[Insert table content from `report/tables/table5_ablations.md` here.]*

The rules differ in provenance: Arm 1's and Arm 3's were pre-registered before the numbers were seen, Arm 2's only after — disclosed rather than smoothed over. Both the Arm 2 and Arm 3 priors deliberately retained the lower-scoring option, the intended behaviour of a declared prior rather than a hidden accuracy claim.

### 4.3 Implications for Healthcare

85 identical question strings map to more than one disease, an irreducible error floor. A triage system here should abstain rather than guess: validation accuracy climbs from 0.850 to 0.945 once only the most-confident 36.5% of items are answered (Figure F7). Arm 3 shows the same principle accidentally, its only component that never hurt accuracy was the fallback that declined to re-rank. With the sharp degradation once phrasing shifts, patient-facing classifiers must be validated on genuinely independent data and deployed with a route to human review.

**Figure F7: Arm 1 accuracy–coverage curve (validation).** *Accuracy of retained predictions and fraction of items retained (coverage) as the nearest-neighbour similarity abstention threshold varies; the source for the 0.850→0.945 accuracy-at-36.5%-coverage figure above.* *[Insert `figures/fig7_coverage_curve.png` here.]*
