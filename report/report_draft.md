## Introduction

### 1.1 Literature Review

Natural language processing has become invaluable for many clinical tasks, from Electronic Health Records (EHR) phenotyping to patient communication systems. Patient question classification is a vital task that can support medical information retrieval, automated patient triage and clinical decision support systems.

Historically, healthcare NLP relied on rule-based methods or classical machine learning models, with Term Frequency - Inverse Document Frequency (TF – IDF) vectorization being a traditional baseline, paired with distance-based retrieval. Questions are represented as sparse TF-IDF vectors, and test questions are classified based on the labels most similar. Studies have demonstrated that TF-IDF baselines remain highly robust and computationally efficient despite struggling with semantic variation, synonyms, and new vocabulary.

Transition to deep learning has allowed models to capture context-aware semantic embeddings, allowing them to understand specialised medical terminology better than traditional approaches. Bidirectional Encoder Representations from Transformers (BERT) models fine-tuned on clinical corpora—such as BioBERT and ClinicalBERT have achieved state-of-the-art results on clinical text classification. The emergence of large language models (LLMs) has also revolutionised clinical NLP, allowing to perform complex text classification with effective prompting techniques at entirely inference time (Sander Schulhoff et al., 2024). However, fine-tuned encoders beat zero-shot LLMs on biomedical classification tasks (Chen et al., 2025).

To bridge the performance gap without resource-intensive parameter updates, for clinical tasks where specialised domain knowledge is missing from the pre-training corpus, Retrieval-Augmented Generation (RAG) architectures can be deployed to dynamically retrieve relevant clinical guidelines and prepend them as context, reducing model hallucinations and ensuring factually grounded responses.

\+ zero-/few-shot and chain-of-thought prompting, RAG.

### 1.2 Motivation and Rationale

The primary aim of this coursework is to evaluate and compare traditional lexical retrieval and deep general neural approaches on a 906-way medical condition classification task.

The comparison will be evaluated using classification accuracy and F1-score, and the implications for digital health in terms of safety, transparency, and computational resource requirements.

name the three arms. Retrieval will be a strong baseline because support is uniform at around 10 examples, and the encoder's advantage will be limited by that same sparsity.

state the contribution beyond running models: the near-duplication audit, the \*\*sibling-homogeneity, diagnosis of the validation–test gap\*\*, and paired significance testing.

## Methodology

### 2.1 Dataset Description

Dataset has been synthetically generated using ChatGPT, derived from the patient information section of the NHS UK website, and released as part of the OpenGPT dataset, after manual human validation.

The dataset covers 906 medical conditions, with multiple Q&A pairs for each. The training set covers all 906 conditions across 8,891 questions, with the validation set containing 102 diseases all also present in the training set, both containing no duplicates or nulls. 87.9% of the classes have exactly 10 Q&A pairs, with only 4 classes exceeding 10 and one class below 5, confirming a near-uniform distribution that requires no class weighting or resampling.

Average lengths of training and testing sets are 8.4 and 7.4 respectively – relatively similar and justify the 64 limit for BERT.

### 2.2 Data Preprocessing

Text was converted to features by two routes: scikit-learn's TfidfVectorizer for the retrieval arm and the Bio\_ClinicalBERT WordPiece tokeniser for the neural arm.

Only the question field is used as model input and reference\_url was dropped to avoid leaking the target because it directly maps onto the disease label. Labels were preserved, including the six non-lower-case, because the NHS document filenames match label casing exactly and the lookup runs on a case-sensitive filesystem. Support is near-uniform at roughly ten questions per class, so up-sampling, down-sampling or augmentation was not necessary.

Three vectoriser settings with stop-word removal and lemmatisation were searched over the grid:

- \`ngram\_range\`to select between indexing single words or also word pairs;
- \`min\_df\` to set the minimum number of documents a term must appear in to be kept, eventually set as min\_df=1 to avoid discarding rare class-identifying terms;
- \`sublinear\_tf\` to verify that replacing raw term count with is not effective for short questions (8.4 words) with limited repetitions.

Stop-word removal and lemmatization were run as separate ablations. Removal of common words in short questions averaging 8.4 words in length risked deleting a large meaningful proportion of the question. Additionally, because the text is lowercased, two clinical abbreviations – AS (ankylosing spondylitis) and ME (myalgic encephalomyelitis) would have been interpreted as stop words and discarded. Therefore, removal was considered lossy and redundant, as IDF already assigns near-zero weights to frequent terms. The ablation test shown in Table X **{TABLE REF}** demonstrated no benefit of lemmatisation and stop word removal, so these were not adopted.

Lowercasing, the tokenisation pattern (?u)\\b\\w\\w+\\b (which splits on non-word characters and discards single-character tokens – deemed safe for clinical text), smoothed inverse document frequency (IDF) weighting and Euclidean normalisation were left at their defaults.

For the neural model, questions were tokenised into WordPiece units and capped at 48 instead of default 512 tokens, which Bio\_ClinicalBERT splits into 44 tokens including the two special tokens, leaving four tokens of headroom. This adjustment allows to reduce the attention cost to comply with limited computational resources, since self-attention scales quadratically with sequence length. The measured truncation rate of 0% confirmed that no question was shortened.

The NHS reference documents required cleaning, because they form part of the search index (under the selected QLAD) and are scored at retrieval time. Navigation text, image-credit links and "Page last reviewed" footers, accounting for 4.96% of characters, were stripped to avoid diluting the documents containing them. Cleaning has cut the number of such terms present in over 95% of the documents from 22 to 13. However, no spelling correction, negation or missing values handling has been performed due to the clean nature of the documents. Other function words were handled by the Term Frequency – inverse document frequency (TF-IDF), which adjusts for disproportionate word frequency.

**Table 1. Preprocessing and index-construction ablations.**

| **Stage** | **Parameter** | **Value used** | **Library default** | **Changed** | **Source** |
| --- | --- | --- | --- | --- | --- |
| Case folding | lowercase | True | True | No | default |
| Tokenisation | token\_pattern | (?u)\\b\\w\\w+\\b | (?u)\\b\\w\\w+\\b | No | default |
| N-grams | ngram\_range | (1, 1) | (1, 1) | No | grid |
| Stop words | stop\_words | None | None | No | grid |
| DF floor | min\_df | 1 | 1 | No | grid |
| DF ceiling | max\_df | 1.0 | 1.0 | No | fixed |
| Term weighting | sublinear\_tf | False | False | No | grid |
| IDF | use\_idf | True | True | No | default |
| IDF smoothing | smooth\_idf | True | True | No | default |
| Normalisation | norm | l2 | l2 | No | default |
| Lemmatisation | spaCy model / enabled | en\_core\_web\_sm / disabled | n/a (not a TfidfVectorizer parameter) | No | ablation |
| BERT tokenisation | WordPiece: max\_length/padding/truncation | not frozen -- config.py HYPERPARAMETERS.bert\_model\_name and max\_length are None as of this run (Arm 2 not yet run/selected) | model\_max\_length=512 (BERT-family default) | \- | pending |

Settings applied to the TF–IDF retrieval arm (scikit-learn TfidfVectorizer) and the neural arm (Bio\_ClinicalBERT WordPiece tokeniser), shown against the corresponding library defaults. Source indicates how each value was arrived at: selected by grid search, resolved by ablation (Table X), fixed a priori, or left at the library default.

**Table X. Preprocessing and index-construction ablations.**

| **Variant** | **Hold-out acc** | **Shift-aware acc** | **Δ vs frozen (shift-aware)** | **Vocab size** | **Test acc (frozen row only)** |
| --- | --- | --- | --- | --- | --- |
| Frozen configuration (baseline) | 0.785 | 0.180 | 0.000 (baseline) | 4257 | not run |
| ngram\_range=(1,2) | 0.745 | 0.205 | +0.025 | 29181 | — |
| min\_df=2 | 0.490 | 0.077 | \-0.102 | 2052 | — |
| sublinear\_tf=True | 0.790 | 0.225 | +0.045 | 4257 | — |
| stop\_words='english' | 0.765 | 0.182 | +0.003 | 4056 | — |
| Lemmatisation enabled (en\_core\_web\_sm) | 0.810 | 0.215 | +0.035 | 3472 | — |
| NHS documents uncleaned | 0.785 | 0.180 | +0.000 | 4257 | — |
| index\_variant=QL | 0.795 | 0.172 | \-0.008 | 4333 | — |
| index\_variant=QLA | 0.770 | 0.263 | +0.083 | 10924 | — |
| index\_variant=QLAD | 0.835 | 0.328 | +0.148 | 16671 | — |

### 2.3 Traditional NLP Approach

#### Method

First, TF–IDF vectorisation applied to the dataset, cosine k-nearest-neighbours similarity-weighted vote over neighbour labels selecting k (1) candidates with the highest sum of cosine similarities of neighbours.

\- Rationale for weighting over majority vote: at k\_neighbors=1 (frozen) the two collapse, but the same ranking function backs the depth-20 shortlist Arm 3 consumes, where weighting lets a single very close neighbour outrank several weak ones.

Four index variants have been tested:

- question text only (Q),
- +label text (QL),
- +training answers (QLA),
- +NHS documents (QLAD).

Training-side answers/documents are indexed as class evidence, so the only input at prediction time is the test question. This is not leakage since the documenta are not read at inference time and is knowledge-based construction. The NHS .txt documents are used as an external knowledge source.

Supervised baseline.

As a second traditional approach, LinearSVC, Logistic Regression, and Random Forest were fit directly on TF-IDF question vectors, testing a learned decision boundary against simple neighbour matching. On the standard hold-out set, LinearSVC reached 0.815 accuracy (0.905 accuracy at 5 **{at 5 WHAT?},** macro-averaged **{WHAT IS MACRO AVERAGED}** F1 0.671), Logistic Regression and Random Forest had accuracy of 0.765 and 0.635 respectively. The frozen retrieval model scored 0.850 - 3.5 percentage points above the best supervised model. This gap sits within the roughly 7-percentage-point noise band expected **{WHY IS IT ROUGHLY 7 EXPECTED?}** from a 200-item validation set, so the improvement is not statistically significant but sufficient to prefer retrieval.

#### Selecting the index variant and scheme.

On the standard hold-out, all four index variants tied at 0.820 accuracy. A follow-up validation split was used to rank them instead: QLAD (0.3275) led the next-best variant QLA (0.2625) by 0.065, more than its standard error, so was the selected index variant.

For the indexing scheme, the standard hold-out selected class\_blob (0.850) over additive\_per\_row (0.730), a margin of 0.120, which was confirmed by the follow-up split (0.3975 vs 0.325, a margin of 0.0725). Both margins exceeded their standard errors, informing the index\_scheme = "class\_blob" decision.

Switching to QLAD required re-tuning the vectoriser, since QLAD's indexed text is full NHS prose rather than short questions and changing the index variant required to switch the vectoriser. The re-tuned grid search selected an n-gram range of (1,2) — single words and word pairs — a minimum document frequency of 1, and no stop-word removal.

#### Bounding Arm 3

To measure how much room a later shortlist-based stage has to work with, how often the correct label appears within the top k retrieved candidates was checked: 0.85 at k=1, 0.98 at k=5, 0.99 at k=10, and 0.995 at k=20. The shortlist size was fixed at 20 as a declared setting rather than tuned further as there was little change.

### 2.4 Neural (LLM) Approach

\- Architecture & training setup — Bio\_ClinicalBERT (emilyalsentzer/Bio\_ClinicalBERT) with a 906-way classification head, AdamW optimiser, learning\_rate=2e-5, batch\_size=16, max\_length=48 (carried over from the §2.2 question-length justification). Source: config.py.

\- Epoch/checkpoint selection rule — standard hold-out, within-1-SE-prefer-fewest-epochs. Bio\_ClinicalBERT: epoch 22 reaches 0.870 accuracy but epoch 14 reaches 0.850 — a 2pp gap, inside the ~3.5pp SE for n=200 — so the cheaper checkpoint is kept: num\_epochs=15, val accuracy 0.850 (arm2\_val\_metrics.csv, arm2\_history\_bioclinicalbert.csv).

\- Encoder comparison, the tie-break in action. Bio\_ClinicalBERT (0.850) vs bert-base-uncased (0.875, its own selected checkpoint at epoch 19) — bert-base scores 2.5pp higher. McNemar's exact test over the 200 paired predictions: 6 items only Bio\_ClinicalBERT got right, 11 only bert-base got right, 17 discordant total, p = 0.332 (arm2\_encoder\_mcnemar.csv). Since p ≥ 0.05, the comparison is unresolved — Bio\_ClinicalBERT is kept on the declared clinical-domain prior, not because it scored higher. State plainly it did not.

\- Disclosure sentence required here (per CLAUDE.md): this rule was formalised on 2026-08-17, after the comparison was seen — not pre-registered like the Arm 1 tie-break. Say so explicitly; don't let it read as if it were decided in advance.

\- Loss/accuracy curves — both full 24-epoch histories exist (arm2\_history\_bioclinicalbert.csv, arm2\_history\_bertbase.csv), so Figure F6 (train/val loss curves, "explicitly required" by the brief) is plottable now, even though it's filed under §3.2 in the figure inventory.

### 2.5 Experimental Protocol

Stratified splitting via train\_test\_split is infeasible here: scikit-learn requires at least one held-out example per class, but the label space (906) exceeds the target validation size (200). An unstratified split leaves the number of represented classes to chance (177 under seed 42) and provides no guarantee that each class retains sufficient examples for fitting. We therefore allocate a fixed quota across a sampled subset of classes, mirroring the test set's size and class count while guaranteeing every class retains at least two training examples.

Results

3.1 Exploratory Data Analysis

Two representations tested: sibling homogeneity (**why chosen?)** and lexeme (**why chosen?),** median max cosine distance was 0.485 with only 0.5% above 0.9, while under a unigram representation the classifier uses 0.613.

Training questions are lexical siblings, so sibling homogeneity used. In the training, the nearest same-class sibling was 0.572, while in the test the nearest own-class training question had the score of 0.391, while the nearest other-class of 0.510, an increase of 0.119. **{Insert conclusion sentence}**

As a robustness check and to investigate the nature of the data, it was verified that 84.0% of training questions contain a word from their own disease label, while only 5.5% of test questions do.

Therefore, a wrong-class training question is closer than any own-class one for **59.5%** of test questions, against **33.8%** under training leave-one-out. Nearest-neighbour decision rule is therefore disadvantaged on this test distribution, which will motivate the shift-aware selection protocol described in section 2.5.

3.2 Performance Comparison

3.3 Error Analysis

Discussion

4.1 Comparison of Approaches

4.2 Impact of Design Choices

4.3 Implications for Healthcare

Notes:

Truncation- robust split – already in section 2.5. Diverges from course so needs justification.

Run the unstratified train\_test\_split too so that both can be reported in the ablation table — one extra row. It demonstrates knowledge of the standard module tool, shows the choice was tested rather than assumed, and if the two produce similar hyperparameter rankings that's evidence the selection is robust to split design.

**A methodological problem to settle before the EDA notebook**

The sibling-homogeneity measurement and the acronym analysis both use **test labels** — computing "test question → nearest own-class training question" requires knowing each test question's class. Under a strict reading of "the test set is evaluated once," putting those in 01\_eda breaks the rule I wrote into CLAUDE.md.

Two defensible resolutions:

1. **Compute in 01\_eda, declare it.** Add to §2.5: _"Test labels were used for one distributional diagnostic characterising the validation–test relationship. No model, hyperparameter, preprocessing setting or index variant was selected using test data."_ Honest, and the finding shapes how you read every validation number thereafter.
