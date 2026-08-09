# Report Outline — AMLH Coursework, Patient Question Classification

**Target: 2,000–2,500 words** excluding figures, tables, captions and references.
Write to 2,400 and trim.

> **Revised after auditing the real data.** The dataset is *not* long-tailed — support is
> near-uniform at ~10 questions per class. Do not describe it as long-tailed anywhere.

---

## Established facts to build the report around

| Finding | Value | Where it lands |
|---|---|---|
| Class support | min 4 / median 10 / max 20, sd 0.98 | §2.1 — few-shot, not imbalanced |
| Label families | 355/906 share a prefix (`baby_` 53, `pregnancy_` 24, `social_` 23, `cosmetic_` 20) | §2.1, §3.3 |
| Ambiguous questions | 85 identical strings map to >1 disease | §3.3, §4.3 |
| Train–test duplication | 0.5% above 0.9 cosine; median max 0.485 | §3.1 — validity check passed |
| Question length | mean 8.4 words, 99th pct 18 | §2.2 — justifies `max_length=48` |
| Sibling homogeneity | train→sibling 0.572 vs test→own-class 0.391 (test→other-class 0.510) | §3.1, §4.2 — **the centrepiece** |
| Wrong class closer | 59.5% of test questions | §3.1 |
| Validation→test gap | ~0.77 vs ~0.40 (Arm 1) | §2.5, §4.2 |

---

## 1. Introduction — 350 words · 10 marks

### 1.1 Literature Review (~220 w)
1. **Clinical stakes.** Patient question classification underpins triage and information
   retrieval; the cost of error is asymmetric — an unanswered question is recoverable, a
   confidently wrong condition label is not.
2. **Classical retrieval.** TF–IDF term weighting, BM25, question entailment for consumer
   health questions.
3. **Transformer-era supervised classification.** BERT and biomedical/clinical variants; the
   general finding that in-domain pretraining helps.
4. **The LLM turn.** In-context learning, zero-/few-shot and chain-of-thought prompting, RAG.
   Cite the benchmarking evidence that fine-tuned encoders still beat zero-shot LLMs on
   biomedical classification (Chen et al., 2025, reproduced in the Week 9/10 slides).

### 1.2 Motivation and Rationale (~130 w)
State the problem formally (906-way classification from one short question), name the three arms,
and commit to a hypothesis: *retrieval will be a strong baseline because support is uniform at
~10 examples, and the encoder's advantage will be limited by that same sparsity.* Then state the
contribution beyond running models — the near-duplication audit, the **sibling-homogeneity
diagnosis of the validation–test gap**, and paired significance testing.

---

## 2. Methodology — 15 + 40 marks

### 2.1 Dataset Description (~180 w)
- Provenance: NHS.UK content, synthetically expanded via ChatGPT (OpenGPT, CogStack), test set
  manually validated.
- **Say `answer` is excluded from all inference paths and why.**
- Descriptive statistics from the table above. Characterise as **extreme multi-class, few-shot,
  fine-grained** — and note the label families explicitly, since they drive the error analysis.

### 2.2 Data Preprocessing (~270 w)
Contains the **required preprocessing table** — chosen settings *and* library defaults:

| Stage | Setting | Value used | sklearn default |
|---|---|---|---|
| Case folding | `lowercase` | True | True |
| Tokenisation | `token_pattern` | `(?u)\b\w\w+\b` | same |
| N-grams | `ngram_range` | *from grid* | (1,1) |
| Stop words | `stop_words` | *from ablation* | None |
| DF floor / ceiling | `min_df` / `max_df` | *from grid* / 1.0 | 1 / 1.0 |
| Term weighting | `sublinear_tf` | *from grid* | False |
| IDF | `use_idf` / `smooth_idf` | True / True | True / True |
| Normalisation | `norm` | l2 | l2 |
| Lemmatisation | spaCy `en_core_web_sm` | *from ablation* | n/a |
| BERT tokenisation | WordPiece, `max_length=48`, pad + truncate | | 512 |

Three things must be **argued**, not stated:

- **Lemmatisation and stop-word removal are an ablation.** For short questions, stop-word removal
  can destroy interrogative signal ("how long does X last"). Report the numbers.
- **`max_length=48`** from the length percentiles: covers the 99th percentile, ~10× cheaper than
  512, and the reason the neural arm fits a free T4.
- **NHS document cleaning.** Boilerplate ("Skip to main content", Alamy credit URLs, "Page last
  reviewed") is ~22% of characters. Also note six labels break the naming convention
  (`Bronchitis`, `Multiple_sclerosis`, `Pneumonia`, `Tonsillitis`, `Bronchiolitis`, `Laryngitis`)
  and are resolved by lowercasing the filename lookup.

### 2.3 Traditional NLP Approach (~200 w)
- TF–IDF → cosine k-NN → **similarity-weighted** vote; say why weighted rather than majority.
- Index variants Q / Q+L / Q+L+A / Q+L+A+D.
- **The `answer` argument, stated plainly:** including *training* reference answers adds no
  test-side information and the only inference input remains the test question, so it is
  knowledge-base indexing rather than leakage. Validation supported inclusion
  (0.765 → 0.790 → 0.810). A marker will wonder if you don't address it.
- Second variant: TF–IDF + LinearSVC/LogReg, contrasting learned boundaries against neighbour
  matching under ~10 examples per class.
- **Figure: Arm 1 workflow diagram** (required).

### 2.4 Neural Approach (~250 w)
- **Arm 2**: Bio_ClinicalBERT + 906-way head; AdamW; selection on validation accuracy; ablation
  against `bert-base-uncased`.
- **Arm 3**: TF–IDF shortlists 20 candidates, LLM selects. Say why not all 906 (~3k tokens,
  degraded selection). Three prompt conditions; constrained output parsing with retriever
  fallback, and report the fallback rate. **Report the shortlist ceiling** (Arm 1 acc@20) — it
  bounds Arm 3 from above and makes its numbers interpretable.
- **Figure: Arms 2–3 workflow diagram**.

### 2.5 Experimental Protocol (~250 w) — the highest-value section
- **Protocol: single stratified hold-out** (200 items / 102 classes), mirroring the test set's
  decision load. Follows the Week 9 practical.
- **State its limitation and why you chose it anyway.** Roughly this: *"Because the ~10 questions
  per disease were generated in one pass, a held-out validation question is a sibling of those
  remaining in training (cosine 0.572) whereas a test question is not (0.391). Hold-out validation
  therefore over-estimates test accuracy. Cross-validation would not remedy this, since siblings
  remain within every training fold; the bias follows from data generation, not from slicing. We
  retain the hold-out for hyperparameter selection — the ranking of configurations transferred to
  the test set — and report a novelty-calibrated estimate as a secondary check."*
- **Note the noise floor:** 200 validation items give SE ≈ 3.5pp, so grid differences below ~7pp
  are not meaningful. Say this once; it protects every claim that follows.
- **Test set used exactly once**, after hyperparameters were frozen. Note: Test labels were used for one distributional diagnostic characterising the validation–test relationship; no model, hyperparameter, preprocessing setting or index variant was selected using test data.
- Hyperparameter table; prompt table (verbatim); compute (Colab T4, wall-clock, peak memory);
  reproducibility (seed 42, pinned versions, persisted artefacts).
- Truncation- robust split – already in section 2.5. Diverges from course so needs justification.
Run the unstratified train_test_split too so that both can be reported in the ablation table — one extra row. It demonstrates knowledge of the standard module tool, shows the choice was tested rather than assumed, and if the two produce similar hyperparameter rankings that's evidence the selection is robust to split design.


---

## 3. Results — 600 words · 25 marks

### 3.1 Exploratory Data Analysis (~120 w)
Lead with three findings: near-uniform support, label-family granularity, and the
**near-duplication check passing**. Then the novelty-distribution figure and the
sibling-homogeneity table — this is the report's strongest original content.

> State the fitting corpus for reproducibility: the diagnostic TF-IDF vectorisers behind the
> near-duplication and sibling-homogeneity figures (`data.run_integrity_audit`,
> `eda.sibling_homogeneity`) are fit on **training questions only**. Fitting the same vectoriser
> including test-side text instead shifts the measured cosine similarities by +0.035 (own-class)
> and +0.057 (other-class) — an unstated fitting corpus makes these figures unreproducible.

### 3.2 Performance Comparison (~300 w)

| System | Accuracy | 95% CI | Macro-F1 | s/query |
|---|---|---|---|---|

Plus ablation tables (grid, preprocessing, index variant, BERT model, prompt condition), the
**training/validation loss curves** (explicitly required), the accuracy–coverage curve, the
novelty-calibration table as a secondary estimate, and **McNemar** results.

Phrase every comparison carefully: *"X exceeded Y by N pp; McNemar gives p = …, so the difference
is / is not statistically reliable at n=200."* That sentence pattern is what separates a strong
report from a list of numbers.

### 3.3 Error Analysis (~180 w)
- Most-confused class pairs, with the justification for not printing a 906×906 matrix.
- **Percentage of errors falling inside the same label family** — granularity artefact versus
  genuine clinical confusion. This is the most informative single number in the section.
- **Two correct and two incorrect examples per arm, quoted verbatim.** Explicitly required.
- The rejected acronym hypothesis, in two sentences: acronym-bearing test questions were *easier*
  (0.69 vs 0.27) and post-stratifying validation to the test marginals moved the estimate the
  wrong way, so surface features do not explain the gap.

---

## 4. Discussion and Conclusion — 400 words · 10 marks

### 4.1 Comparison of Approaches (~150 w)
Predictive performance, computational cost, implementation effort — using measured
seconds-per-query and GPU memory, not adjectives. If the traditional arm proves competitive at a
fraction of the cost, say so directly.

### 4.2 Impact of Design Choices (~150 w)
Preprocessing ablation, `max_length`, index variant, shortlist size, prompt condition, and the
loss-curve gap as evidence of memorisation under ~10 examples per class. Then the headline
limitation: **synthetic training data and expert-validated test data are not drawn from the same
distribution**, and the sibling measurement quantifies it. Any model tuned on synthetic siblings
will be optimistically evaluated.

### 4.3 Implications for Healthcare (~100 w)
Abstention as a safety mechanism, tied to the coverage curve. Label ambiguity means some questions
have no single correct answer — 85 identical question strings carry different labels — so a
top-1-only interface misrepresents the system's certainty. Add automation bias, human oversight,
and the fact that an NHS-derived synthetic corpus encodes one health system's phrasing.

Close with two or three next steps: sentence-embedding bi-encoder retrieval, cross-encoder
reranking, hierarchical classification exploiting label families, LoRA/QLoRA fine-tuning
(Week 10), calibration.

---

## Figure and table inventory

| # | Item | Section |
|---|---|---|
| T1 | Preprocessing settings and defaults | 2.2 |
| T2 | Hyperparameters and prompts | 2.5 |
| T3 | Sibling-homogeneity measurements | 3.1 |
| T4 | Arm 1 grid search | 3.2 |
| T5 | Preprocessing and index ablations | 3.2 |
| T6 | Main test results with CIs | 3.2 |
| T7 | McNemar comparisons | 3.2 |
| T8 | Novelty-calibration (secondary estimate) | 3.2 |
| F1 | Four-panel EDA | 3.1 |
| F2 | Novelty distributions | 3.1 |
| F3 | PCA of TF-IDF vectors | 3.1 |
| F4 | Arm 1 workflow diagram | 2.3 |
| F5 | Arms 2–3 workflow diagram | 2.4 |
| F6 | Training / validation loss curves | 3.2 |
| F7 | Accuracy–coverage curve | 3.2 |
| F8 | Most frequent confusions | 3.3 |

---

## Reference shortlist (verify every one before citing)

From the brief: Dubois et al. (2023) AlpacaFarm; Kweon et al. (2024) Asclepius; Schulhoff et al.
(2024) The Prompt Report.

Cited in the Week 7–10 slides, so safe and course-aligned: Devlin et al. (2019) BERT;
Wei et al. (2022) Chain-of-Thought Prompting; Kojima et al. (2022) Zero-Shot Reasoners;
Hu et al. (2022) LoRA; Dettmers et al. (2023) QLoRA; Chen et al. (2025) *Nature Communications*.

Worth adding — **check each on PubMed or Semantic Scholar first**: Alsentzer et al. (2019)
clinical BERT embeddings; Lee et al. (2020) BioBERT; Gu et al. (2021) domain-specific biomedical
pretraining; Reimers & Gurevych (2019) Sentence-BERT; Robertson & Zaragoza (2009) BM25;
Lewis et al. (2020) RAG; Ben Abacha & Demner-Fushman (2019) question entailment;
Singhal et al. (2023) Med-PaLM.

**Do not cite anything you have not opened.**

---

## Writing order

1. §2.1 / §2.2 — straight after Phase 1, while the EDA numbers are final.
2. §2.3 / §2.4 / §2.5 — as you implement; the protocol table fills itself in.
3. §3 — only after the single frozen test run.
4. §1 and §4 — last, so the introduction promises exactly what the results deliver.

---
