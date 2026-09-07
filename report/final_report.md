# AMLH 2026 - NLP Coursework: Patient Question Classification

## Introduction

### 1.1 Literature Review

Natural language processing (NLP) has become invaluable for many clinical tasks, from Electronic Health Records (EHR) phenotyping to patient communication systems. Patient question classification supports medical information retrieval, automated patient triage and clinical decision support.

Historically, healthcare NLP relied on rule-based methods or classical machine learning, with TF-IDF vectorisation a traditional baseline, paired with distance-based retrieval: questions are represented as sparse TF-IDF vectors and classified by most similar labels. TF-IDF baselines remain robust and computationally efficient despite struggling with semantic variation, synonyms and novel vocabulary.

Deep learning captures contextual embeddings, representing specialised terminology better than lexical methods: BERT (Devlin et al., 2019) models fine-tuned on clinical corpora, like BioBERT (Lee et al., 2020) and Bio\_ClinicalBERT (Alsentzer et al., 2019), achieve state-of-the-art clinical text classification. Large language models (LLMs) have enabled in-context learning, classifying purely through prompting, with zero-shot, few-shot and chain-of-thought strategies for supplying context without any parameter update (Schulhoff et al., 2024). Fine-tuned encoders nonetheless beat zero-shot LLMs on biomedical classification (Chen et al., 2025), motivating retrieval-augmented generation (RAG), which adds clinical text as context to improve accuracy and reduce hallucination.

### 1.2 Motivation and Rationale

The primary aim is to evaluate traditional lexical retrieval against deep neural approaches on a 906-class medical condition classification task, using accuracy and macro-F1, and considering implications for digital health safety, transparency and compute cost.

Three arms implement this comparison:

-   Arm 1, TF-IDF vectorisation with cosine k-nearest-neighbour retrieval over a class-level index;
-   Arm 2, Bio\_ClinicalBERT fine-tuned as a 906-way classifier;
-   Arm 3, TF-IDF retrieval shortlisting candidates for an LLM to re-rank by prompting.

Class support is near-uniform at around 10 examples per class for most of the 906 classes, which is too sparse for Arm 2 to learn separate per-class decision boundaries, which is expected to favour the retrieval-based arms (1 and 3) over the fine-tuned encoder.

## Methodology

### 2.1 Dataset Description

The dataset was synthetically generated using ChatGPT from the NHS UK website's patient-information section, released as part of the OpenGPT dataset (CogStack, 2023) after manual human validation.

It covers 906 conditions with multiple Q&A pairs each, across 8,891 questions. The provided test set covers 200 questions over 102 of those conditions, all present in training. It is distinct from the internally constructed validation split, taken from training and matching the test set's size and class count. Neither contains null values, exact duplicate rows or repeated question-disease pairs. 87.9% of training classes have exactly 10 pairs, requiring no resampling due to near-uniform distribution; the test set averages 1.96 questions across 102 classes. Questions are short, averaging 8.4 words in training, 7.4 in test.

### 2.2 Data Preprocessing

Text was converted using TfidfVectorizer for retrieval and Bio\_ClinicalBERT WordPiece tokeniser for the neural arm. Only the question field is used as model input and reference\_url was dropped since it maps directly to the disease label. Case-sensitive labels were preserved since NHS filenames match label casing. Setting min\_df=1 avoids discarding rare class-identifying terms (Table 1).

Stop-word removal and lemmatisation were tested as ablations and rejected. Removal isn't safe due to the short question length (8.4), and because the text is lowercased, two of the 155 clinical acronyms in the training questions - AS (ankylosing spondylitis) and ME (myalgic encephalomyelitis) appear in scikit-learn's stop-word list and would be discarded, IDF already down-weights frequent terms.

Questions were capped at max\_length=48 rather than BERT's default 512 because the longest is 44 tokens including special tokens, and truncation rate at 48 tokens is 0%, allowing computational costs to be reduced safely.

NHS documents were cleaned before indexing, since for QLAD they are scored at retrieval time. Navigation text, image-credit links and "page last reviewed" footers made up 4.96% of characters and removing them reduced the terms present in over 95% of documents from 22 to 13.

**Table 1: Preprocessing and index-construction ablations.**

**Stage**

**Parameter**

**Value used**

**Library default**

**Changed**

**Source**

Case folding

lowercase

True

True

No

default

Tokenisation

token\_pattern

(?u)\\b\\w\\w+\\b

(?u)\\b\\w\\w+\\b

No

default

N-grams

ngram\_range

(1, 2)

(1, 1)

Yes

grid

Stop words

stop\_words

None

None

No

grid

DF floor

min\_df

1

1

No

grid

DF ceiling

max\_df

1.0

1.0

No

fixed

Term weighting

sublinear\_tf

False

False

No

grid

IDF

use\_idf

True

True

No

default

IDF smoothing

smooth\_idf

True

True

No

default

Normalisation

norm

l2

l2

No

default

Lemmatisation

spaCy model / enabled

en\_core\_web\_sm / disabled

n/a (not a TfidfVectorizer parameter)

No

ablation

BERT tokenisation

WordPiece: model / max\_length/padding

Bio\_ClinicalBERT / 48, pad to max\_length

model\_max\_length=512 (default)

Yes

grid

Source indicates how each value was chosen: grid search, kept default, or by ablation (Table 8).

### 2.3 Traditional NLP Approach

TF-IDF vectorisation is followed by a cosine nearest-neighbour classifier over a class-level index: each of the 906 classes is one document, concatenating all text assigned to it (label, training questions, answers and the NHS page, depending on the variant below), rather than one document per training question. No class can then appear twice in the ranking, so similarity-weighted voting over any k returns the same label as k=1, and prediction reduces to the closest class by cosine similarity (Figure 1). Its top 20 form Arm 3's shortlist.

**Figure 1: Arm 1 - TF-IDF retrieval over a QLAD class-blob index.** 

Pipeline diagram: split\_fit questions and the 906 NHS documents were indexed under the frozen QLAD/class\_blob/ngram(1,2) configuration. The test question is the only inference-time input, producing a top 1 prediction and a top 20 shortlist for Arm 3. Green blocks mark settings frozen in config.py.

Four index variants were tested:

-   question text only (Q);
-   +label text (QL);
-   +training answers (QLA);
-   +NHS documents (QLAD). Training-side answers and documents are indexed only as class evidence and prediction-time input remains the test question alone, so this is knowledge-base construction, not leakage.

LinearSVC, Random Forest and Logistic Regression were fit directly on TF-IDF vectors as supervised baselines, scoring 0.815, 0.765 and 0.635. They did not beat retrieval's 0.850 so neighbour matching was preferred.

NHS prose index text forced a vectoriser re-tune, selecting ngram\_range=(1,2), min\_df=1 and no stop-word removal. Shortlist depth was fixed at 20. This limits Arm 3's achievable accuracy at the fraction of questions whose true label appears somewhere in the top 20 candidates, measured at the more conservative 0.975 on test, rather than the 0.995 seen on validation.

### 2.4 Neural Approaches

**Arm 2: fine-tuned BERT classifier**

Bio\_ClinicalBERT was fine-tuned with a 906-way head (AdamW, learning\_rate=2e-5, batch\_size=16, max\_length=48). Learning rate and batch size were left at standard fine-tuning values across both encoders, since at 3.5 percentage points (pp) standard error (SE) a 200-item hold-out could not resolve the difference. Epoch selection takes the fewest epochs whose validation accuracy is within 1 SE of the best (Figure 2; epochs are numbered from zero throughout, as on the figure's axis). The best accuracy was 0.870 at epoch 22 but epoch 14 already reached 0.850, inside that margin, so the earlier and cheaper checkpoint was kept at 15 epochs of training in total.

**Figure 2: Arm 2 training/validation loss.**

Cross-entropy loss by epoch for Bio\_ClinicalBERT and bert-base-uncased. The checkpoint was chosen on validation accuracy, not on loss convergence: validation loss was still falling at the selected epoch 14 but choice comes from accepting an accuracy within 1 SE of the best.

Bio\_ClinicalBERT (0.850) was compared against bert-base-uncased (0.875) by McNemar: 6 items only the first got right, 11 only the latter, p=0.332. The comparison failed to resolve the selection, so Bio\_ClinicalBERT was kept as the domain-specific model.

**Arm 3: LLM shortlist re-ranking**

Arm 3 is a RAG pipeline that reuses the frozen Arm 1 index to select 20 candidate labels per question and prompts an LLM to pick one, under three conditions:

-   zero-shot (candidates only);
-   few-shot (with two worked examples);
-   and chain-of-thought (COT) (with a reasoning instruction).

The reply was matched with a candidate by name and Arm 1's top 1 was used as a safe fallback if no match was found.

**Figure 3: Arm 2 and Arm 3 workflow diagrams.**

Arm 3 is dependent on Arm 1 as it re-ranks Arm 1’s shortlist and returns Arm 1’s own top 1 as a fallback if parsing finds no candidate. Therefore, its accuracy closely follows Arm 1’s accuracy. Green shading signifies items frozen in config.py.

Two generators were compared at temperature 0: primary microsoft/MediPhi-Guidelines and secondary google/flan-t5-large. Rather than testing all six prompt-by-model combinations at once, selection ran in two stages fixed in advance: the three prompt conditions (Table 2) were first compared on the primary generator alone by McNemar's test, then the two generators were compared at that fixed condition. Splitting the choice this way improves clarity and avoids the extra false-positive risk of six simultaneous pairwise comparisons on a 200-item hold-out.

**Table 2: Prompts used for Arm 3, comparing zero shot, few shot and COT approaches.**

**Condition**

**Template**

**Prompt length (tokens)**

**Decoding**

zero shot 

**(chosen)**

You are a helpful assistant trained to identify relevant medical question topics.  
Given the question below, which one of the following diagnoses is most relevant?  
  
Diagnoses:  
<20 retrieved candidate labels>  
Question: <test question>  
Respond with only the diagnosis name.

median 214, p95 345, max 372; truncated at 512: 0%

T=0, max\_new\_tokens=20

few shot

You are a helpful assistant trained to identify relevant medical question topics.  
Given the question below, which one of the following diagnoses is most relevant?  
  
Diagnoses:  
<20 retrieved candidate labels>  
Worked examples:

  
Question: <example question>  
Diagnosis: jellyfish and other sea creature stings  
  
Question: <example question>  
Diagnosis: amyloidosis  
  
Respond with only the diagnosis name.

median 279, p95 410, max 437; truncated at 512: 0%

T=0, max\_new\_tokens=20

cot

You are a helpful assistant trained to identify relevant medical question topics.  
Given the question below, which one of the following diagnoses is most relevant?  
  
Diagnoses:  
<20 retrieved candidate labels>  
Question: <test question>  
  
Think step by step about which diagnosis the question is asking about, then end your reply with 'Final answer:' followed by the diagnosis name.

median 236, p95 367, max 394; truncated at 512: 0%

T=0, max\_new\_tokens=128

### 2.5 Experimental Protocol

Stratified splitting by train\_test\_split is infeasible, because it requires one held-out example per class, but the 906-class label space exceeds the 200-item validation size, and an unstratified split can result in non-uniform class representation (177 classes under seed 42). Therefore quota-based sampling was used, mirroring the test set's size while keeping at least two training examples per class.

At n=200 the single-proportion SE is ≈3.5pp, so differences below 7pp (2SE) are treated as noise. Every comparison was instead checked with McNemar's test, which only uses the questions where the systems disagreed, since paired predictions on the same items are more informative than two independent ones. Accuracy is reported with bootstrap 95% CIs (Table 3).

A shift-aware protocol was used to resolve ties which the standard hold-out failed. A hold-out was drawn only from training questions sharing no words with their own label (1,423 of 8,891), because the test distribution more closely resembles this sample, in which the question never names its own label. The standard hold-out decided between candidates if the gap between them exceeded one SE, otherwise the shift-aware hold-out's ranking was used to break the tie. This was used for Arm 1 index selection only, as discussed in 4.2.

Validation informed every model, hyperparameter, and prompt decision, and test predictions were generated only once after freezing.

## Results

### 3.1 Exploratory Data Analysis

Class support, question length and label-family structure are summarised in Figure 4. Near-duplication and sibling homogeneity use different representations, with bigrams detecting duplicate phrasing, unigrams the within-class comparison. Median per-test-question maximum similarity to any training question is 0.485, with only 0.5% above the 0.9 near-duplicate threshold, rejecting near-duplication as an explanation for the results.

**Figure 4: Dataset composition and length/duplication diagnostics.** 

(a) Training questions per class are near uniform (min 4, median 10, max 20) except a small left tail.

(b) Train vs. test question-length distributions in words.

(c) The twelve largest label-prefix families.

(d) Maximum cosine similarity from each test question to any training question, with the 0.9 near-duplicate threshold marked with the dashed line.

Training questions are lexical siblings (Figure 5), where the nearest same-class sibling scores 0.572. The test question's nearest own-class training question scores only 0.391 and its nearest other-class question scores 0.510 (0.119 difference). 84.0% of training questions, but only 5.5% of test questions, contain a word from their own disease label.

**Figure 5: Sibling phrasing homogeneity chart to explain the val-test gap.**

Cosine-similarity distributions: each training question to its nearest same-class sibling, each test question to its nearest own-class training question, and each test question to its nearest other-class training question — the basis for the shift-aware validation protocol.

Consequently, for 59.5% of test questions a wrong-class training question is closer than any own-class one. A nearest-neighbour rule over training questions alone is therefore unreliable on the test distribution, which is why Arm 1 indexes NHS document text at class level rather than the training questions by themselves. It also means validation estimate drawn from siblings will be optimistic, supporting the shift-aware protocol.

### 3.2 Performance Comparison

Table 3 shows each arm's validation and test accuracy with bootstrap confidence intervals, and Figure 6 plots the test figures side by side.

**Table 3: Validation and test accuracy by arm, with bootstrap 95% confidence intervals, optimism, and macro-F1.**

**System**

**Validation**

**Test**

**95% CI (test)**

**Optimism**

**Macro-F1 (test)**

Arm 1 (TF–IDF k-NN)

0.850

0.765

\[0.705, 0.820\]

8.5pp

0.563

Arm 2 (Bio\_ClinicalBERT)

0.850

0.615

\[0.545, 0.680\]

23.5pp

0.354

Arm 3 (shortlist + LLM)

0.720

0.720

\[0.660, 0.780\]

0.0pp

0.494

Optimism is validation accuracy minus test accuracy. Macro-F1 is the unweighted average of each class's F1-score, which demonstrates the model's true performance across all categories, including minority classes.

All four index variants scored 0.820 on the standard hold-out, so the shift-aware split was the tiebreaker, ranking QLAD (0.328) over QLA (0.263). Those two figures are from the grid run before the vectoriser was re-tuned, so they are not comparable with the frozen-configuration ablations in Table 8. In the generator comparison via McNemar's test MediPhi (0.720 validation accuracy) beat flan-t5-large (0.635) (p=0.033).

Prompt design was the first of Arm 3's two selection stages, and the one where accuracy is least informative (Table 4). No condition separated from another on the primary generator, McNemar's smallest p being 0.511 for chain-of-thought against few-shot, so under the pre-registered rule the comparison is reported as unresolved and zero-shot was retained on the declared prior of the simplest prompt. That prior kept the lower-scoring condition, since few-shot reached 0.730 against zero-shot's 0.720. Accuracy did not choose the prompt.

**Table 4: Arm 3 prompt conditions on the primary generator (MediPhi-Guidelines, validation, n=200).**

**Condition**

**Accuracy**

**Fallback rate**

**Override rate**

**Override precision**

**Wall clock**

zero-shot **(chosen)**

0.720

0.160

0.250 (50/200)

0.120 (6/50)

78 s

few-shot

0.730

0.180

0.215 (43/200)

0.093 (4/43)

84 s

chain-of-thought

0.705

0.130

0.310 (62/200)

0.145 (9/62)

759 s

Override rate is the share of items where the LLM chose a candidate other than Arm 1's top 1; override precision is the share of those overrides that rescued an Arm 1 error. Kept and fallback items both return Arm 1's top 1 unchanged, so every accuracy difference between conditions comes from the overrides. Wall clock covers generation over the 200 items.

What the prompt changes is how often the LLM overrides Arm 1's top candidate: 21.5% of items under few-shot, 25.0% under zero-shot and 31.0% under chain-of-thought. Override precision stays between 0.09 and 0.15 across all three, so the extra interventions are mostly wrong and accuracy falls as the override rate rises. Prompting for reasoning made the model more willing to depart from the retrieval ranking rather than better at judging when it should, which is why chain-of-thought is the weakest condition despite reasoning being the intervention most often expected to help. It is also 9.7 times slower at 759 s against 78 s, needing max\_new\_tokens=128 rather than 20, for no accuracy return.

Fallbacks, where the reply matched no candidate so Arm 1's top choice stood, fired on 13% to 18% of items. Few-shot, whose two worked examples exist mainly to demonstrate the expected output format, produced the highest rate of the three, so exemplars did not enforce format compliance. No fallback output in any of the six prompt-by-generator cells matched any of the 906 labels, so the parser behaved as specified and was left untouched. However, 14 of zero-shot's 32 fallbacks are shortened renderings of a candidate that was in that item's own prompt, such as "skin lightening" for cosmetic\_procedures\_non\_surgical\_cosmetic\_procedures\_skin\_lightening. The failure is one of output format rather than of medical knowledge, which is why §4.4 proposes constraining decoding to the candidate strings instead of relying on a fallback.

**Figure 6: Test accuracy by arm.** 

Test-set accuracy (200 items) with bootstrap 95% confidence intervals for each arm.

**Table 5: Pairwise McNemar comparisons between arms, validation vs. test.**

**Pair**

**Validation**

**Test**

Arm 1 vs Arm 2

15/15, p=1.000, unresolved

37/7, p=5.3e-06, Arm 1 wins

Arm 1 vs Arm 3

32/6, p=2.4e-05, Arm 1 wins

20/11, p=0.150, unresolved

Arm 2 vs Arm 3

33/7, p=4.2e-05, Arm 2 wins

11/32, p=0.0019, Arm 3 wins

  
Every ranking that seemed reliable on the hold-out was reversed on the test set (Table 5). Arm 2 learned to recognise the phrasing of training questions rather than the disease, so it performed well on questions with similar phrasing and struggled when test questions used different wording. Arm 1's TF-IDF index was less influenced because it's built from NHS description text, not paraphrases. This is also why Arm 3 closely tracks Arm 1, since the LLM returns Arm 1's top candidate on 81% of test items.

Arm 1 and Arm 2 reached identical validation accuracy but disagreed on 22% of test items (44 of 200), suggesting they make mistakes on different questions. However, combining them would likely underperform Arm 1 alone, because Arm 2 is only uniquely correct on 2 of the 200 test items. Since Arm 1 is correct on almost every item where the two disagree, merging Arm 2's answers would add more mistakes than corrections.

### 3.3 Error Analysis

Macro-F1, the unweighted average of each class's F1-score, is lower than accuracy for every arm on test (Arm 1 scores 0.563 against its accuracy of 0.765), suggesting errors are not spread evenly.

Many of the 906 disease labels share a text prefix, forming label families (Figure 4c). 88 of the 200 validation questions have a true label with at least one family member. Both arms made 30 validation errors. For Arm 1, 25 of those errors were on questions where a within-family confusion was possible, and 18 of the 25 (72%) were due to that. For Arm 2 these figures are 21 and 10 (48%).  

This cannot be reported for the test set, where too few (18 of the 200) test questions have a label-family member for the rate to be meaningful. Figure 7 shows each arm's most frequent test confusions.

**Figure 7: Most frequent confusions.**

Arm 3 uses the same 20-candidate shortlist as Arm 1 (built by TF-IDF retrieval of candidates, adding them to the prompt, then generating an answer). Out of 200 test questions the LLM kept Arm 1's top choice for 120 (accuracy 0.958, unchanged), fell back to Arm 1's top choice for 42, because its own output was not a valid candidate (accuracy 0.429, also unchanged), and chose a different candidate to Arm 1's on 38 (an override), where accuracy fell from 0.526 (Arm 1) to 0.289 (Arm 3).

On those 38 override questions, Arm 1 was originally correct on 20 and wrong on 18, so overriding fixed 11 previously wrong answers but broke 20 previously correct ones.

The true label appeared in the LLM's 20-candidate list on 97.5% of test questions, so the LLM was rarely missing the right answer. Its weaker accuracy on override questions suggests misplaced confidence when it does intervene.

Table 6 puts individual test items behind these rates.

**Table 6: Example correct and incorrect test predictions, per arm.**

**Arm and outcome**

**Question**

**True label**

**Prediction**

Arm 1, correct

What are the treatments for MDS?

myelodysplasia

myelodysplasia

Arm 1, correct

What is the treatment for HHT?

hereditary\_haemorrahagic\_telangiectasia

hereditary\_haemorrahagic\_telangiectasia

Arm 1, incorrect

What is the diabetic eye screening?

diabetes

diabetic\_retinopathy

Arm 1, incorrect

What should I do if someone has a seizure?

epilepsy

what\_to\_do\_if\_someone\_has\_a\_seizure\_fit

Arm 2, correct

How is GA1 diagnosed in babies?

glutaric\_aciduria

glutaric\_aciduria

Arm 2, correct

Who can have BDD?

body\_dysmorphia

body\_dysmorphia

Arm 2, incorrect

What causes sudden cardiac death?

arrhythmia

brain\_death

Arm 2, incorrect

What is spinal fusion?

spondylolisthesis

braces\_and\_orthodontics

Arm 3, correct

Who is at risk of getting a C. diff infection?

c\_difficile

c\_difficile

Arm 3, correct

What are the symptoms of MDS?

myelodysplasia

myelodysplasia

Arm 3, incorrect

Can I get medication for bedbug bites?

bedbugs

insect\_bites\_and\_stings

Arm 3, incorrect

What is hyperhidrosis?

night\_sweats

excessive\_sweating\_hyperhidrosis

Two correct and two incorrect test predictions per arm, drawn from the frozen test run.

The retrieval arms' errors are near misses inside a clinical neighbourhood rather than category failures, since bedbugs against insect bites and stings, or diabetes against diabetic retinopathy, are defensible readings of a short question. Two are better described as label-granularity artefacts than as model errors: what\_to\_do\_if\_someone\_has\_a\_seizure\_fit is a closer literal match to its question than the gold epilepsy, and excessive\_sweating\_hyperhidrosis names the exact term the question asks about. This is the same ambiguity that gives 85 identical question strings more than one disease label (§4.3). Arm 2's errors travel further, reaching brain\_death for a question on sudden cardiac death and braces\_and\_orthodontics for one on spinal fusion, which is consistent with its lower macro-F1 of 0.354. Acronym questions separate the arms least: "What are the treatments for MDS?" is answered correctly by the retriever and by the LLM re-ranker alike, while one question, "How does everolimus work?" (neuroendocrine\_tumours), defeats Arm 1 and Arm 2 both.

## Discussion

### 4.1 Comparison of Approaches

Traditional retrieval beats both neural arms on test at a fraction of the compute (Table 7): Arm 1 needs no GPU and no training step, while Arm 3's inference cost is in addition to Arm 1, since it builds on it.

**Table 7: Computational resources per arm (hardware, time, what was measured).**

**Arm**

**Hardware**

**Time**

**What was measured**

Arm 1: TF-IDF + k-NN

CPU (local)

8.2 s

index build, then vectorise and rank 200 test questions.

Arm 2: Bio\_ClinicalBERT

Colab T4 GPU

32.9 min

full epochs sweep behind the checkpoint choice. The frozen checkpoint is epoch 15, so training only to that point costs less.

Arm 3: shortlist + LLM

Colab T4 GPU

78 s

generation over 200 items at the selected zero\_shot condition, on top of Arm 1.

Computational resources per arm. Arm 1 is timed in this run on local CPU, while Arms 2 and 3 are read from the artefacts their Colab runs persisted, since neither can be re-run on the laptop. Arm 2 is training, Arm 3 is inference, orders of magnitude instead of durations are compared in the report.

Near-uniform distribution at around 10 examples per class is too little per-class data for fine-tuning to learn separate decision boundaries across 906 classes. Retrieval does not have to learn them, as the class-blob index adds each condition's NHS document, so a test question is matched against description paragraphs. The label-overlap cue is mostly absent at test time (5.5% of test questions contain a word from their own label against 84.0% of training questions), which is likely why the encoder, having memorised sibling phrasing, scored well on validation and then degraded.

### 4.2 Impact of Design Choices

The index variant was Arm 1's largest lever, and the shift-aware protocol is what identified it: the standard hold-out could not separate the four variants at all. The test results confirm that when a hold-out is drawn from paraphrase siblings, a shift-aware split is the better guide to generalisation. No isolating ablation was re-run at the frozen vectoriser, so the index switch's contribution to that gain is stated as consistent with the evidence, not as demonstrated.

Table 8 shows sublinear\_tf=True scoring 0.530 on the shift-aware split against the frozen configuration's 0.398, a difference of over five SE, while the standard hold-out separates them by 2pp, within one SE. Sublinear weighting replaces a raw term count with 1+log(count), which matters when each class document concatenates a full NHS page, because a term repeated throughout that page would otherwise dominate its class vector. The tie-break rule was scoped to index selection, as applied more widely it would have selected True. This surfaced only after the test set had been read, so it was left unchanged to avoid the test-set bias that fixing the rule in advance was meant to prevent. It is the highest priority change to make in a repeat.

**Table 8: vectoriser and index ablations: standard vs. shift-aware hold-out accuracy.**

**Variant**

**Hold-out accuracy**

**Shift-aware accuracy**

**Δ vs frozen (shift-aware)**

**Vocab size**

Frozen configuration (baseline)

0.850

0.398

0.000 (baseline)

237037

ngram\_range=(1,1) (unigrams only)

0.820

0.328

\-0.070

16671

min\_df=2

0.755

0.305

\-0.093

78757

sublinear\_tf=True

0.870

0.530

+0.133

237037

stop\_words='english'

0.850

0.400

+0.003

247743

Lemmatisation enabled (en\_core\_web\_sm, stop words kept as frozen)

0.840

0.352

\-0.045

239335

NHS documents uncleaned (boilerplate retained)

0.850

0.400

+0.003

242212

index\_variant=QL

0.815

0.203

\-0.195

30700

index\_variant=QLA

0.840

0.305

\-0.093

141423

Both protocols use the frozen Arm 1 vectoriser/index settings except the single varied factor per row, seed=44, and the same std-val split (n=200, 102 classes) / hard-val split (n=400, 400 classes) across all rows. Standard-error reference at the frozen row's accuracy: std-val SE ≈ 2.52pp (n=200), and shift-aware SE ≈ 2.45pp (n=400).

The rules differ in provenance since Arm 1's and Arm 3's were fixed in advance before the numbers were seen, Arm 2's only after. Both the Arm 2 and Arm 3 pre-declared rules deliberately retained the lower-scoring option as previously declared.

### 4.3 Implications for Healthcare

85 identical question strings map to more than one disease, an irreducible error floor. A triage system here should abstain rather than guess: validation accuracy climbs from 0.850 to 0.945 once only the most confident 36.5% of items are answered (Figure 8). Arm 3 shows the same principle accidentally, its only component that never hurt accuracy was the fallback that declined to re-rank. With the sharp degradation once phrasing shifts, patient-facing classifiers must be validated on genuinely independent data and deployed with a route to human review.

**Figure 8: Arm 1 accuracy-coverage curve (validation).** 

Accuracy of retained predictions and fraction of items retained (coverage) as the nearest-neighbour similarity abstention threshold varies; the source for this 0.850 to 0.945 accuracy at 36.5% coverage figure above.

### 4.4 Limitations and Future Work

Every selection decision here rests on one 200-item hold-out with a 3.5pp standard error, drawn from questions that are phrasing siblings of those left in training. Cross-validation would not repair this, because the siblings sit inside every fold. A repeat should make the shift-aware split the primary protocol rather than a tie-breaker, and keep the standard hold-out only for reporting.

The clearest cost of not having done so is sublinear\_tf (§4.2), left at False because the pre-registered rule's scope reached the index but not the vectoriser, and it is the first change to make. A related weakness is one of provenance rather than measurement, in that Arm 2's encoder rule was written down only after its comparison had been seen, so a repeat should pre-register every tie-break rather than most of them.

Arm 3 is the most improvable component, and its own diagnostics indicate how. Overriding the retriever was net-negative on both splits, at 0.120 precision on validation and 0.289 on test, so the LLM should be gated on the retriever's similarity margin and allowed to re-rank only where the top 1 is genuinely contested, instead of on every item. Constraining decoding to the twenty candidate strings would likewise remove the 13% to 18% fallback rate by construction rather than absorbing it (§3.2), given that those outputs fail on format rather than on content.

Scope limits the rest. Models were fit on split\_fit alone and never refit on the validation items, so that the tested system is exactly the validated one, at the cost of 2.2% of the training data. Results come from a single seed and a single 200-item test set covering 102 of the 906 classes, and the abstention threshold behind Figure 8 was set on validation and never confirmed on test.

## References

Alsentzer, E., Murphy, J., Boag, W., Weng, W.-H., Jindi, D., Naumann, T. and McDermott, M. (2019) 'Publicly available clinical BERT embeddings', *Proceedings of the 2nd Clinical Natural Language* *Processing Workshop*, NAACL, pp. 72-78.

Available at: https://aclanthology.org/W19-1909/

Chen, Q., Hu, Y., Peng, X., Xie, Q., Jin, Q., Gilson, A., Singer, M.B., Ai, X., Lai, P.-T., Wang, Z., Keloth, V.K., Raja, K., Huang, J., He, H., Lin, F., Du, J., Zhang, R., Zheng, W.J., Adelman, R.A., Lu, Z. and Xu, H. (2025) 'Benchmarking large language models for biomedical natural language processing applications and recommendations', *Nature Communications*, 16(1), 3280. doi: 10.1038/s41467-025-56989-2

CogStack (2023) *OpenGPT: a framework for creating grounded instruction-based datasets and training*  
*conversational domain expert LLMs*. Available at: https://github.com/CogStack/OpenGPT

Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019) 'BERT: pre-training of deep bidirectional transformers for language understanding', *Proceedings of NAACL-HLT 2019*, pp. 4171-4186.  
Available at: [https://aclanthology.org/N19-1423/](https://aclanthology.org/N19-1423/)

Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C.H. and Kang, J. (2020) 'BioBERT: a pre-trained biomedical language representation model for biomedical text mining', *Bioinformatics*, 36(4), pp. 1234-1240. doi: 10.1093/bioinformatics/btz682

Schulhoff, S. et al. (2024) *The prompt report: a systematic survey of prompting techniques*. arXiv:2406.06608.  
Available at: https://arxiv.org/abs/2406.06608

## Appendix

### Submitted code

Two files accompany this report:

- **`AMLH_patient_question_classification.ipynb`** — the reproducible notebook, submitted with all outputs saved from the run described below.
- **`amlh_submission_inputs.zip`** — the data archive the notebook prompts for. It holds the training CSV, the test CSV with its `answer` column removed, all 906 NHS reference documents, and the recorded sweeps described under *Runtime* below.

The notebook is self-contained. It clones no repository and reads no path outside its own working directory. Its §0.3 writes the project's twelve Python modules to `src/amlh/` with `%%writefile` and then imports them, so the submission is one file while the code retains the package structure that lets the same code path serve the validation grids and the frozen test run. Target runtime is a free Google Colab T4; Arms 2 and 3 require the GPU.

`SEED = 44`, set in `config.py` and re-seeded before every stochastic stage.

### Section map

Notebook sections carry the report section they back, so any figure or table above can be traced to the cell that produced it.

| Notebook | Contents | Report |
|---|---|---|
| §0 | environment, input archive, source modules | — |
| §1 | integrity audit, document cleaning, validation split, EDA | §2.1, §2.2, §3.1, F1–F3 |
| §2 | Arm 1 — vectoriser grid, preprocessing ablation, index variants, shift-aware tie-break, indexing scheme, split robustness, supervised baselines | §2.3, §4.2, F7 |
| §3 | Arm 2 — label encoding, encoder sweep, epoch and encoder selection, frozen training | §2.4, §4.2, F6 |
| §4 | Arm 3 — shortlist and reproduction guard, prompt budget, condition × model grid, two-stage selection, cross-arm validation comparison | §2.4, §3.2, T2 |
| §5 | the frozen test run: freeze check, three arms on test, McNemar, optimism, error analysis | §3.2, §3.3, F8–F9 |
| §6 | workflow diagrams, artefact inventory, wall clock | F4, F5 |

Test data is read in two places only, both labelled where they occur: §1.4–§1.6, for the distributional diagnostics that select nothing, and §5, after the freeze check has printed every hyperparameter.

### One reproducibility caveat

The validation split is **shipped in the archive rather than regenerated by the notebook**, and §1.3 both demonstrates why and prints the evidence. `make_validation_split` is deterministic within an environment — called twice, it returns the same split — but it does not reproduce the committed split bit for bit on a current numpy and pandas. The algorithm is unchanged, which §1.3 verifies structurally: the shipped split has exactly the quota shape the code produces, 102 classes with 98 taking two items and 4 taking one. What differs is the `rng.choice` draw itself, relative to the environment the split was originally made in.

Regenerating it would move every number in the report by roughly a standard error while appearing to reproduce them, so the notebook loads the shipped split and reports the divergence instead of hiding it. This is the sharpest form of the seed-stability problem: a seed fixes an algorithm's behaviour within one environment, not across library versions, and pinning exact versions in an environment file is the change to make in any repeat of this work.

### Runtime

The notebook carries one flag, `RUN_FULL_SELECTION`, submitted as `False`. This gates the two selection *sweeps* — Arm 2's 24-epoch comparison of both encoders, recorded at 66.6 minutes on a T4, and Arm 3's three-condition × two-model prompt grid, recorded at 24.1 minutes. With the flag off, both are loaded from the recorded run in the archive. Every selection *rule* still executes live on them: the within-one-standard-error epoch choice, all McNemar tests, and both tie-breaks run and print their decisions either way. Setting the flag to `True` re-runs both sweeps, which does not fit inside one free Colab session.

Everything else executes: the whole of §1 and §2, the frozen 15-epoch Bio\_ClinicalBERT training in §3.4, the frozen prompt condition on validation in §4.5, and the entire test run in §5.

<!-- TODO before submission: replace with the total wall clock printed by §6.2 of the executed run. -->
Total runtime as submitted: 2 hrs 28 mins.