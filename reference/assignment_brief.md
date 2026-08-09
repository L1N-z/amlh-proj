


## 1
AMLH 2025-2026 Assignment Instruction

Deadline for submission:  Wednesday, 08 July 2026, 16:00 BST
•Length: The summative assessment refers to the outcome of the entire module and
culminates in a documented paper of 2000-2500 words (excluding figures,
references) and the submission of an iPython-notebook.
•Python packages: tensorflow, pytorch,keras, scikit-learn, etc.

•Each student needs to select one dataset as your choice from the provided list. The
maximum number of students for any topic/dataset is 54. First come first served.
(Last year, all students selected their favorite datasets)

Dataset options for the Assignment:
A. Lateral flow test classification
B. Glucose prediction for diabetes
C. NLP- patient question classification
The assignment should follow this overall outline (the requirements of
each dataset are provided in a separate section):
## 1. Introduction Words:
Please describe the background, motivation and importance of
the data in light of related literature. Show a sound
interpretation of the medical problems presented in the data,
and introduce necessary concepts and existing research.

[10 marks]

## 350

## 2. Methodology
## 2.1 Preprocessing Words:
Outline the selected dataset (including features and class
labels) and provide descriptive statistics of the contained
variables. Visualise the data or feature space in a plot if it is
possible and explore the underlying characteristics.
Please describe details of preprocessing of the data, including
data cleaning, imputation, normalization, augmentation, up or
down sampling, feature engineering etc. of the chosen
dataset.

[15 marks]

## 450
2.2 Algorithm design and implementation Words:
Select two AI models of the course, or two neural networks
which have very different structures. Given a high-level
description of both algorithms including their rationale or model
structure. Describe and demonstrate for both algorithms:

## 700


## 2
a) How to generate training and testing data with
appropriate format that suit the AI model
b) Optimization of hyper-parameters
c) Model evaluation based on the outcomes including
widely-used criteria that mentioned in the course

Demonstrate your solution with an attached iPython notebook.
Ensure reproducibility and transparency.

[40 marks]
## 3. Results Words:
Present optimized hyper-parameters and reasonable
evaluation criteria such as a confusion matrix, precision, recall,
RMSE, MARD, F1, AUC and a ROC-plot, etc. Provide an
analysis for both algorithms with different parameters and give
a textual description of the results. (please choose appropriate
metrics carefully)

[25 marks]

## 600
- Discussion and Conclusion Words:
Compare and discuss your findings (results of two algorithms).
If it is possible, it would be good to compare with other
scientific publications that used the same medical dataset.
Discuss how you would improve your methodology, current
limitations and future work.

[10 marks]

## 400
## 5. Reference
## 6. Appendix
Attach a reproducible iPython Jupiter notebook.
Total: 100 marks

Plagiarism or collusion is not allowed. The module can be failed straightaway if the
assignment or codes notebook fails in the plagiarism test (please note the
assignment is within AI category 2).
Markers will look for the following sections in the assignment:
- Sound understanding of the provided dataset and appropriate pre-processing
to obtain a dataset that is suitable for the machine learning model
- Appropriate selection and learning/training of two algorithms (one of them can
be seen as the baseline) to address the target problems in medical imaging,
time series or NLP
- Evaluation of the performance and meaningful discussion
- Other requirements that have been asked in the associated dataset
instruction
- The layout, presentation, references of the paper/report
If there is any question regarding the dataset or coursework, please post it in the
forum on moodle page, or email Kevin/Ken/Julia directly.
Provisional mark and feedback will be released in Autumn 2026.


## 10
AMLH 2026 - NLP Coursework: Patient Question
## Classification
## 1. Introduction

In this assignment, you are required to build a patient question classification system
using Natural Language Processing (NLP) techniques. Your will be able to predict the
most likely disease or condition associated with a patient question.
Patient  question  classification  is  an  important  component  of  many  healthcare  AI
systems.  Correctly  identifying  the  underlying  disease  from  a  patient's  question  can
support information retrieval, patient triage, and clinical decision support systems. The
objective  of  this  coursework  is  comparing  traditional  NLP  approaches  and  modern
neural approaches (such as Large Language Models (LLMs)) for the task.

## 2. Dataset
a. Description

The  dataset  is  derived  from  the  patient  information  section  of  the  NHS.UK  website,
which provides explanations of various diseases, their symptoms, and treatments. It
was  synthetically  generated  using  ChatGPT  and  released  as  part  of  the  OpenGPT
dataset by the CogStack KCL/UCL team (https://github.com/CogStack/opengpt). The
test  set  has  been  manually  validated  by  a  human  expert  to  ensure  accuracy.  This
methodology is state-of-the-art in NLP [1,2].

The dataset covers a wide range of medical conditions (e.g., bronchiolitis, laryngitis,
acne,  gallstones,  pneumonia,  tonsillitis)  and  includes  multiple  Q&A  pairs  for  each
condition. It consists of:
- Two CSV files containing patient questions and corresponding disease labels
(for training and testing).
- A  ZIP  archive  (db_nhs_qa_classification.zip)  with  906  plain  text  documents
from  the  NHS.UK  website.  Each  document  contains  information  relevant  to
answering   questions   about   a   specific   disease   or   condition   (e.g.,   face
blindness).

b. Data format

CSV Files:
The dataset includes two CSV files:
- patient_qa_classification_train.csv – the     training     set,     containing
approximately 8,900 questions and  corresponding disease  labels across  906
diseases.
- patient_qa_classification_test.csv – the test set, with 200 question-disease
pairs across 102 diseases. The 102 diseases appearing in the test set are also
present in the training set.

Each file contains the following columns (see the table below):
- question: a natural language question related to a medical condition
- disease: the associated disease or condition
- answer: a reference answer to the question
- reference_url:  the  NHS.UK  webpage  from  which  the  document  for  the  zip
archive was scraped.


## 11


For  the  plain  text  files: Each  file  is  named  in  the  format  of xxx.txt,  where xxx is  a
disease. The disease value appearing in the CSV disease column corresponds directly
to the filename after removing the .txt extension: for example, gallstones corresponds
to gallstones.txt
The plain text documents may be used as an external knowledge source for retrieval-
based or retrieval-augmented approaches.

Acknowledgments: The dataset is public and is available at
https://github.com/CogStack/OpenGPT/blob/main/data/nhs_uk_full/prepared_generat
ed_data_for_nhs_uk_qa.csv

## 3. Problem Formulation

The  primary  objective  of  this  coursework  is  to  build  a  patient  question  classification
system that can accurately identify the disease associated with a patient question.
Given a patient question, the system should assign the most appropriate disease label
from the set of 906 disease classes available in the training data.
This  is  therefore  a  multi-class  text  classification  problem  with  906  possible  output
labels.

For example:
## Question
Is there a specific test for prosopagnosia?
## Predicted Disease
face_blindness

Please  refer  to  the  practical  session  in  Week  10 - Building  a  Retrieval-Augmented
Generation (RAG) System Using Large Language Models (LLMs) for Patient Q&A for
sample codes.

- Requirements on coursework report:

The report typically is composed of the following sections.
## Introduction
This section provides the background and literature review for the coursework. Discuss
why patient question classification is an important healthcare NLP task, and what NLP
techniques  have  been  used  to  solve  similar  problems  in  recent  years. Provide  an
overview  of  the  approaches  investigated  in  this  coursework  and  how  they  relate  to
recent literature.

## Preprocessing:


## 12
This section describes and justifies all preprocessing and conversion steps applied to
the data.
Where relevant, justify preprocessing choices such as tokenisation, lowercasing, stop-
word  removal  and  lemmatisation.  Text  conversion  may  be  performed  using  TF–IDF
representations,  for  example  through  the TfidfVectorizer class  from  the scikit-learn
(sklearn) library, or using transformer tokenisers such as the AutoTokenizer class from
the transformers library.  If  you  use  these  classes,  describe  the  preprocessing  and
conversion settings used. If default settings were not modified, look up and report the
relevant default behaviour.
Report preprocessing and text conversion settings in a dedicated table.

## Methodology
Implement and compare at least one traditional NLP approach and one neural-based
approach, such as LLMs. Include diagrams to illustrate the workflow of each method.
A suitable traditional baseline is a TF–IDF-based information retrieval system. In this
approach,  patient  questions are  represented  as  TF–IDF  vectors,  and  disease  labels
for  test  questions  are  predicted  using  the  labels  associated  with  the  most  similar
training questions. Describe and justify important hyperparameter choices, such as the
number of retrieved neighbours and similarity thresholds.
The  neural  approach  may  use  an  LLM.  Possible  approaches  include  zero-shot
prompting, few-shot prompting [3], or fine-tuning a BERT-based classifier. Examples
are provided in Labs 9 and 10.
Describe the complete experimental protocol in sufficient detail to enable reproduction
of  the  results.  This  should  include  key  hyperparameters  and  prompts  (reported  in  a
dedicated  table),  computational  resources  used,  and  any  ablation  studies  or  model
comparisons performed. If a validation set is used, describe how it was constructed.
Example  hyperparameters  include  temperature,  maximum  number  of  generated
tokens, learning rate, batch size, and number of epochs. Briefly justify the main design
choices.

Results analysis
Evaluate model performance on the test set using accuracy, defined as:

## 퐴푐푐푢푟푎푐푦 =
## 푁푢푚푏푒푟 표푓 퐶표푟푟푒푐푡푙푦 퐿푎푏푒푙푙푒푑 푆푎푚푝푙푒푠
## 푇표푡푎푙 푁푢푚푏푒푟 표푓 푆푎푚푝푙푒푠


Additional  metrics  may  also  be  reported  together  with  a  brief  explanation  of  the
additional insights they provide beyond accuracy.
Present results using tables and diagrams wherever possible. Where model training is
performed,  include  training  and  validation  loss  curves.  A  detailed  error  analysis  is
expected, including examples of correct and incorrect predictions per method.

## Discussion
Discuss  the  relative  strengths  and  weaknesses  of  the  traditional  and  neural  (LLM-
based)    approaches,    considering    factors    such    as    predictive    performance,
computational    requirements    and    ease    of    implementation.    Reflect    on    how
methodological choices, such as preprocessing, training settings, and prompt design
may   influence  performance.   The  discussion   should  also   consider   the  broader
implications of patient question classification systems in healthcare.

Organisation of Report



## 13
## Introduction
## 1.1 Literature Review
1.2 Motivation and Rationale

## Methodology
## 2.1 Dataset Description
## 2.2 Data Preprocessing
2.3 Traditional NLP Approach
2.4 Neural (LLM) Approach
## 2.5 Experimental Protocol

## Results
## 3.1 Exploratory Data Analysis
## 3.2 Performance Comparison
## 3.3 Error Analysis

## Discussion
4.1 Comparison of Approaches
4.2 Impact of Design Choices
4.3 Implications for Healthcare


## 5. Coding

Demonstrate your solution with an attached iPython notebook. Ensure reproducibility
and transparency.

## References

[1] Dubois, Y. et al. (2023). AlpacaFarm: A simulation framework for methods that learn
from  human  feedback. Neural  Information  Processing  Systems,  abs/2305.14387,
pp.30039–30069.
[2] Kweon, S. et al. (2024). Publicly shareable clinical large language model built on
synthetic  clinical  notes.  In: Findings  of the Association for  Computational  Linguistics
## ACL 2024. 2024.
[3]  Schulhoff,  S.  et  al.  (2024).  The  prompt  report:  A  systematic  survey  of  prompt
engineering techniques. arXiv [cs.CL].



