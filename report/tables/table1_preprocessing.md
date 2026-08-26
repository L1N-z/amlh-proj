| Stage | Parameter | Value used | Library default | Changed? | Source |
|---|---|---|---|---|---|
| Case folding | `lowercase` | True | True | No | default |
| Tokenisation | `token_pattern` | `(?u)\b\w\w+\b` | `(?u)\b\w\w+\b` | No | default |
| N-grams | `ngram_range` | (1, 2) | (1, 1) | Yes | grid |
| Stop words | `stop_words` | None | None | No | grid |
| DF floor | `min_df` | 1 | 1 | No | grid |
| DF ceiling | `max_df` | 1.0 | 1.0 | No | fixed |
| Term weighting | `sublinear_tf` | False | False | No | grid |
| IDF | `use_idf` | True | True | No | default |
| IDF smoothing | `smooth_idf` | True | True | No | default |
| Normalisation | `norm` | l2 | l2 | No | default |
| Lemmatisation | spaCy model / enabled | en_core_web_sm / disabled | n/a (not a TfidfVectorizer parameter) | No | ablation |
| BERT tokenisation | WordPiece: max_length/padding/truncation | BertTokenizer, checkpoint=emilyalsentzer/Bio_ClinicalBERT, model_max_length=unset (no tokenizer-imposed limit), configured max_length=48, padding=max_length, truncation=True | model_max_length=512 (BERT-family default) | - | fixed |

Fitted on: the frozen Arm 1 TF-IDF vectoriser is fit only on the class_blob index text built from the training split's fit rows (`amlh.features.build_index`, variant=`QLAD`) — never on validation or test text (fit/transform boundary enforced by `amlh.arm1_tfidf.knn_rank`, which calls `.fit_transform` on the index only and `.transform` on queries). BERT row: see note in that cell — Arm 2 hyperparameters are frozen in config.py: checkpoint emilyalsentzer/Bio_ClinicalBERT, max_length=48, selected epoch 15.
