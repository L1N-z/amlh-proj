| Variant | Hold-out acc | Shift-aware acc | Δ vs frozen (shift-aware) | Vocab size | Test acc (frozen row only) |
|---|---|---|---|---|---|
| Frozen configuration (baseline) | 0.850 | 0.398 | 0.000 (baseline) | 237037 | 0.765 |
| ngram_range=(1,1) (unigrams only) | 0.820 | 0.328 | -0.070 | 16671 | — |
| min_df=2 | 0.755 | 0.305 | -0.093 | 78757 | — |
| sublinear_tf=True | 0.870 | 0.530 | +0.133 | 237037 | — |
| stop_words='english' | 0.850 | 0.400 | +0.003 | 247743 | — |
| Lemmatisation enabled (en_core_web_sm, stop words kept as frozen) | 0.840 | 0.352 | -0.045 | 239335 | — |
| NHS documents uncleaned (boilerplate retained) | 0.850 | 0.400 | +0.003 | 242212 | — |
| index_variant=QL | 0.815 | 0.203 | -0.195 | 30700 | — |
| index_variant=QLA | 0.840 | 0.305 | -0.093 | 141423 | — |

Both protocols use the frozen Arm 1 vectoriser/index settings except the single varied factor per row, seed=44, and the same std-val split (n=200, 102 classes) / hard-val split (n=400, 400 classes) across all rows. Standard-error reference at the frozen row's accuracy: std-val SE ≈ 2.52pp (n=200), shift-aware SE ≈ 2.45pp (n=400); do not call a sub-1-SE difference "better". Vocab size is fit on the row's index text over the FULL training set (split-independent), not the std/hard fit subsets. The Test-acc column is populated for the frozen row only, read from `artefacts/arm1_test_predictions.csv` as persisted by the single frozen run in `05_results.ipynb`; no ablation row is ever evaluated against the test set, and this script computes no test accuracy of its own.
