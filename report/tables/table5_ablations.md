| Variant | Hold-out acc | Shift-aware acc | Δ vs frozen (shift-aware) | Vocab size | Test acc (frozen row only) |
|---|---|---|---|---|---|
| Frozen configuration (baseline) | 0.785 | 0.180 | 0.000 (baseline) | 4257 | not run |
| ngram_range=(1,2) | 0.745 | 0.205 | +0.025 | 29181 | — |
| min_df=2 | 0.490 | 0.077 | -0.102 | 2052 | — |
| sublinear_tf=True | 0.790 | 0.225 | +0.045 | 4257 | — |
| stop_words='english' | 0.765 | 0.182 | +0.003 | 4056 | — |
| Lemmatisation enabled (en_core_web_sm, stop words kept as frozen) | 0.810 | 0.215 | +0.035 | 3472 | — |
| NHS documents uncleaned (boilerplate retained; no-op under frozen variant 'Q', which has no D component) | 0.785 | 0.180 | +0.000 | 4257 | — |
| index_variant=QL | 0.795 | 0.172 | -0.008 | 4333 | — |
| index_variant=QLA | 0.770 | 0.263 | +0.083 | 10924 | — |
| index_variant=QLAD | 0.835 | 0.328 | +0.148 | 16671 | — |

Both protocols use the frozen Arm 1 vectoriser/index settings except the single varied factor per row, seed=44, and the same std-val split (n=200, 102 classes) / hard-val split (n=400, 400 classes) across all rows. Standard-error reference at the frozen row's accuracy: std-val SE ≈ 2.90pp (n=200), shift-aware SE ≈ 1.92pp (n=400); do not call a sub-1-SE difference "better". Vocab size is fit on the row's index text over the FULL training set (split-independent), not the std/hard fit subsets. Test-acc column is populated only for the frozen row and is "not run": no persisted test-accuracy artefact for the frozen configuration exists as of this run (05_results.ipynb has not executed; Arm 2/3 hyperparameters are still unset in config.py, so the full frozen pipeline is not yet complete).
