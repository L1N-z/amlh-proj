| Arm | Hardware | Wall clock | What was measured |
|---|---|---|---|
| Arm 1 -- TF-IDF + k-NN | CPU (workstation) | 8.2 s | index build, vectorise and rank 200 test questions; timed in this run |
| Arm 2 -- Bio_ClinicalBERT | Colab T4 GPU | 32.9 min | full epoch sweep behind the checkpoint choice; peak GPU memory 2.2 GB. The frozen checkpoint is epoch 15, so training only to that point costs a fraction of this. |
| Arm 3 -- shortlist + LLM | Colab T4 GPU | 78 s | generation over 200 items at the selected `zero_shot` condition, on top of Arm 1 -- Arm 3 adds to the retrieval cost, it does not replace it. |

Computational resources per arm. Arm 1 is timed in this run on CPU; Arms 2 and 3 are read from the artefacts their Colab runs persisted, since neither can be re-timed on the workstation. The figures are not like-for-like units of work -- Arm 2's is training, Arm 3's is inference -- so the report compares orders of magnitude, not durations.
