# Prompt for the next Claude Code session — Arm 1 ranking depth + per-item predictions

Copy everything below the line into a fresh Claude Code session in this repo.

---

Give Arm 1 a usable ranking depth and persist its per-item validation predictions. Read
`CLAUDE.md` first and treat its hard rules as binding — especially rule #2 (no test-set cell in
notebooks 02–04; this task touches **no** test data at all), rule #3 (never state a number code
did not just print), rule #4 (`SEED = 44`) and rule #5 (fit on training data only).

**Plan before implementing.** Show me the module API change and the notebook section, and wait
for my approval before writing code — this touches the frozen Arm 1 retrieval path, and the
whole point is that it must not disturb any decision already frozen in `config.py`.

## The problem

`arm1_tfidf.knn_rank` builds its ranking from a similarity-weighted vote over the `k` nearest
neighbours: `ranked.append(sorted(vote, key=vote.get, reverse=True))`, where `vote` has one entry
per *distinct label among those k neighbours*. At the frozen `k_neighbors = 1` the vote therefore
holds exactly one label, and the ranked list has length 1.

Confirm this before changing anything — in `artefacts/arm1_grid_QLAD.csv`, every `k=1` row should
show `accuracy == acc_at_5 == mrr` (e.g. the first row reads 0.820 / 0.820 / 0.820). Print it,
don't take my word for it.

Two things break as a result:

1. **Arm 3 has nothing to shortlist from.** The planned design is "TF-IDF shortlists 20
   candidates, LLM selects", with Arm 1's acc@20 as the ceiling that bounds Arm 3 from above.
   There is no top-20 at `k=1`.
2. **Arm 1's Top-5 and MRR columns are degenerate** — they are accuracy repeated. Arm 2 produces
   genuine ones, so a side-by-side §3.2 table currently misrepresents Arm 1.

Separately: `artefacts/` holds ten Arm 1 tables and **all of them are aggregate**. No per-item
predictions are persisted anywhere, so McNemar between Arm 1 and Arm 2 cannot be computed, and
§3.3's error analysis (most-confused pairs, share of errors inside a label family, two correct
and two incorrect examples per arm) has nothing to read.

## Deliverables

1. A ranking-depth path in `src/amlh/arm1_tfidf.py` that **provably does not change any frozen
   top-1 prediction**.
2. `artefacts/arm1_val_predictions.csv` — per-item, aligned to `artefacts/split_val.csv` order.
3. `artefacts/arm1_shortlist_ceiling.csv` — acc@k for k ∈ {1, 5, 10, 20}, the bound Arm 3 is
   interpreted against.
4. Tests in `tests/test_arm1_tfidf.py`.
5. A section in `notebooks/02_arm1.ipynb` that produces 2 and 3, runnable standalone after a
   kernel restart from `artefacts/`.

## The non-negotiable constraint

**Top-1 must be identical to the current frozen output, for every validation item.** This is a
comparability fix, not a re-tuning: nothing about `index_variant`, `index_scheme`, `ngram_range`
or `k_neighbors` may change, and no selection may be revisited. Assert the identity in code over
all 200 items and print the result — do not eyeball it, and do not assert it only for the
class-blob case if you implement something more general.

Why it should hold: the frozen `index_scheme = "class_blob"` builds one index row per class, so
ranking all class blobs by cosine puts the same class at rank 1 as the `k=1` nearest-neighbour
argmax. That is an argument, not evidence. Measure it.

## Suggested design — argue for a different one if you prefer

Add a `depth: int | None = None` parameter to `knn_rank`:

- `None` keeps today's behaviour exactly, so every existing caller
  (`run_vectoriser_grid`, `run_index_variant_comparison`, `run_indexing_scheme_comparison`,
  `run_variant_scheme_grid`, `run_split_robustness`, `scripts/measure_protocol_ranking.py`) is
  untouched and every persisted grid stays reproducible.
- When set, retrieve `min(max(k, depth), len(index_texts))` neighbours. The head of the ranking
  is still the `k`-neighbour weighted vote, unchanged. Labels appearing only in neighbours
  `k+1 … depth` are appended below it, ordered by their best cosine similarity, deduplicated.

That construction preserves the head of the ranking by definition for any `k`, not just `k=1`.
State in the docstring *why* the parameter exists, in terms of what breaks without it.

Reuse `evaluate.accuracy_at_k` for the ceiling table rather than writing a second implementation.

## Artefact schemas

`arm1_val_predictions.csv` — one row per validation item, in `split_val.csv` order:
`question, gold, pred, top_1 … top_20, top_sim`

`top_sim` is `knn_rank`'s existing top-similarity output; §4.3's abstention argument and the
accuracy–coverage curve both key off it. Where fewer than 20 labels are ranked, leave the
remaining columns empty rather than padding with a placebo label.

`arm1_shortlist_ceiling.csv` — `k, accuracy` for k ∈ {1, 5, 10, 20}.

Read every hyperparameter from `config.HYPERPARAMETERS`. Do not restate literal values in the
notebook; if a value has to be typed twice, that is a bug.

## Reporting back

Show me the real printed output: the degeneracy check on the k=1 grid rows, the top-1 identity
assertion over all 200 items, the ceiling table, and the head of the predictions file. Run
`pytest tests/` before you finish (`CLAUDE.md` requires it after changes to `features.py` or
`evaluate.py`; run the whole suite regardless).

Finally, draft two sentences for report §3.2 disclosing that Arm 1's ranking depth was added
after the fact for cross-arm comparability, that it changes no top-1 prediction and no frozen
hyperparameter, and that Arm 1's previously recorded acc@5/MRR at `k=1` were identical to its
accuracy by construction. Do not write them into the report — just hand them to me.
