# Prompt for the next Claude Code session — bootstrap confidence intervals

Copy everything below the line into a fresh Claude Code session in this repo.

---

Add bootstrap confidence intervals to `src/amlh/evaluate.py`. Read `CLAUDE.md` first and treat
its hard rules as binding — especially rule #3 (never state a number code did not just print) and
rule #4 (`SEED = 44`, re-seed before every stochastic stage; the bootstrap is a stochastic stage).

This is a small, self-contained module change. No notebook needs to run, no model needs to train,
and no data beyond what is already in `artefacts/` is touched. Plan briefly, then implement.

## Why

`CLAUDE.md`'s metrics rule requires bootstrap 95% CIs **and** McNemar's exact test for every
system comparison, and report §3.2's main results table has a `95% CI` column. `mcnemar_exact`
already exists in `evaluate.py`; the CI half does not.

The two answer different questions and both are needed. McNemar asks whether two systems differ,
using only the items they disagree on. A bootstrap CI expresses how precisely a *single* system's
accuracy is estimated from 200 validation items. Do not let one stand in for the other.

## Deliverable

`bootstrap_accuracy_ci` in `src/amlh/evaluate.py`, plus tests in `tests/test_evaluate.py`.

Suggested surface — argue for a different one if you prefer:

```python
def bootstrap_accuracy_ci(
    pred: list[str],
    gold: list[str],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = SEED,
) -> dict[str, float]:
```

Returning `{"accuracy", "ci_low", "ci_high", "n", "n_boot"}`.

Design notes:

- **Percentile bootstrap over per-item correctness.** Resample the 200 items with replacement,
  recompute accuracy, take the `alpha/2` and `1 - alpha/2` quantiles. Say in the docstring why
  resampling *items* is the right unit here.
- **Seeded and deterministic.** Use a local `numpy.random.default_rng(seed)`, not global NumPy
  state — this function must not perturb the RNG that surrounding code depends on. Same inputs
  and seed must give a bit-identical interval; assert that in a test.
- **numpy only.** No new dependencies (`CLAUDE.md`'s stack rule). Vectorise the resampling rather
  than looping 10,000 times in Python.
- Match the existing module's style: docstrings that state *why*, no printing inside functions,
  `ValueError` on misaligned or empty input exactly as `mcnemar_exact` does.

## Edge cases the tests must cover

- Misaligned lengths and empty input raise `ValueError`.
- A system that is correct on every item gives `ci_low == ci_high == 1.0` — the bootstrap cannot
  express uncertainty when there is no variation to resample. Assert it rather than being
  surprised by it later, and note it in the docstring, because Arm 1's per-class slices in §3.3
  will hit this.
- Determinism under a fixed seed; a different seed gives a different but similar interval.
- The point estimate equals plain accuracy and lies inside the interval.
- A coverage sanity check: for a known Bernoulli process at a fixed seed, the interval width is
  in a plausible range for n=200. Assert bounds loose enough not to be flaky — this is a
  correctness check, not a statistics exam.

## Do not

- Do not add a paired-difference bootstrap yet. Comparisons go through `mcnemar_exact`; adding a
  second, differently-shaped comparison instrument invites quoting whichever looks better.
- Do not touch any notebook, artefact, or `config.py`.
- Do not read the test set.

## Reporting back

Run `pytest tests/` and show the real output. Then demonstrate the function on data already in
the repo — `artefacts/arm2_val_predictions.csv` has `pred` and `gold` columns aligned to
`artefacts/split_val.csv` — and print the resulting interval. That is a live check that the
function works on the real artefact shape, not a result to record anywhere.
