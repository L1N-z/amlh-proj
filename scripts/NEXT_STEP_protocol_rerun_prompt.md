# Prompt for the next Claude Code session — re-run the protocol-ranking diagnostic

Copy everything below the line into a fresh Claude Code session in this repo.

---

Re-run `scripts/measure_protocol_ranking.py` against the **currently frozen** configuration and
report what changed. Read `CLAUDE.md` first and treat its hard rules as binding.

**This task touches the test set.** It is the one place in the project that may, under
`CLAUDE.md`'s **declared exception #2**. Before running anything, re-read that exception and the
pre-registered index-variant tie-break rule directly above it, and confirm to me that you have.
The constraints are absolute:

- The script is a **protocol diagnostic, not model selection**. `index_variant` and
  `index_scheme` were decided from `std_val_acc` / `hard_val_acc` alone.
- **`config.py` must not change as a result of this run**, whatever the numbers say. If the
  output tempts you to revisit a frozen choice, stop and tell me instead of acting.
- **Nothing test-derived may be written into `artefacts/`.** The script's own docstring commits
  to this, precisely so no notebook can later load a test-tainted value. Preserve that property.
- Test-set access stays confined to this one script. Do not add a test-set cell to notebooks
  02–04, and do not write a new script that reads the test set.

## Why re-run

`report_outline.md` records that every test-side figure quoted in earlier report drafts — the
0.700 / 0.550 pair, both Spearman coefficients (0.549, p=0.159 and 0.755, p=0.031), and the
changed-prediction matrices in §3.2 — was computed at the **old** `ngram_range=(1,1)` /
variant-`Q` configuration. The frozen configuration is now `index_variant="QLAD"`,
`index_scheme="class_blob"`, `ngram_range=(1,2)`. Those figures must be **re-derived, not
reused**, before §3.2 and §4.2 are written.

## Before you trust the output — verify the config plumbing

The script builds its vectoriser kwargs in `_vec_kwargs()` from `HYPERPARAMETERS`. Check, and
report, whether that helper actually reflects everything frozen in `config.py`:

- It passes `ngram_range`, `sublinear_tf`, `min_df`, `stop_words`. `HYPERPARAMETERS` also holds
  `max_df` and `lemmatise`. Does `features.build_vectoriser` receive them by default, and do its
  defaults match the frozen values? If a frozen setting is silently not reaching the vectoriser,
  the diagnostic is measuring a configuration that does not exist. Say so before running.
- Confirm where `k` comes from in `main()` and that it is the frozen `k_neighbors`.

Report any mismatch to me and wait. Do not "fix" `config.py` or the frozen values to make things
agree — if the script is wrong, fix the script; if the freeze is wrong, that is my call.

## Deliverable

1. The script re-run at the frozen configuration, with its **complete output captured verbatim**
   to `report/protocol_ranking_rerun_2026-08.md` — under `report/`, deliberately not under
   `artefacts/`, so nothing test-derived sits where a notebook could load it. Head the file with
   the date, the git commit hash, and the exact frozen values it ran under.
2. A short comparison section in that same file: for each figure `report_outline.md` currently
   quotes from the stale run, the old value, the newly measured value, and whether the
   qualitative claim built on it still holds. The claims to check specifically are:
   - the four class_blob variants tie on the standard hold-out while disagreeing on many items;
   - "the standard hold-out redistributes errors rather than reducing them";
   - the two Spearman coefficients, which the outline already proposes demoting to a footnote at
     n=8 with a four-way tie block.
3. Your assessment of whether §3.2's and §4.2's arguments survive the re-derivation unchanged,
   are merely re-numbered, or are actually undermined. Be blunt if it is the third. A weakened
   argument that gets disclosed is worth more marks than a strong one built on stale numbers.

## Do not

- Do not edit `config.py`, any notebook, or `report_outline.md`. Hand me the findings; the report
  edits are a separate decision.
- Do not persist test-derived values to `artefacts/`, in any format, under any filename.
- Do not state a single number that the re-run did not print (rule #3). If a figure in the
  outline has no counterpart in the new output, say it is missing rather than carrying the old
  one forward.

## Reporting back

Show me the raw output and then the comparison table. Run `pytest tests/` if you changed anything
under `src/`.
