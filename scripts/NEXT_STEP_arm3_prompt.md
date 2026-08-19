# Prompt for the next Claude Code session — Arm 3 (LLM shortlist selection)

Copy everything below the line into a fresh Claude Code session in this repo.

---

Implement Arm 3 of the AMLH coursework: frozen Arm 1 retrieval shortlists 20 candidate labels,
and a Flan-T5-large LLM selects one. Read `CLAUDE.md` first and treat its hard rules as binding —
especially rule #2 (no test-set cell in notebooks 02–04), rule #3 (never state a number that code
did not just print), rule #4 (`SEED = 44`, re-seed before every stochastic stage), rule #5 (fit on
training data only) and rule #6 (notebooks exchange state through `artefacts/`).

**Plan before implementing. Show me the module API and the notebook section list, and wait for my
approval before writing code.** This step freezes four hyperparameters into `config.py` and adds a
pre-registered selection rule, so it touches the evaluation boundary.

## Do task 0 first, before any Arm 3 code exists

**Task 0 — pre-register the prompt-condition tie-break in `CLAUDE.md`.** This must be committed
*before* any prompt condition is run, and the commit order must show that. The rule, agreed
2026-08-19:

> **Arm 3 prompt-condition tie-break (pre-registered 2026-08-19, before any condition was run).**
> The three prompt conditions are compared pairwise on the standard hold-out with **McNemar's
> exact test**, over the same 200 validation items. If no condition separates from the others at
> p < 0.05, the comparison is **reported as unresolved** and `zero_shot` is selected on the
> declared prior of the simplest prompt — fewest tokens, no exemplar-selection confound, lowest
> inference cost. **This can and does retain a lower-scoring condition**, which is the intended
> behaviour of a prior. Only where McNemar resolves a comparison does measured accuracy decide.

Place it alongside the Arm 1 and Arm 2 tie-break rules in rule #2, and state explicitly that
unlike the Arm 2 encoder rule this one *was* pre-registered before the numbers were seen — the
report contrasts the two, so the record has to support the contrast.

Then implement tasks 1–3 below.

## Deliverables

1. `src/amlh/arm3_llm.py` — all logic.
2. `notebooks/04_arm3_llm.ipynb`, plus `notebooks/04_arm3_llm_colab.ipynb` following the Arm 2
   pattern (a `QUICK_SMOKE_TEST` flag, and a final cell that zips the artefacts for download).
   Thin notebooks: they import from `src/amlh`, display results, and persist to `artefacts/` and
   `figures/`. Must run standalone after a kernel restart from `artefacts/split_fit.csv` and
   `artefacts/split_val.csv`.
3. `tests/test_arm3_llm.py` — CPU-only, **no model download in the test path** (mock the
   generator; test the prompt builder, the parser and the fallback logic directly).
4. Arm 3 fields frozen in `src/amlh/config.py`.

## What Arm 3 is

`question` → frozen Arm 1 retrieval returns a 20-label shortlist → prompt lists the candidates →
Flan-T5-large emits a choice → constrained parse → label. `answer` is never an input.

The shortlist comes from `arm1_tfidf.knn_rank(..., depth=20)` at the **frozen** Arm 1
configuration in `config.py`: `index_variant="QLAD"`, `index_scheme="class_blob"`,
`ngram_range=(1,2)`, `k_neighbors=1`. Do not re-tune any of it. Read the values from
`HYPERPARAMETERS`; do not hardcode them.

## Design constraints, all non-negotiable

- **Model: `google/flan-t5-large`.** Seq2seq, so `AutoModelForSeq2SeqLM`, not
  `AutoModelForCausalLM`. No chat template. Chosen over `microsoft/MediPhi-Guidelines` (the NLP4
  practical's model) as a deliberate call to stay with the T5 family the NLP3 practical taught.

- **Load it in `float32`, not `float16`.** This is not a preference, it is a correctness
  requirement. Flan-T5 was trained in bfloat16 and produces NaN logits in fp16; the target Colab
  T4 is Turing (SM 7.5) and has **no native bf16** (that arrived with Ampere, SM 8.0), so fp32 is
  the only safe dtype. At 780M parameters fp32 is roughly 3 GB of weights, which is comfortable
  inside the T4's 16 GB. If you see NaNs or empty generations, check the dtype first.

- **Greedy decoding**: `do_sample=False`, temperature 0. Runs are then deterministic, and
  `llm_temperature` is a declared default rather than a swept hyperparameter.

- **`shortlist_k = 20` is declared, not tuned.** Justified by the measured ceiling in
  `artefacts/arm1_shortlist_ceiling.csv` (acc@5 0.98, acc@10 0.99, acc@20 0.995): 5 → 20 buys
  1.5pp of headroom, while all 906 labels would be ~3k tokens and degrade selection. Do not run a
  k grid. Report the ceiling — it bounds Arm 3 from above and is what makes its numbers
  interpretable.

- **`num_workers=0`** in any DataLoader (rule #7).

- **Fit on training data only.** Few-shot exemplars come from `split_fit.csv` and nothing else.
  The retrieval index is training-side only. Never read validation or test text into any fitted
  object.

## Measure the prompt budget before you build the conditions

Flan-T5's encoder caps at **512 tokens**, and this dataset's labels are long. Measured from
`artefacts/split_fit.csv`: label word count mean 3.5, p95 11, max 21, so a 20-label shortlist runs
to ~69 words typically but **~420 words worst case**, before the question and instructions.

So, first: build the prompts, tokenise them, and **print the token-length distribution and the
truncation rate at 512** — the same discipline `arm2_bert.truncation_rate` applied at
`max_length=48`. Report those numbers to me before going further. A silently truncated shortlist
would drop candidates off the end of the list and corrupt the whole arm.

Two mitigations to apply, both of which also help the model:

- **Prettify labels in the prompt** — replace underscores with spaces. `baby_teething_symptoms`
  reads as natural language that way, and tokenises shorter.
- **Number the candidates and have the model emit the index**, not the label text. Output is then
  1–3 tokens, exactly parseable, and the model never has to reproduce a 137-character label
  string. Map the index back to the label yourself.

If the budget is still exceeded after both, tell me before truncating anything — do not silently
shrink the shortlist, since `shortlist_k=20` is a declared and reported quantity.

## The three prompt conditions

Reported verbatim in the report's §2.5 prompt table, so build them as inspectable strings, not
f-strings scattered through the notebook.

- `zero_shot` — instruction + question + numbered candidates.
- `few_shot` — the same, preceded by `n_shots` worked examples drawn **from the fit split only**.
  Note the budget tension: exemplars carrying their own 20-candidate lists will not fit in 512
  tokens. Compact exemplars (question → correct label, without a full candidate list each) are the
  expected resolution. Whatever you choose, print the resulting token lengths and justify it.
- `cot` — elicit brief reasoning, then a constrained final answer that the parser reads.
  **Be honest in the write-up if this condition performs poorly.** Flan-T5-large is small and
  seq2seq, and is known to be weak at chain-of-thought relative to large decoder models. A clear
  negative result here is a legitimate finding for §4.2, not a bug to engineer around.

## Constrained parsing and fallback

Parse the model's output to a candidate index. When it cannot be parsed to a valid candidate,
**fall back to the Arm 1 top-1 label** and count it.

**Report the fallback rate per condition** — it is an explicitly graded item in the brief, not a
footnote. Persist it alongside accuracy so §3.2 can quote both. A condition with high accuracy and
a high fallback rate is largely reporting Arm 1's accuracy, and the table must let a reader see
that.

## Selection rule

Apply the pre-registered rule from task 0: pairwise McNemar (`evaluate.mcnemar_exact`, which
already exists) across the three conditions on the standard hold-out; unresolved at p ≥ 0.05 means
report it as unresolved and keep `zero_shot` on the declared prior. Compute a bootstrap 95% CI per
condition with `evaluate.bootstrap_accuracy_ci` (also already exists) — CLAUDE.md's metrics rule
requires both, and they answer different questions.

Print every comparison with its p-value and discordant counts. Do not describe one condition as
"better" on an unresolved comparison.

## Artefacts to persist

- `artefacts/arm3_val_predictions.csv` — per-item, one row per validation question, with the
  selected label, gold, the shortlist rank of the gold label, whether the fallback fired, and the
  raw model output. Per-item records are what make McNemar against Arms 1 and 2 possible.
- `artefacts/arm3_prompt_conditions.csv` — accuracy, fallback rate, CI and wall-clock per
  condition.
- `artefacts/arm3_condition_mcnemar.csv` — the pairwise comparisons.
- `artefacts/arm3_prompts.txt` — the three prompts verbatim, for the §2.5 table.

## Config fields to freeze

`shortlist_k = 20`, `llm_temperature = 0.0`, `prompt_mode` (from the selection rule),
`n_shots`. Print them from the notebook, then copy them into `config.py` as a follow-up source
edit with a provenance comment, exactly as Arms 1 and 2 did. Also add the model id — Arm 3 needs
an equivalent of `bert_model_name`; propose a field name rather than reusing Arm 2's.

## Reporting back

- Run `pytest tests/` and show the real output.
- Print the prompt token-length distribution and truncation rate before anything else.
- Show the per-condition table, every McNemar p-value, and the fallback rates.
- State plainly whether the tie-break fired and which condition was selected — and if the
  unresolved branch fired, say that `zero_shot` was kept on the declared prior and **not** because
  it scored best.
- Do not state any number that was not printed by code you just ran.
