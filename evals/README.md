# Evaluation set

Questions with reference answers for measuring the quality of the agent's final
answers. This folder holds the questions and the code that computes the expected
values; running the agent on them and grading the answers are separate steps.

## Files

- `questions.json` - the questions and their checks (expected values included).
- `reference.py` - one pandas function per question that computes its expected values.
- `build_expected.py` - validates `questions.json`, runs every reference function and
  writes the values into the checks.
- `crosscheck.py` - computes every expected value a second time with the `csv` module
  and plain Python (no pandas) and compares the two.

## Question format

| Field | Meaning |
|---|---|
| `id` | `d..` dev, `h..` holdout, `f..` final |
| `dataset` | file name in `data/` |
| `split` | `dev`, `holdout` or `final` (see below) |
| `question` | in English, as a user would ask it |
| `category` | `lookup`, `aggregation`, `time_series`, `group_comparison`, `data_quality_trap`, `multi_step` |
| `traps` | pitfalls the question contains, in words (may be empty) |
| `checks` | what a correct answer must contain (see below) |
| `reference` | name of the function in `reference.py` |
| `notes` | optional: why a tolerance was chosen when several readings are reasonable |

Check types:

- `number`: `{type, key, description, value, abs_tol | rel_tol}` - a value the answer
  must state; `key` links it to the reference function's output.
- `label`: `{type, key, description, value}` - one named value the answer gives as its
  answer (a region, a month, a year); compared ignoring case, abbreviations allowed.
- `ranking`: `{type, key, description, items, top_k, order}` - the first `top_k` items in
  order; `order` is `descending` (highest first) or `ascending` (lowest first).
- `mention`: `{type, description}` - something the answer must say (graded by a judge).
  If an answer states different values for one quantity and its result is unclear, the
  number, label or ranking check fails as "answer contradicts itself".
- `not_claim`: `{type, description}` - something the answer must not say (graded by a judge).

Ambiguous questions either fix the interpretation in the question text (e.g. "net
revenue as Quantity x Price over all rows, after removing exact duplicates") or use a
tolerance wide enough for every reasonable reading, explained in `notes`.

## Splits

- `dev` (Superstore, Titanic, Online Retail II): for debugging prompts, tools and
  skills. Look at failures freely.
- `dev_hard` and `dev_open` (the same datasets): harder dev questions; `dev_open` names no
  trap in the question text. Looked at like dev.
- `holdout` (IBM HR Attrition, Daily Climate): datasets never used while developing.
  Report only the overall numbers; do not tune the agent on individual questions.
- `final` (Heart Failure, King County House Sales): the exam. Run the agent on it
  once, at the very end, after all changes. Do not look at its expected values
  before that; the scripts hide them unless `--show-final` is given.
  **Used on 2026-10-29** (run `final_exam`, default configuration). The final split is
  spent: further tuning cannot be checked on it; a new final set is needed for that.

## Recompute the expected values

From the project root, after `python scripts/download_data.py`:

```
.venv\Scripts\python.exe evals\build_expected.py
.venv\Scripts\python.exe evals\crosscheck.py
```

`build_expected.py` rewrites the `value` and `items` fields of `questions.json`.
`crosscheck.py` must end with "All values match."
