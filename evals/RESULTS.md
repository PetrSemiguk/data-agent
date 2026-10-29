# Evaluation results (stage 5)

How well does the agent answer data questions, what do its extra parts (skills,
the verifier) add, and what does an answer cost? All numbers below come from the
eval runs in October 2026 with `claude-sonnet-5-5` as the agent.

## What was measured, and how

**Question sets** (`evals/questions.json`, 39 questions in 5 splits):

| Split | Questions | Datasets | Purpose |
|---|---|---|---|
| `dev` | 11 | Superstore, Titanic, Online Retail II | debugging; failures looked at in detail |
| `dev_hard` | 10 | same | multi-step questions, near ties, several denominators, 5+ numbers per answer; interpretation fixed in the question text |
| `dev_open` | 8 | same | questions as a user would ask them, with a real trap the question does not mention (tiny groups in a ranking, an incomplete month, duplicates, service rows, skew, missing values) |
| `holdout` | 5 | IBM HR Attrition, Daily Climate | never used while building the agent; only totals are looked at |
| `final` | 5 | Heart Failure, King County House Sales | the exam, run once at the end with the chosen configuration |

Categories: lookup, aggregation, time series, group comparison, data quality trap,
multi-step. 32 of 39 questions contain at least one trap.

**Expected values** are computed by code (`reference.py`, pandas) and computed a
second time by independent code (`crosscheck.py`, the `csv` module and plain Python).
Both agree on every value. Where several readings of a question are reasonable, the
question fixes the reading or the check accepts every reasonable answer (a tolerance,
or a list of accepted labels), with the reason in `notes`.

**Grading** (`grade.py`): a judge model only *extracts* what the answer states (a
number in the check's unit, a named value, a ranking, or yes/no for a required
mention or a forbidden claim) with a verbatim quote; it never sees the expected
values. Code compares the extraction with the reference: tolerances, rounding to the
answer's precision (capped at 1%), percent vs fraction, ranges, rankings with
alternative names, and answers that state several values (pass only if every value
is acceptable). Each answer passes *strictly* only if every check passes.

**Checking the grader:**
- 64 checks on 23 hand-written answers with known correct grades (right and wrong
  numbers in human formats, swapped rankings, numbers about another year or group,
  forbidden claims in other words, self-contradicting answers): Sonnet 64/64 (three
  runs), Haiku 64/64.
- A human reviewed 27 checks of 10 random graded dev answers: agreement 27/27.
- Every graded failure was examined (holdout and final only by a classification,
  without details). All but one borderline holdout answer were grader or reference
  errors: the direction of a one-item ranking, a missing judge verdict, a product
  with several descriptions, a state with two names, and an answer naming two
  acceptable values. Each was fixed with a general rule, then all runs were graded again.

**Runs:** each question runs in a fresh sandbox, without memory and without a human
(clarifying questions get "user not available", exports are denied). A check for
reads of the eval files runs on every log; no run touched them.

## Main results

Strict pass rate = share of runs where every check passed. 95% Wilson intervals;
repeats of a question are not independent, so the true uncertainty is larger.

| Set (run) | Config | Runs (n) | Strict pass | 95% CI | Without verifier (draft) | Cost per question |
|---|---|---|---|---|---|---|
| dev (`dev_baseline`, 3 repeats) | skills on | 33 | 33/33 | 89.6-100% | 32/33 | $0.031 |
| dev_hard (`hard_baseline`) | skills on | 10 | 10/10 | 72.2-100% | 10/10 | $0.035 |
| dev_open (`open_baseline`) | skills on | 8 | 8/8 | 67.6-100% | 8/8 | $0.061 |
| dev_open (`open_noskills`) | skills off | 8 | 8/8 | 67.6-100% | 8/8 | $0.043 |
| holdout (`holdout_baseline`, 3 repeats) | skills on | 15 | 15/15 | 79.6-100% | 15/15 | $0.034 |
| **final (`final_exam`)** | **default** | **5** | **5/5** | **56.6-100%** | **4/5** | **$0.036** |

Cost per question = agent + verifier + judge (Haiku). All 79 graded runs passed
strictly (95% CI 95.4-100%). On holdout one borderline answer (it led with a
distorted figure and then gave the right one as "the better estimate") failed under
the Sonnet judge and passes under the Haiku judge; this is the only grade that
depended on the judge.

## Experiments

**Skills on/off** (`dev_open`, same verifier, 8 questions each): accuracy 8/8 vs 8/8,
every trap mention 10/10 in both; with skills the agent took 4.6 vs 3.8 steps and
cost $0.061 vs $0.043 per question (+43%). No measurable gain at this n.

**Verifier** (a second model that recomputes the claims of the final answer):
- Precision of its remarks: of 12 claims it contradicted on dev, dev_hard and
  holdout, 11 were real errors when rechecked with pandas, 1 was a false alarm.
- None of those 11 errors was in a main number covered by our checks; all were side
  claims (a wrong count of loss-making sub-categories, "slightly higher" for +19.6%,
  "no date column" when there is one).
- Effect on the graded result: across dev, dev_hard, dev_open and holdout, revised
  answers changed the strict result once (dev, a contradictory side claim). On the
  final exam it changed one answer from fail to pass (4/5 → 5/5).
- Cost: about as much as the agent itself ($0.016-0.027 per question).

**Haiku instead of Sonnet:**
- As the judge: 64/64 on the grader cases, at about 1/14 of the cost. Adopted.
- As the verifier: 4/5 regression cases vs Sonnet 5/5. It computed the right numbers
  for a claim mixing two different sets of rows, but marked it supported. Not adopted.

## Chosen configuration (`agent/config.py`)

- Skills **off** by default (`--skills` or the "Use skills" toggle turns them on): no
  accuracy gain on dev_open, about 40% higher cost.
- Verifier **on**, model **Sonnet 5.5**: its remarks are accurate, it fixed one
  final-exam answer, and Haiku missed a planted error.
- Judge: **Haiku 5.5** (grading only).
- Prompt caching on, memory per dataset on in the app (off in evals).

## Limitations

- **Small n, wide intervals.** 5 final questions give a 95% interval of 57-100%
  even with no failure; the sets show that the agent is good on these kinds of
  questions, not how often it fails in general.
- **Same author.** The questions, the expected answers and the agent were written in
  the same project; questions a different person would ask may be harder.
- **The judge is not deterministic.** Temperature cannot be set on these models;
  one borderline holdout grade depended on the judge model.
- **The verifier does not see charts**; it checks numbers and statements in text only.
- **Code guard protects against mistakes, not attackers.** It blocks writes, deletes,
  shell, network and reads outside the dataset folders, but can be bypassed on
  purpose; real isolation is a container (stage 6).
- **The final set is used.** Further changes cannot be checked on it.

## Conclusions

1. On 79 graded runs over 39 questions, including traps the question does not
   mention, every answer passed every check; the agent notices incomplete periods,
   duplicates, service rows, tiny groups and skew on its own.
2. Skills did not improve accuracy at the measured n and cost about 40% more, so they
   are off by default.
3. The verifier is accurate (11/12 remarks correct) but mostly cleans up side claims;
   it changed the graded result twice in 79 runs. It stays on for answer hygiene.
4. Haiku is good enough to grade, not to verify.
5. To measure further progress, the eval needs questions where the agent actually
   fails, written by someone else, and more repeats.
