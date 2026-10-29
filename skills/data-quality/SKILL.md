---
name: data-quality
description: Use before any analysis whose result depends on clean rows - counts, totals, averages or rankings - to find and handle duplicates, missing values, impossible values and non-standard records.
---

# Data quality check

Goal: know what is in the table before computing anything, decide what to exclude,
and be able to state every exclusion with a row count and a reason.

## Method

1. Work on a copy: `data = df.copy()`. Never change `df`.
2. Shape and types
   - Print the shape, the dtypes and a few rows.
   - Find columns stored as text that should be numbers or dates, and convert them
     on the copy with `errors="coerce"`. Count how many values failed to convert.
3. Duplicates
   - Count fully identical rows: `data.duplicated().sum()`.
   - Count duplicates on the natural key too (the columns that should identify one
     record), because near-duplicates often differ only in an id or a timestamp.
   - Decide: drop them (accidental double entries) or keep them (legitimate repeats).
     Look at a few examples before deciding. State the decision and the count.
4. Missing values
   - Count missing values per column, in numbers and in percent.
   - For each column that matters to the question, decide: drop rows, keep them in a
     separate "unknown" group, or leave them (if the column is not used).
   - Do not fill missing values with a mean or zero unless the question requires it,
     and then say so.
5. Impossible and suspicious values
   - For every numeric column used: min, max, count of negative values and count of
     zeros.
   - Ask whether a negative or zero value can be real here. Negative amounts often
     mean reversals, refunds or corrections; zeros often mean placeholders.
   - Check dates for values outside a plausible range and for future dates.
6. Non-standard records
   - Look at the most frequent values of id, code and category columns. Codes with a
     different pattern (letters where digits are expected, a prefix, a fixed text
     such as "test", "dummy", "total") often mark records that are not ordinary
     observations.
   - Count them, look at examples, and decide whether the question is about them.
7. Outliers
   - Use quantiles (1%, 99%) or the IQR rule to find extreme values.
   - Look at the extreme rows before removing anything. An outlier is removed only
     if it is an error or out of scope, never just because it is large.
   - If the result changes a lot with and without the outliers, report both.
8. Apply the exclusions in a fixed order and print the row count after each step,
   so the final answer can list them.

## Typical pitfalls

- Dropping duplicates or outliers silently: the reader cannot reproduce the numbers.
- Counting duplicates only on full rows and missing near-duplicates.
- Treating reversal records (negative values) as errors and dropping them, which
  overstates totals; or keeping them when the question is about original events.
- Dropping every row with any missing value (`dropna()` on the whole table) when
  only one or two columns are needed.
- Converting text to numbers without checking how many values became missing.
- Describing cleaning with numbers that were never printed by code.

## Checklist for the final answer

- [ ] The number of rows before and after cleaning.
- [ ] Each exclusion: what, how many rows, and why.
- [ ] The decision on duplicates and on missing values, with counts.
- [ ] How negative, zero and non-standard records were treated.
- [ ] Whether the result is sensitive to the exclusions (if it was checked).
