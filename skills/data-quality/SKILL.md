---
name: data-quality
description: Use before any analysis whose result depends on clean rows - counts, totals, averages or rankings - to find and handle duplicates, missing values, impossible values and non-standard records.
---

# Data quality checklist

Work on a copy (`data = df.copy()`); print the row count after every exclusion.

Easy to forget:
- Duplicates: count full-row duplicates AND duplicates on the natural key; look at
  examples before dropping; state the decision and the count.
- Text columns that hold numbers or dates: convert with `errors="coerce"` and count
  the values that failed.
- Negative and zero values: decide whether they are real (reversals, corrections,
  placeholders) or errors - do not drop them silently, and do not keep reversals
  when the question is about original events.
- Non-standard records: codes with a different pattern (a prefix, letters instead of
  digits, fixed texts like "test" or "total") are often not ordinary observations.
- Missing values: drop only for the columns the question uses, never `dropna()` on
  the whole table; do not fill with mean or zero unless asked.
- Outliers: look at the extreme rows; remove only errors or out-of-scope rows; if the
  result changes a lot, report it with and without them.

The answer must state:
- [ ] rows before and after cleaning;
- [ ] each exclusion with its row count and reason (numbers printed by code);
- [ ] how duplicates, missing, negative/zero and non-standard records were treated.
