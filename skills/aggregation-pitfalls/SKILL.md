---
name: aggregation-pitfalls
description: Use when the answer is built from averages, ratios, percentages, shares or rankings of groups - to avoid wrong averages, misleading ratios and unstable rankings.
---

# Aggregation pitfalls

Goal: make every summary number mean what the answer says it means.

## Method and pitfalls

1. Mean vs median
   - Check skewness before choosing: compare mean and median, and look at the
     99th percentile and the maximum.
   - If the mean is much larger than the median, a few large values drive it. Report
     the median as the typical value and the mean as "average", or both.
   - Say which statistic is used; "average" alone usually means the mean.
2. Mean of ratios vs ratio of sums
   - The mean of per-row ratios (e.g. the average of each group's rate) weights every
     row equally; the ratio of sums (total A / total B) weights by size.
   - They answer different questions and can differ a lot. Decide which one the
     question asks for, compute it, and name it in the answer.
   - Rows with a tiny denominator make the mean of ratios unstable; check them.
3. Percentages
   - Say what the percentage is of: share of the total, change from a base, or rate
     within a group.
   - A change of a percentage is in percentage points (from 20% to 25% is +5 points,
     or +25% relative). Say which one.
   - Do not average percentages of groups of different sizes; recompute from counts.
   - Shares of a total must add up to about 100%; check it.
4. Simpson's paradox
   - A difference between groups overall can reverse within every subgroup when the
     groups have a different mix of subgroups.
   - When comparing groups, check the result within the main subgroup variable too.
     If the direction changes, report both levels and explain the mix.
5. Group sizes
   - Print the count next to every group average or rate. Averages of tiny groups are
     noisy; flag groups with few rows or set a minimum size for rankings.
6. Rankings
   - Rank on the measure the question asks for (total vs average vs rate give
     different leaders).
   - Check for ties and near-ties: if two values differ by less than the rounding
     used in the answer, say they are about equal.
   - When a ranking is the answer, show the values, not only the order.
   - Verify comparison words ("higher", "ahead of", "second") against the printed
     values before writing them.
7. Rounding
   - Compute with full precision, round only in the answer.
   - Round consistently (same number of decimals in one table), and make sure the
     rounded parts still match the rounded total, or say that they do not add up due
     to rounding.
   - Do not claim a difference that disappears after rounding.

## Typical pitfalls

- Reporting the mean of a skewed measure as the "typical" value.
- Averaging group rates instead of dividing the totals.
- Writing "+5%" for a change in percentage points.
- A ranking built from groups with very different sizes, without the counts.
- Saying one value is higher than another when the printed numbers show the opposite.

## Checklist for the final answer

- [ ] Each summary number names its statistic (mean, median, total, rate, share).
- [ ] Ratios state the definition (ratio of sums or mean of ratios).
- [ ] Percentages state their base; changes say points or relative percent.
- [ ] Group sizes are shown next to group averages, rates and rankings.
- [ ] Ties and near-ties in rankings are named.
- [ ] Comparison words match the printed numbers.
