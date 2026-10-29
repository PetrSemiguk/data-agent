---
name: aggregation-pitfalls
description: Use when the answer is built from averages, ratios, percentages, shares or rankings of groups - to avoid wrong averages, misleading ratios and unstable rankings.
---

# Aggregation pitfalls checklist

Easy to forget:
- Mean vs median: if the mean is far above the median, a few large values drive it;
  report the median as the typical value, or both. Name the statistic.
- Mean of ratios vs ratio of sums: they answer different questions (each row equally
  vs weighted by size). Choose the one the question asks for and name it; rows with a
  tiny denominator distort the mean of ratios.
- Percentages: say what they are a percentage of. A change of a percentage is in
  points (20% -> 25% is +5 points, +25% relative). Never average percentages of
  groups of different size - recompute from counts. Shares should sum to ~100%.
- Simpson's paradox: when comparing groups, check the main subgroup variable too; if
  the direction flips, report both levels.
- Show the count next to every group average, rate or rank; flag tiny groups.
- Rankings: rank on the measure asked for (total, average and rate give different
  leaders); name ties and near-ties; check every comparison word ("higher", "second",
  "ahead of") against the printed numbers.
- Round only in the answer, consistently; do not claim a difference that disappears
  after rounding.

The answer must state:
- [ ] the statistic behind every summary number and the definition of each ratio;
- [ ] the base of every percentage, and points vs relative change;
- [ ] group sizes next to group values and rankings.
