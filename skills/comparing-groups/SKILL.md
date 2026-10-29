---
name: comparing-groups
description: Use when the question asks whether two or more groups differ - in an average, a typical value, a distribution or a proportion - and whether the difference is meaningful.
---

# Comparing groups checklist

Easy to forget:
- Print n, mean, median and spread per group first; check skewness and outliers.
- Pick the test by what is compared: means - Welch's t-test; skewed data or ranks -
  Mann-Whitney U (compares distributions, not means); proportions - chi-square or
  Fisher's exact; 3+ groups - ANOVA or Kruskal-Wallis.
- Always add an effect size: the difference in units and percent, plus Cohen's d or
  rank-biserial r = 1 - 2U/(n1*n2); for proportions, percentage points.
- Large groups make tiny differences significant: judge by the effect size.
- Small groups: "not significant" is not "no difference" - give the observed gap.
- Several tests: apply Bonferroni or say how many tests were run.
- Check an obvious confounder in the data (compare within its levels) or say it was
  not controlled. No causal claims from observational data.
- Drop missing values per comparison and say how many.

The answer must state:
- [ ] group sizes and rows dropped;
- [ ] mean and median per group;
- [ ] the test's name, what it compares, and the p-value;
- [ ] the effect size and whether the difference matters in practice;
- [ ] caveats: skewness, multiple tests, confounders.
