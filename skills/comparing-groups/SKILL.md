---
name: comparing-groups
description: Use when the question asks whether two or more groups differ - in an average, a typical value, a distribution or a proportion - and whether the difference is meaningful.
---

# Comparing groups

Goal: say whether groups differ, by how much, and how sure we can be, with a test
that fits the data and an effect size that tells whether the difference matters.

## Method

1. Define the groups and the measure
   - Name the grouping column and the measure; count rows per group.
   - Count missing values of the measure per group and drop them per comparison
     (state how many).
2. Describe before testing
   - Per group: n, mean, median, standard deviation, quartiles.
   - Look at the shape: skewness, outliers, a few values far out. A histogram or
     box plot per group helps.
3. Choose the test by what is compared
   - Means of two groups, roughly symmetric data or large groups: Welch's t-test
     (`scipy.stats.ttest_ind(a, b, equal_var=False)`).
   - Typical values / ranks with skewed data or outliers: Mann-Whitney U
     (`scipy.stats.mannwhitneyu(a, b)`); it compares distributions (whether values in
     one group tend to be larger), not means.
   - Whole distributions: Kolmogorov-Smirnov (`scipy.stats.ks_2samp`).
   - Proportions or counts across categories: chi-square on the contingency table
     (`scipy.stats.chi2_contingency`); Fisher's exact test for small 2x2 tables.
   - More than two groups: one-way ANOVA (means) or Kruskal-Wallis (ranks), then
     pairwise tests for the pairs that matter.
4. Effect size: always report one
   - Difference in means or medians, in the units of the measure and in percent.
   - Standardized: Cohen's d = difference in means / pooled standard deviation
     (about 0.2 small, 0.5 medium, 0.8 large).
   - For Mann-Whitney: rank-biserial correlation r = 1 - 2U / (n1 * n2).
   - For proportions: the difference in percentage points and the ratio.
5. Significance
   - Report the p-value of the chosen test.
   - With very large groups, tiny differences become significant; judge the size by
     the effect, not the p-value.
   - With small groups, "not significant" often means "not enough data", not "no
     difference". Say so and give the observed difference.
6. Several comparisons
   - Testing many pairs or many measures inflates false positives. Use a Bonferroni
     correction (multiply p-values by the number of tests) or say how many tests were run.
7. Confounding
   - A difference between groups may come from another variable that differs between
     them (size, type, period). If an obvious one exists in the data, compare within its
     levels too, or say that it was not controlled.

## Typical pitfalls

- Using a t-test on heavily skewed data and calling it a comparison of typical values.
- Reporting a p-value without the group sizes or the size of the difference.
- Saying "no effect" because p > 0.05.
- Saying "large effect" because p is tiny.
- Claiming causation from an observational comparison.
- Testing many subgroups and reporting only the significant one.

## Checklist for the final answer

- [ ] Groups and their sizes (n per group), and rows dropped for missing values.
- [ ] Descriptive numbers per group: mean and median at least.
- [ ] The test by name, what it compares (means, distributions, proportions), and
      the p-value.
- [ ] The effect size: the difference in units and percent, plus a standardized measure.
- [ ] A plain conclusion: is the difference meaningful in size, not only significant.
- [ ] Caveats: skewness, multiple tests, possible confounders, no causal claim.
