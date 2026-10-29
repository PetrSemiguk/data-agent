---
name: time-series
description: Use when the question is about change over time - trends, seasonality, growth, comparing periods, or the best and worst months, weeks or years.
---

# Time series checklist

Easy to forget:
- Parse dates with an explicit format; count failures; check day and month are not
  swapped.
- Print the exact first and last date. The first and last period (month, year) are
  often INCOMPLETE: check by dates or days with data per period. Never compare an
  incomplete period with full ones without saying so - drop it, compare the same
  span (e.g. Jan-Jun of each year), or use per-day values.
- Reindex to a full calendar and count missing periods inside the range; a missing
  period is not zero.
- Sum for totals and counts, mean for levels and measurements.
- Seasonality: compare the same period across complete years (share of the year or
  mean per month) and check that the pattern repeats every year, not only on average.
- Year over year: compare the same months in both years.
- Trend: use a rolling mean over a full cycle (12 months, 52 weeks) or remove the
  seasonal pattern first; never read a trend from two points; say "steady" only if
  most steps go the same way.

The answer must state:
- [ ] the exact date range and which periods are incomplete, and how they were handled;
- [ ] the aggregation step and sum vs mean;
- [ ] seasonal highs and lows and whether they repeat every year;
- [ ] the trend's direction and size, from comparable periods.
