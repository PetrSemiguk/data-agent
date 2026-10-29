---
name: time-series
description: Use when the question is about change over time - trends, seasonality, growth, comparing periods, or the best and worst months, weeks or years.
---

# Time series analysis

Goal: describe how a measure changes over time without being misled by incomplete
periods, gaps or a badly chosen aggregation step.

## Method

1. Parse the dates
   - Convert the date column on a copy with an explicit format when it is known,
     e.g. `pd.to_datetime(data["date"], format="%Y-%m-%d", errors="coerce")`.
   - Count values that failed to parse. Check that day and month are not swapped:
     look at the maximum day and month values.
2. Check the range
   - Print the minimum and maximum date and the number of distinct days.
   - Note the first and last date exactly; they decide which periods are complete.
3. Choose the aggregation step
   - Daily data is noisy; weeks or months usually show the pattern better.
   - The step must be longer than the gaps in the data and short enough to show the
     pattern that is asked about (seasonality within a year needs months or weeks).
   - Aggregate with `resample` or `groupby` on a period, and decide between sum
     (totals, counts) and mean (levels, rates, measurements).
4. Incomplete periods at the edges
   - The first and the last period are often incomplete: the data starts or ends in
     the middle of a month or year.
   - Check it by dates, not by eye: compare the first and last date with the start
     and end of their periods, or count days with data per period.
   - Never compare an incomplete period with complete ones without saying so. Either
     drop it, compare the same span (e.g. January-June of each year), or normalize
     (per day).
5. Missing periods inside the range
   - Reindex to a complete calendar (`asfreq` or `reindex` with `date_range`) and count
     periods with no data. A missing period is not the same as zero.
6. Seasonality
   - Compare the same period across years: the share of each month in its year, or the
     average of each month over complete years.
   - Check that the pattern repeats in every year, not only on average.
   - For year-over-year comparisons, use the same months in both years.
7. Trend
   - Use a rolling mean over a full seasonal cycle (e.g. 12 months, 52 weeks) so the
     season cancels out; it needs a full window, so the first values are missing.
   - Compare complete years or the same months across years.
   - Describe the direction and the size of the change with numbers; say "steady" only
     if most steps go the same way.
8. A chart with time on the x-axis usually helps; mark incomplete periods.

## Typical pitfalls

- Calling the last year "a decline" when it simply has fewer months of data.
- Ranking months over all years when some months appear in more years than others.
- Comparing totals of periods with different numbers of days.
- Treating a missing period as zero, or interpolating it without saying so.
- Reading a trend from two points (first vs last period).
- Mixing up the order of day and month when parsing dates.

## Checklist for the final answer

- [ ] The exact date range of the data.
- [ ] Which periods are incomplete and how they were handled.
- [ ] The aggregation step and whether values are sums or means.
- [ ] Missing periods inside the range, if any.
- [ ] Seasonality: which periods are high or low, and whether it repeats every year.
- [ ] Trend: direction and size, based on comparable periods.
