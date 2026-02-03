# PR Grid Load Data Analysis

## Major Outages

| # | Duration | Gap Period |
|---|----------|------------|
| 1 | **10.4 days** | 2025-07-13 → 2025-07-23 |
| 2 | **4.2 days** | 2025-12-21 → 2025-12-25 (Christmas) |
| 3 | **1.5 days** | 2025-09-21 → 2025-09-23 |
| 4 | **23 hours** | 2025-11-22 → 2025-11-23 |

## Recommended Continuous Segments

| Segment | Date Range | Duration | Records |
|---------|------------|----------|---------|
| **1 (Best)** | 2025-04-05 → 2025-07-13 | 98 days | 10,251 |
| 2 | 2025-07-23 → 2025-09-21 | 59 days | 4,324 |
| 3 | 2025-09-23 → 2025-11-22 | 60 days | 3,564 |
| 4 | 2025-11-23 → 2025-12-21 | 28 days | 2,649 |
| 5 | 2025-12-25 → 2026-01-26 | 31 days | 2,477 |

---

## Autocorrelation Explained

### Mathematical Definition

Autocorrelation at lag *k* measures how correlated a time series is with itself shifted by *k* time steps:

```
         Σ (xₜ - μ)(xₜ₊ₖ - μ)
ρₖ = ────────────────────────────
              Σ (xₜ - μ)²
```

Where:
- `xₜ` = value at time t
- `μ` = mean of the series
- `k` = lag (number of time steps)
- `ρₖ` = correlation coefficient between -1 and +1

### Intuitive Explanation

**Autocorrelation answers:** "If I know the value now, how well can I predict the value *k* steps in the future?"

| ρₖ Value | Meaning |
|----------|---------|
| +1.0 | Perfect positive correlation (identical pattern) |
| +0.7 to +0.9 | Strong correlation (very predictable) |
| +0.3 to +0.7 | Moderate correlation (somewhat predictable) |
| 0 | No correlation (random, unpredictable) |
| -1.0 | Perfect negative correlation (inverted pattern) |

### PR Grid Data Autocorrelation

```
Lag 1h:   0.945  ←── "Demand in 1 hour will be very similar to now"
Lag 2h:   0.846  ←── "Still very predictable 2 hours out"
Lag 6h:   0.309  ←── "6 hours out, much less predictable"
Lag 24h:  0.729  ←── "Same hour TOMORROW is predictable!" (daily pattern)
```

---

## Daily Pattern Explained

### What It Means

The **0.729 correlation at lag 24h** means:

> "The demand at 3pm today is strongly correlated with demand at 3pm yesterday"

This is the **daily pattern** or **diurnal cycle** - electricity demand follows predictable daily rhythms:
- Low at night (people sleeping)
- Rising in morning (waking up, businesses open)
- Peak in afternoon/evening (AC, cooking, activities)
- Falling at night

### Visual Intuition

```
Day 1:    ___/‾‾‾\___/‾‾‾‾\____
Day 2:    ___/‾‾‾\___/‾‾‾‾\____   ← Similar shape!
Day 3:    ___/‾‾‾\___/‾‾‾‾\____
                ↑
         Daily pattern repeats
```

The 0.729 correlation means the pattern repeats ~73% consistently day-to-day.

### Why This Matters for Forecasting

1. **High 1h autocorrelation (0.945):** Short-term forecasts are reliable
2. **High 24h autocorrelation (0.729):** The model can learn daily patterns
3. **Low 168h/weekly autocorrelation (0.144):** Weekly patterns are weak - weekdays vs weekends aren't very different in this data

This is **excellent** for univariate forecasting - the signal has strong, learnable structure.

---

## Fill Rate Analysis

| Interval | Fill Rate | Recommendation |
|----------|-----------|----------------|
| 5 min    | 27%       | Poor - too granular |
| 10 min   | 47%       | Poor |
| 15 min   | 60%       | Fair |
| 20 min   | 69%       | Fair |
| **30 min** | **79%** | **Good** |
| **1 hour** | **89%** | **Good** |

## Recommendations

- **30-minute intervals:** 79% fill rate, good balance of granularity and coverage
- **1-hour intervals:** 89% fill rate, best for stable forecasting with Chronos-2
