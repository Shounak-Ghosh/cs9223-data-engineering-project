# Replay benchmark: sembench-movie

Objective `input_tokens`; 20 seeds; k in [10, 20]; cheap-mode guard rows [3, 5, 10] (`g`); agreement threshold 0.95. No API calls.

## acting_quality AND positive

Gold positives: 14

| Plan (all rows) | Total input_tokens | Calls | F1 |
| --- | --- | --- | --- |
| `sequential:acting_quality,positive` | 36,362 | 116 | 0.727 |
| `sequential:positive,acting_quality` | 50,889 | 164 | 0.727 |
| `combined:acting_quality,positive` | 32,956 | 100 | 0.727 |
| `independent:acting_quality,positive` (always separate) | 62,212 | 200 | 0.727 |

| Mode, k | Picked oracle plan | Mean sampling cost | Mean total | vs always-combined | vs always-separate | vs best order | vs worst order | Prediction error | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full-k10 | 90% | 9,546 | 39,554 | +20.0% | -36.4% | +8.8% | -22.3% | -1.2% | 0.727 |
| full-k20 | 95% | 19,070 | 45,594 | +38.3% | -26.7% | +25.4% | -10.4% | -0.3% | 0.727 |
| cheap-g3-k10 | 90% | 5,195 | 35,419 | +7.5% | -43.1% | -2.6% | -30.4% | -1.1% | 0.727 |
| cheap-g3-k20 | 100% | 8,481 | 34,833 | +5.7% | -44.0% | -4.2% | -31.6% | +0.2% | 0.727 |
| cheap-g5-k10 | 90% | 6,435 | 36,597 | +11.0% | -41.2% | +0.6% | -28.1% | -1.1% | 0.727 |
| cheap-g5-k20 | 95% | 9,728 | 36,501 | +10.8% | -41.3% | +0.4% | -28.3% | -0.3% | 0.727 |
| cheap-g10-k10 | 90% | 9,546 | 39,554 | +20.0% | -36.4% | +8.8% | -22.3% | -1.2% | 0.727 |
| cheap-g10-k20 | 95% | 12,837 | 39,516 | +19.9% | -36.5% | +8.7% | -22.3% | -0.3% | 0.727 |

| Mode, k | vs combined @100 | vs combined @1,000 | vs combined @10,000 | vs best order @100 | vs best order @1,000 | vs best order @10,000 | Break-even vs best order | Break-even vs always-separate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full-k10 | +19.9% | +2.9% | +1.2% | +8.7% | -6.7% | -8.3% | 203 rows | 21 rows |
| full-k20 | +38.3% | +4.3% | +0.9% | +25.3% | -5.5% | -8.6% | 385 rows | 43 rows |
| cheap-g3-k10 | +7.4% | +1.7% | +1.1% | -2.7% | -7.9% | -8.4% | 68 rows | 7 rows |
| cheap-g3-k20 | +5.7% | +0.6% | +0.1% | -4.2% | -8.8% | -9.3% | 55 rows | 6 rows |
| cheap-g5-k10 | +10.9% | +2.0% | +1.1% | +0.5% | -7.5% | -8.3% | 106 rows | 11 rows |
| cheap-g5-k20 | +10.7% | +1.5% | +0.6% | +0.3% | -8.0% | -8.8% | 104 rows | 12 rows |
| cheap-g10-k10 | +19.9% | +2.9% | +1.2% | +8.7% | -6.7% | -8.3% | 203 rows | 21 rows |
| cheap-g10-k20 | +19.9% | +2.5% | +0.7% | +8.7% | -7.1% | -8.7% | 197 rows | 22 rows |
