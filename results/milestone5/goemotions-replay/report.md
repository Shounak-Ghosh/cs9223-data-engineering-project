# Replay benchmark: goemotions

Objective `input_tokens`; 20 seeds; k in [10, 20]; cheap-mode guard rows [3, 5, 10] (`g`); agreement threshold 0.95. No API calls.

## gratitude AND admiration

Gold positives: 0

| Plan (all rows) | Total input_tokens | Calls | F1 |
| --- | --- | --- | --- |
| `sequential:gratitude,admiration` | 31,630 | 109 | 0.000 |
| `sequential:admiration,gratitude` | 36,164 | 124 | 0.000 |
| `combined:gratitude,admiration` | 30,697 | 100 | 0.000 |
| `independent:gratitude,admiration` (always separate) | 58,194 | 200 | 0.000 |

| Mode, k | Picked oracle plan | Mean sampling cost | Mean total | vs always-combined | vs always-separate | vs best order | vs worst order | Prediction error | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full-k10 | 75% | 8,873 | 36,781 | +19.8% | -36.8% | +16.3% | +1.7% | -2.5% | 0.000 |
| full-k20 | 45% | 17,773 | 42,917 | +39.8% | -26.3% | +35.7% | +18.7% | -3.4% | 0.000 |
| cheap-g3-k10 | 75% | 4,813 | 33,228 | +8.2% | -42.9% | +5.1% | -8.1% | -2.5% | 0.000 |
| cheap-g3-k20 | 55% | 7,879 | 35,186 | +14.6% | -39.5% | +11.2% | -2.7% | -3.0% | 0.000 |
| cheap-g5-k10 | 75% | 5,969 | 34,242 | +11.5% | -41.2% | +8.3% | -5.3% | -2.6% | 0.000 |
| cheap-g5-k20 | 45% | 9,042 | 36,635 | +19.3% | -37.0% | +15.8% | +1.3% | -3.3% | 0.000 |
| cheap-g10-k10 | 75% | 8,873 | 36,781 | +19.8% | -36.8% | +16.3% | +1.7% | -2.5% | 0.000 |
| cheap-g10-k20 | 55% | 11,961 | 38,355 | +24.9% | -34.1% | +21.3% | +6.1% | -2.9% | 0.000 |

| Mode, k | vs combined @100 | vs combined @1,000 | vs combined @10,000 | vs best order @100 | vs best order @1,000 | vs best order @10,000 | Break-even vs best order | Break-even vs always-separate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full-k10 | +19.6% | +2.6% | +0.9% | +16.1% | -0.4% | -2.0% | 826 rows | 21 rows |
| full-k20 | +39.2% | +5.4% | +2.0% | +35.1% | +2.3% | -1.0% | 2,746 rows | 43 rows |
| cheap-g3-k10 | +8.0% | +1.5% | +0.8% | +4.8% | -1.5% | -2.1% | 318 rows | 8 rows |
| cheap-g3-k20 | +14.1% | +2.6% | +1.5% | +10.8% | -0.4% | -1.5% | 763 rows | 14 rows |
| cheap-g5-k10 | +11.3% | +1.8% | +0.9% | +8.0% | -1.2% | -2.1% | 463 rows | 12 rows |
| cheap-g5-k20 | +18.8% | +3.4% | +1.8% | +15.3% | +0.3% | -1.2% | 1,250 rows | 19 rows |
| cheap-g10-k10 | +19.6% | +2.6% | +0.9% | +16.1% | -0.4% | -2.0% | 826 rows | 21 rows |
| cheap-g10-k20 | +24.5% | +3.7% | +1.6% | +20.8% | +0.6% | -1.4% | 1,381 rows | 26 rows |

## admiration AND optimism

Gold positives: 0

| Plan (all rows) | Total input_tokens | Calls | F1 |
| --- | --- | --- | --- |
| `sequential:admiration,optimism` | 36,236 | 124 | 0.000 |
| `sequential:optimism,admiration` | 32,196 | 110 | 0.000 |
| `combined:admiration,optimism` | 30,997 | 100 | 0.000 |
| `independent:admiration,optimism` (always separate) | 58,494 | 200 | 0.000 |

| Mode, k | Picked oracle plan | Mean sampling cost | Mean total | vs always-combined | vs always-separate | vs best order | vs worst order | Prediction error | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full-k10 | 70% | 8,933 | 37,246 | +20.2% | -36.3% | +15.7% | +2.8% | -3.2% | 0.000 |
| full-k20 | 60% | 17,893 | 43,206 | +39.4% | -26.1% | +34.2% | +19.2% | -2.5% | 0.000 |
| cheap-g3-k10 | 70% | 4,852 | 33,776 | +9.0% | -42.3% | +4.9% | -6.8% | -3.0% | 0.000 |
| cheap-g3-k20 | 60% | 7,948 | 35,332 | +14.0% | -39.6% | +9.7% | -2.5% | -2.8% | 0.000 |
| cheap-g5-k10 | 70% | 6,014 | 34,764 | +12.2% | -40.6% | +8.0% | -4.1% | -3.1% | 0.000 |
| cheap-g5-k20 | 60% | 9,117 | 36,265 | +17.0% | -38.0% | +12.6% | +0.1% | -2.6% | 0.000 |
| cheap-g10-k10 | 70% | 8,933 | 37,246 | +20.2% | -36.3% | +15.7% | +2.8% | -3.2% | 0.000 |
| cheap-g10-k20 | 70% | 12,051 | 38,136 | +23.0% | -34.8% | +18.4% | +5.2% | -2.0% | 0.000 |

| Mode, k | vs combined @100 | vs combined @1,000 | vs combined @10,000 | vs best order @100 | vs best order @1,000 | vs best order @10,000 | Break-even vs best order | Break-even vs always-separate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full-k10 | +19.9% | +3.0% | +1.3% | +15.4% | -0.8% | -2.4% | 691 rows | 21 rows |
| full-k20 | +39.0% | +5.3% | +1.9% | +33.8% | +1.4% | -1.9% | 1,612 rows | 43 rows |
| cheap-g3-k10 | +8.7% | +1.9% | +1.2% | +4.6% | -1.9% | -2.5% | 277 rows | 9 rows |
| cheap-g3-k20 | +13.6% | +2.7% | +1.7% | +9.3% | -1.1% | -2.1% | 517 rows | 14 rows |
| cheap-g5-k10 | +11.9% | +2.2% | +1.3% | +7.7% | -1.6% | -2.5% | 395 rows | 12 rows |
| cheap-g5-k20 | +16.6% | +3.0% | +1.7% | +12.2% | -0.8% | -2.1% | 647 rows | 17 rows |
| cheap-g10-k10 | +19.9% | +3.0% | +1.3% | +15.4% | -0.8% | -2.4% | 691 rows | 21 rows |
| cheap-g10-k20 | +22.7% | +3.3% | +1.4% | +18.1% | -0.5% | -2.4% | 795 rows | 25 rows |

## gratitude AND optimism

Gold positives: 1

| Plan (all rows) | Total input_tokens | Calls | F1 |
| --- | --- | --- | --- |
| `sequential:gratitude,optimism` | 31,639 | 109 | 0.667 |
| `sequential:optimism,gratitude` | 32,176 | 110 | 0.667 |
| `combined:gratitude,optimism` | 30,797 | 100 | 0.500 |
| `independent:gratitude,optimism` (always separate) | 58,294 | 200 | 0.667 |

| Mode, k | Picked oracle plan | Mean sampling cost | Mean total | vs always-combined | vs always-separate | vs best order | vs worst order | Prediction error | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full-k10 | 45% | 8,893 | 37,290 | +21.1% | -36.0% | +17.9% | +15.9% | -4.9% | 0.592 |
| full-k20 | 30% | 17,813 | 43,244 | +40.4% | -25.8% | +36.7% | +34.4% | -4.6% | 0.617 |
| cheap-g3-k10 | 45% | 4,826 | 34,340 | +11.5% | -41.1% | +8.5% | +6.7% | -4.6% | 0.592 |
| cheap-g3-k20 | 35% | 7,902 | 36,538 | +18.6% | -37.3% | +15.5% | +13.6% | -4.4% | 0.608 |
| cheap-g5-k10 | 45% | 5,984 | 35,180 | +14.2% | -39.7% | +11.2% | +9.3% | -4.9% | 0.592 |
| cheap-g5-k20 | 35% | 9,067 | 37,324 | +21.2% | -36.0% | +18.0% | +16.0% | -4.3% | 0.608 |
| cheap-g10-k10 | 45% | 8,893 | 37,290 | +21.1% | -36.0% | +17.9% | +15.9% | -4.9% | 0.592 |
| cheap-g10-k20 | 35% | 11,991 | 39,245 | +27.4% | -32.7% | +24.0% | +22.0% | -3.9% | 0.608 |

| Mode, k | vs combined @100 | vs combined @1,000 | vs combined @10,000 | vs best order @100 | vs best order @1,000 | vs best order @10,000 | Break-even vs best order | Break-even vs always-separate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full-k10 | +20.6% | +3.8% | +2.1% | +17.4% | +1.0% | -0.6% | 2,352 rows | 21 rows |
| full-k20 | +39.7% | +6.0% | +2.6% | +35.9% | +3.2% | -0.1% | 7,930 rows | 43 rows |
| cheap-g3-k10 | +11.0% | +2.9% | +2.0% | +8.1% | +0.1% | -0.7% | 1,146 rows | 10 rows |
| cheap-g3-k20 | +17.9% | +3.6% | +2.2% | +14.8% | +0.9% | -0.5% | 2,284 rows | 18 rows |
| cheap-g5-k10 | +13.8% | +3.1% | +2.1% | +10.7% | +0.4% | -0.7% | 1,489 rows | 14 rows |
| cheap-g5-k20 | +20.5% | +3.9% | +2.2% | +17.3% | +1.1% | -0.5% | 2,651 rows | 21 rows |
| cheap-g10-k10 | +20.6% | +3.8% | +2.1% | +17.4% | +1.0% | -0.6% | 2,352 rows | 21 rows |
| cheap-g10-k20 | +26.8% | +4.4% | +2.2% | +23.4% | +1.7% | -0.5% | 3,171 rows | 28 rows |

## gratitude AND admiration AND optimism

Gold positives: 0

| Plan (all rows) | Total input_tokens | Calls | F1 |
| --- | --- | --- | --- |
| `sequential:gratitude,admiration,optimism` | 32,504 | 112 | 0.000 |
| `sequential:gratitude,optimism,admiration` | 32,221 | 111 | 0.000 |
| `sequential:admiration,gratitude,optimism` | 37,038 | 127 | 0.000 |
| `sequential:admiration,optimism,gratitude` | 37,661 | 129 | 0.000 |
| `sequential:optimism,gratitude,admiration` | 32,758 | 112 | 0.000 |
| `sequential:optimism,admiration,gratitude` | 33,621 | 115 | 0.000 |
| `combined:gratitude,admiration,optimism` | 32,497 | 100 | 0.000 |
| `independent:gratitude,admiration,optimism` (always separate) | 87,491 | 300 | 0.000 |

| Mode, k | Picked oracle plan | Mean sampling cost | Mean total | vs always-combined | vs always-separate | vs best order | vs worst order | Prediction error | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full-k10 | 5% | 11,977 | 41,802 | +28.6% | -52.2% | +29.7% | +11.0% | -7.7% | 0.000 |
| full-k20 | 10% | 23,991 | 50,526 | +55.5% | -42.2% | +56.8% | +34.2% | -7.3% | 0.000 |
| cheap-g3-k10 | 5% | 5,873 | 37,320 | +14.8% | -57.3% | +15.8% | -0.9% | -7.3% | 0.000 |
| cheap-g3-k20 | 10% | 9,116 | 40,010 | +23.1% | -54.3% | +24.2% | +6.2% | -7.5% | 0.000 |
| cheap-g5-k10 | 0% | 7,611 | 38,742 | +19.2% | -55.7% | +20.2% | +2.9% | -7.8% | 0.000 |
| cheap-g5-k20 | 10% | 10,865 | 41,487 | +27.7% | -52.6% | +28.8% | +10.2% | -7.4% | 0.000 |
| cheap-g10-k10 | 5% | 11,977 | 41,802 | +28.6% | -52.2% | +29.7% | +11.0% | -7.7% | 0.000 |
| cheap-g10-k20 | 10% | 15,253 | 44,504 | +36.9% | -49.1% | +38.1% | +18.2% | -7.2% | 0.000 |

| Mode, k | vs combined @100 | vs combined @1,000 | vs combined @10,000 | vs best order @100 | vs best order @1,000 | vs best order @10,000 | Break-even vs best order | Break-even vs always-separate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full-k10 | +27.9% | +3.9% | +1.4% | +29.0% | +4.7% | +2.3% | never rows | 16 rows |
| full-k20 | +54.3% | +5.9% | +1.1% | +55.6% | +6.8% | +2.0% | never rows | 32 rows |
| cheap-g3-k10 | +14.2% | +2.3% | +1.1% | +15.1% | +3.2% | +2.0% | never rows | 8 rows |
| cheap-g3-k20 | +21.9% | +2.7% | +0.8% | +23.0% | +3.6% | +1.6% | never rows | 13 rows |
| cheap-g5-k10 | +18.5% | +2.9% | +1.4% | +19.5% | +3.8% | +2.3% | never rows | 10 rows |
| cheap-g5-k20 | +26.5% | +3.2% | +0.8% | +27.5% | +4.0% | +1.7% | never rows | 15 rows |
| cheap-g10-k10 | +27.9% | +3.9% | +1.4% | +29.0% | +4.7% | +2.3% | never rows | 16 rows |
| cheap-g10-k20 | +35.7% | +4.1% | +0.9% | +36.9% | +5.0% | +1.8% | never rows | 21 rows |
