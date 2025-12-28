# Whale PnL-Weighted Position Strategy

A backtesting framework to test whether **trader quality predicts market outcomes** on Polymarket.

## The Hypothesis

If traders with high lifetime PnL are holding positions, the market is more likely to resolve in favor of their position. By tracking their behavior at different stages of a market's lifecycle, we can identify "Smart Money" signals.

## Key Features (v2.0)

1.  **Multi-Snapshot Analysis**: Instead of a single check, we now analyze whale positioning at 4 key moments:
    *   **50pct** (Midpoint): Early conviction.
    *   **25pctRem** (75% Duration): Late game positioning.
    *   **10pctRem** (90% Duration): End game positioning.
    *   **24hPre** (24h before Resolution): Final conviction.

2.  **Dynamic Side Calculation**: 
    *   The script reconstructs a whale's *exact* net position (YES vs NO) at the specific snapshot time using their historical trade data.
    *   It does *not* assume their final position was held throughout the entire market.

3.  **Robust Pagination**:
    *   Automatically scans thousands of markets to find exactly `N` valid, high-volume markets that meet your criteria.
    *   Filters out "future-dated" snapshots to prevent data leakage.

4.  **Smart Category Filtering**:
    *   Finds markets even if API tags are missing by scanning keywords (e.g. "Trump", "NFL", "Bitcoin").

## Signal Formula
For each snapshot time $T$:

1.  **Identify Whales**: Get top 20 holders.
2.  **Reconstruct Position**: Calculate `Held_YES` and `Held_NO` at time $T$.
3.  **Determine Side**: If `Held_YES > Held_NO`, side is YES.
4.  **Weight by Quality**: `Weight = Log10(Lifetime_PnL + 1)`
5.  **Compute Signal**:
    $$ \text{Signal}_T = \frac{\sum \text{Weight}_{\text{YES}} - \sum \text{Weight}_{\text{NO}}}{|\sum \text{Weight}_{\text{YES}}| + |\sum \text{Weight}_{\text{NO}}|} $$

Range: **-1.0 (Strong NO)** to **+1.0 (Strong YES)**.

## Usage

### 1. Install Dependencies
```bash
pip install requests
```

### 2. Run the Backtest
```bash
# Analyze 50 political markets
python whale_tracker.py --markets 50 --category Politics

# Analyze 100 sports markets with custom volume filter
python whale_tracker.py --markets 100 --category Sports --min-volume 50000

# Full unconstrained run
python whale_tracker.py --markets 100
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--markets` | 50 | Number of *analyzed* markets to produce |
| `--category` | None | Filter by "Politics", "Sports", "Crypto", "Science", "Pop-Culture" |
| `--min-volume` | 10000 | Minimum volume filter (USD) |
| `--holders` | 20 | Number of top holders per side to analyze |
| `--days` | 90 | Lookback period for resolved markets |
| `--output` | backtest_results.csv | Output CSV file (auto-increments) |

## Output Explanation

### Console Output
For each market, you will see a detailed breakdown:
```
[1/50] Analyzing: Will Zohran Mamdani win?...
  Resolution: NO
  Snapshot 50pct     : Signal= 0.646 -> Predicted YES  (Actual NO) ✗ WRONG
  Snapshot 25pctRem  : Signal= 0.792 -> Predicted YES  (Actual NO) ✗ WRONG
  Snapshot 10pctRem  : Signal=-1.000 -> Predicted NO   (Actual NO) ✓ CORRECT
  Snapshot 24hPre    : Signal=-1.000 -> Predicted NO   (Actual NO) ✓ CORRECT
```

### Summary Stats
At the end, you get a performance report for each timeframe:
```
--- SNAPSHOT: 10pctRem ---
Markets: 50
Accuracy: 58.0%
Correlation: 0.142
Avg Signal: 0.365
```

### CSV Output
The CSV contains columns for further analysis:
*   `Question`, `Resolution`, `Volume`
*   `Signal_50pct`, `Signal_25pctRem`, `Signal_10pctRem`, `Signal_24hPre`

## Interpreting "Correlation"
*   **Positive (> 0.1)**: Signal aligns with outcome (High Signal = YES, Low Signal = NO).
*   **Zero (0.0)**: Usually means **no variance** in your dataset (e.g. you analyzed 5 markets and they ALL resolved NO). You need a mix of outcomes to calculate correlation.
*   **Negative (< 0)**: Signal is a counter-indicator (Whales are wrong).

## Next Steps
If you find a timeframe (e.g. `10pctRem`) with **Accuracy > 60%** and **Correlation > 0.2**:
1.  Strategy works!
2.  Build a live scanner that alerts when whalemoney signal crosses 0.5 at that specific timeframe.
