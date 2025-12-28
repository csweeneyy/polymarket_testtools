# Polymarket New Market Scanner

Scans Polymarket for potentially mispriced markets and uses an LLM to analyze betting opportunities.

## The Strategy

New markets on Polymarket are often mispriced for the first 24-48 hours because:
- Thin liquidity means early trades set arbitrary prices
- Domain experts haven't found the market yet
- Low volume = low attention = inefficient pricing

This scanner finds new, low-volume markets with tight spreads and uses an LLM to estimate if they're mispriced.

## Setup

### 1. Install dependencies

```bash
pip install requests
```

### 2. Set your API key

**Gemini (free tier):**
```bash
export GEMINI_API_KEY="your-key-from-aistudio.google.com"
```

**Or Anthropic/OpenAI:**
```bash
export ANTHROPIC_API_KEY="your-key"
# or
export OPENAI_API_KEY="your-key"
```

Then change `"llm_provider"` in the CONFIG to `"anthropic"` or `"openai"`.

### 3. Run

```bash
# Single scan
python scanner.py --once

# Continuous scanning (every 5 min)
python scanner.py

# List available categories
python scanner.py --categories
```

## Configuration

Edit the `CONFIG` dict at the top of `scanner.py`:

```python
CONFIG = {
    # Categories to watch (empty = all)
    "watch_categories": [],
    
    # Market filters
    "max_market_age_hours": 48,      # Only markets < 48h old
    "max_volume_usd": 50000,         # Only markets < $50k volume
    "min_volume_usd": 0,             # Minimum volume (filter dead markets)
    "max_spread": 0.15,              # Max 15% spread (filter illiquid garbage)
    "min_price": 0.05,               # Ignore < 5% (probably accurate)
    "max_price": 0.95,               # Ignore > 95% (probably accurate)
    
    # Scan limits
    "max_markets_to_analyze": 50,    # Stop after 50 markets (post-filter)
    
    # Output
    "output_csv": "scan_results.csv",
    "seen_markets_file": "seen_markets.json",
    
    # LLM settings
    "use_llm": True,
    "llm_provider": "gemini",        # "gemini", "anthropic", or "openai"
    "llm_delay_seconds": 5,          # Delay between API calls (rate limiting)
}
```

## Understanding Spread

The spread is the difference between bid and ask prices. It represents your cost to enter AND exit a position.

```
YES: Bid $0.35 / Ask $0.42 | Spread: 7%
```

- **Bid $0.35** = Price someone will pay to buy your YES shares (what you get if you sell)
- **Ask $0.42** = Price to buy YES shares (what you pay)
- **Spread 7%** = You lose 7% just entering and exiting

A market showing "50% YES" with bid 0.30 / ask 0.70 has a 40% spread - you'd need massive edge just to break even. The scanner filters these out.

**Recommended max_spread values:**
- `0.10` (10%) - Very tight, fewer results but highly tradeable
- `0.15` (15%) - Good balance (default)
- `0.20` (20%) - More results, higher friction

## Output

### Terminal (streaming)

Results display as they're analyzed:

```
[1/50] Analyzing: Will xQc beat Forsen's speedrun record...

================================================================================
[1] 📊 Will xQc beat Forsen's Minecraft speedrun record in 2025?
    YES: Bid $0.07 / Ask $0.09 | Spread: 2.0%
    NO:  Bid $0.91 / Ask $0.93
    Volume: $58,552 | Age: 18.3h
    🔗 https://polymarket.com/event/will-xqc-beat-forsens...

    🟢 LLM Score: 8/10 | Rec: NO | Conf: high
    📈 Estimated edge: 19.0%
    💭 Forsen's 15:28 record has stood for 2+ years...
```

### CSV

All results append to `scan_results.csv`:

| timestamp | question | yes_bid | yes_ask | spread | volume | age_hours | llm_score | llm_recommendation | llm_confidence | llm_edge | llm_reasoning | url |
|-----------|----------|---------|---------|--------|--------|-----------|-----------|-------------------|----------------|----------|---------------|-----|

Import into Google Sheets or Excel for analysis.

## Files

- `scanner.py` - Main scanner script
- `seen_markets.json` - Tracks analyzed markets (prevents duplicates)
- `scan_results.csv` - All results with LLM analysis

## Rate Limiting

**Gemini free tier:** ~15 requests/minute
- Default `llm_delay_seconds: 5` is safe
- If you hit rate limits, increase to 6-8

**Anthropic/OpenAI:** Higher limits but costs money
- Can reduce delay to 1-2 seconds

## Tips

1. **Delete `seen_markets.json`** to re-analyze all markets
2. **Set `"use_llm": false`** to just see filtered markets without LLM analysis (faster, no API costs)
3. **Tighten `max_spread`** if you want only highly liquid markets
4. **Increase `max_market_age_hours`** if you want more results (but older = more likely already efficiently priced)

## Disclaimer

This is a tool for research and education. Prediction markets involve real money and risk. The LLM analysis is not financial advice - it's a starting point for your own research. Always read the resolution criteria before betting.