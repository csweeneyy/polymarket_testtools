"""
Polymarket New Market Scanner
=============================
Scans for new/mispriced markets on Polymarket and uses an LLM to rate betting opportunities.

Based on the strategy of finding inefficiently priced new markets before smart money arrives.

Author: Built with Claude
"""

import os
import json
import time
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
import requests

# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================

CONFIG = {
    # Categories you understand well (your edge)
    # Leave empty [] to watch ALL categories
    "watch_categories": [],
    
    # Market age filter (in hours) - only show markets newer than this
    "max_market_age_hours": 168,
    
    # Volume filter - target low volume markets (more likely mispriced)
    "max_volume_usd": 70000,  # $50k max volume
    "min_volume_usd": 5000,      # Can set minimum to filter out completely dead markets
    
    # Spread filter - CRITICAL for tradeable markets
    # Spread = ask - bid. A 0.15 spread means 15% cost to enter+exit
    "max_spread": 0.20,  # 20% max spread (filter out illiquid garbage)
    
    # Price filters - look for markets not at extremes (based on best ask)
    "min_price": 0.05,  # Ignore < 5% (probably accurate)
    "max_price": 0.95,  # Ignore > 95% (probably accurate)
    
    # How many markets to analyze per scan (prevents runaway API costs)
    "max_markets_to_analyze": 50,
    
    # How often to check (in seconds) - for continuous mode
    "poll_interval_seconds": 300,  # 5 minutes
    
    # Output files
    "seen_markets_file": "seen_markets.json",
    "output_csv": "scan_results.csv",
    
    # LLM Configuration
    "use_llm": False,
    "llm_provider": "gemini",  # "anthropic", "openai", or "gemini"
    "llm_delay_seconds": 0,  # Delay between LLM calls (Gemini free tier: use 4-5s)
    "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
    "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
    "gemini_api_key": os.environ.get("GEMINI_API_KEY", ""),
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Market:
    """Represents a Polymarket market"""
    id: str
    question: str
    description: str
    category: str
    end_date: str
    volume: float
    liquidity: float
    # Bid/Ask for YES shares (what you'd actually pay)
    yes_bid: float  # Price to SELL yes shares
    yes_ask: float  # Price to BUY yes shares
    spread: float   # yes_ask - yes_bid (cost of round trip)
    created_at: str
    resolution_source: str
    url: str
    
    @property
    def no_bid(self) -> float:
        """NO bid = 1 - YES ask"""
        return 1 - self.yes_ask
    
    @property
    def no_ask(self) -> float:
        """NO ask = 1 - YES bid"""
        return 1 - self.yes_bid
    
    @property
    def age_hours(self) -> float:
        """Calculate market age in hours"""
        try:
            created = datetime.fromisoformat(self.created_at.replace('Z', '+00:00'))
            now = datetime.now(created.tzinfo)
            return (now - created).total_seconds() / 3600
        except:
            return 999  # If we can't parse, assume old
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "category": self.category,
            "volume": self.volume,
            "yes_bid": self.yes_bid,
            "yes_ask": self.yes_ask,
            "spread": self.spread,
            "age_hours": round(self.age_hours, 1),
            "url": self.url
        }


@dataclass  
class BetRating:
    """LLM's assessment of a betting opportunity"""
    score: int  # 1-10
    recommendation: str  # "YES", "NO", or "SKIP"
    confidence: str  # "low", "medium", "high"
    reasoning: str
    edge_estimate: Optional[float]  # Estimated edge vs market price


# ============================================================================
# POLYMARKET API CLIENT
# ============================================================================

class PolymarketClient:
    """
    Simple client to fetch market data from Polymarket's Gamma API.
    No authentication needed for read-only market data.
    """
    
    GAMMA_API_BASE = "https://gamma-api.polymarket.com"
    CLOB_API_BASE = "https://clob.polymarket.com"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PolymarketScanner/1.0"
        })
    
    def get_markets(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """
        Fetch markets from Gamma API.
        Returns raw market data as list of dicts.
        """
        try:
            # Gamma API endpoint for markets
            url = f"{self.GAMMA_API_BASE}/markets"
            params = {
                "limit": limit,
                "offset": offset,
                "active": "true",  # Only active markets
                "closed": "false"
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            print(f"[ERROR] Failed to fetch markets: {e}")
            return []
    
    def get_all_active_markets(self) -> list[dict]:
        """
        Fetch ALL active markets by paginating through the API.
        """
        all_markets = []
        offset = 0
        limit = 100
        
        while True:
            batch = self.get_markets(limit=limit, offset=offset)
            if not batch:
                break
            all_markets.extend(batch)
            
            if len(batch) < limit:
                break  # No more pages
            offset += limit
            time.sleep(0.5)  # Rate limiting
        
        return all_markets
    
    def parse_market(self, raw: dict) -> Optional[Market]:
        """
        Parse raw API response into Market dataclass.
        Handles missing fields gracefully.
        """
        try:
            # Get bid/ask/spread - these are what matter for actual trading
            best_bid = float(raw.get("bestBid", 0) or 0)
            best_ask = float(raw.get("bestAsk", 1) or 1)
            spread = float(raw.get("spread", 1) or 1)
            
            # If spread not provided, calculate it
            if spread == 1 and best_bid > 0 and best_ask < 1:
                spread = best_ask - best_bid
            
            # Build market object
            market_id = raw.get("id", raw.get("conditionId", ""))
            slug = raw.get("slug", market_id)
            
            return Market(
                id=market_id,
                question=raw.get("question", "Unknown"),
                description=raw.get("description", ""),
                category=raw.get("category", "unknown").lower() if raw.get("category") else "unknown",
                end_date=raw.get("endDate", ""),
                volume=float(raw.get("volume", 0) or 0),
                liquidity=float(raw.get("liquidity", 0) or 0),
                yes_bid=best_bid,
                yes_ask=best_ask,
                spread=spread,
                created_at=raw.get("createdAt", raw.get("startDate", "")),
                resolution_source=raw.get("resolutionSource", ""),
                url=f"https://polymarket.com/event/{slug}"
            )
        except Exception as e:
            print(f"[WARN] Failed to parse market: {e}")
            return None


# ============================================================================
# LLM INTEGRATION
# ============================================================================

class LLMAnalyzer:
    """
    Uses an LLM to analyze betting opportunities.
    Supports Anthropic (Claude) and OpenAI (GPT).
    """
    
    def __init__(self, provider: str = "anthropic"):
        self.provider = provider
        
    def analyze_market(self, market: Market) -> Optional[BetRating]:
        """
        Send market to LLM for analysis and get a bet rating.
        """
        prompt = self._build_prompt(market)
        
        if self.provider == "anthropic":
            return self._call_anthropic(prompt)
        elif self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "gemini":
            return self._call_gemini(prompt)
        else:
            print(f"[ERROR] Unknown LLM provider: {self.provider}")
            return None
    
    def _build_prompt(self, market: Market) -> str:
        """
        Construct the analysis prompt for the LLM.
        """
        return f"""You are analyzing a Polymarket prediction market for potential mispricing.

MARKET DETAILS:
- Question: {market.question}
- Description: {market.description[:500] if market.description else 'No description'}
- YES shares: Bid ${market.yes_bid:.2f} / Ask ${market.yes_ask:.2f} (to BUY yes, you pay the ask)
- NO shares: Bid ${market.no_bid:.2f} / Ask ${market.no_ask:.2f}
- Spread: {market.spread*100:.1f}% (cost to enter+exit)
- Implied YES probability (from ask): {market.yes_ask*100:.1f}%
- Category: {market.category}
- Volume: ${market.volume:,.0f}
- Market age: {market.age_hours:.1f} hours
- End date: {market.end_date}
- Resolution source: {market.resolution_source}

ANALYSIS TASK:
1. What is your estimate of the TRUE probability this resolves YES?
2. Is the market mispriced? Compare your probability to the YES ask price ({market.yes_ask*100:.1f}%)
3. What information edge might exist here?
4. What are the key risks?

OUTPUT FORMAT (JSON):
{{
    "true_probability": <your estimate 0-1>,
    "score": <1-10, where 10 is best opportunity>,
    "recommendation": "<YES|NO|SKIP>",
    "confidence": "<low|medium|high>",
    "reasoning": "<2-3 sentences explaining your analysis>",
    "edge_estimate": <estimated edge as decimal, e.g. 0.15 for 15% edge>
}}

Respond ONLY with the JSON object, no other text."""

    def _call_anthropic(self, prompt: str) -> Optional[BetRating]:
        """Call Anthropic's Claude API"""
        api_key = CONFIG.get("anthropic_api_key")
        if not api_key:
            print("[ERROR] ANTHROPIC_API_KEY not set")
            return None
        
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["content"][0]["text"]
            return self._parse_llm_response(content)
            
        except Exception as e:
            print(f"[ERROR] Anthropic API call failed: {e}")
            return None
    
    def _call_openai(self, prompt: str) -> Optional[BetRating]:
        """Call OpenAI's GPT API"""
        api_key = CONFIG.get("openai_api_key")
        if not api_key:
            print("[ERROR] OPENAI_API_KEY not set")
            return None
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1024,
                    "temperature": 0.3
                },
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return self._parse_llm_response(content)
            
        except Exception as e:
            print(f"[ERROR] OpenAI API call failed: {e}")
            return None
    
    def _call_gemini(self, prompt: str) -> Optional[BetRating]:
        """Call Google's Gemini API (free tier)"""
        api_key = CONFIG.get("gemini_api_key")
        if not api_key:
            print("[ERROR] GEMINI_API_KEY not set")
            return None
        
        try:
            # Gemini API endpoint with API key as query parameter
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={api_key}"
            
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [
                        {
                            "parts": [{"text": prompt}]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.3,
                        "maxOutputTokens": 1024
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            # Extract text from Gemini response format
            content = result["candidates"][0]["content"]["parts"][0]["text"]
            return self._parse_llm_response(content)
            
        except Exception as e:
            print(f"[ERROR] Gemini API call failed: {e}")
            return None
    
    def _parse_llm_response(self, content: str) -> Optional[BetRating]:
        """Parse LLM JSON response into BetRating"""
        try:
            # Clean up response (handle markdown code blocks)
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            data = json.loads(content)
            
            return BetRating(
                score=int(data.get("score", 5)),
                recommendation=data.get("recommendation", "SKIP").upper(),
                confidence=data.get("confidence", "low"),
                reasoning=data.get("reasoning", "No reasoning provided"),
                edge_estimate=data.get("edge_estimate")
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[WARN] Failed to parse LLM response: {e}")
            print(f"[DEBUG] Raw response: {content[:200]}")
            return None


# ============================================================================
# MARKET SCANNER
# ============================================================================

class MarketScanner:
    """
    Main scanner that finds new/mispriced markets.
    """
    
    def __init__(self):
        self.client = PolymarketClient()
        self.llm = LLMAnalyzer(provider=CONFIG["llm_provider"]) if CONFIG["use_llm"] else None
        self.seen_markets = self._load_seen_markets()
    
    def _load_seen_markets(self) -> set:
        """Load previously seen market IDs from disk"""
        try:
            with open(CONFIG["seen_markets_file"], "r") as f:
                return set(json.load(f))
        except FileNotFoundError:
            return set()
    
    def _save_seen_markets(self):
        """Save seen market IDs to disk"""
        with open(CONFIG["seen_markets_file"], "w") as f:
            json.dump(list(self.seen_markets), f)
    
    def _passes_filters(self, market: Market) -> bool:
        """Check if market passes all configured filters"""
        
        # Category filter
        if CONFIG["watch_categories"]:
            if market.category not in CONFIG["watch_categories"]:
                return False
        
        # Age filter
        if market.age_hours > CONFIG["max_market_age_hours"]:
            return False
        
        # Volume filter
        if market.volume > CONFIG["max_volume_usd"]:
            return False
        if market.volume < CONFIG["min_volume_usd"]:
            return False
        
        # Spread filter - CRITICAL for tradeable markets
        if market.spread > CONFIG["max_spread"]:
            return False
        
        # Price filter based on YES ask (what you'd pay to buy YES)
        if market.yes_ask < CONFIG["min_price"] or market.yes_ask > CONFIG["max_price"]:
            return False
        
        return True
    
    def _init_csv(self):
        """Initialize CSV file with headers if it doesn't exist"""
        import csv
        csv_file = CONFIG["output_csv"]
        try:
            with open(csv_file, 'x', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'question', 'yes_bid', 'yes_ask', 'spread',
                    'volume', 'age_hours', 'llm_score', 'llm_recommendation',
                    'llm_confidence', 'llm_edge', 'llm_reasoning', 'url'
                ])
        except FileExistsError:
            pass  # File already exists, that's fine
    
    def _write_csv_row(self, market: Market, rating: Optional[BetRating]):
        """Append a result row to CSV"""
        import csv
        csv_file = CONFIG["output_csv"]
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                market.question,
                f"{market.yes_bid:.3f}",
                f"{market.yes_ask:.3f}",
                f"{market.spread:.3f}",
                f"{market.volume:.0f}",
                f"{market.age_hours:.1f}",
                rating.score if rating else '',
                rating.recommendation if rating else '',
                rating.confidence if rating else '',
                f"{rating.edge_estimate:.3f}" if rating and rating.edge_estimate else '',
                rating.reasoning if rating else '',
                market.url
            ])
    
    def _display_single_result(self, market: Market, rating: Optional[BetRating], index: int):
        """Display a single market result immediately"""
        print(f"\n{'='*80}")
        print(f"[{index}] 📊 {market.question}")
        print(f"    YES: Bid ${market.yes_bid:.2f} / Ask ${market.yes_ask:.2f} | Spread: {market.spread*100:.1f}%")
        print(f"    NO:  Bid ${market.no_bid:.2f} / Ask ${market.no_ask:.2f}")
        print(f"    Volume: ${market.volume:,.0f} | Age: {market.age_hours:.1f}h")
        print(f"    🔗 {market.url}")
        
        if rating:
            emoji = "🟢" if rating.score >= 6 else "🟡" if rating.score >= 4 else "⚪"
            print(f"\n    {emoji} LLM Score: {rating.score}/10 | Rec: {rating.recommendation} | Conf: {rating.confidence}")
            if rating.edge_estimate:
                print(f"    📈 Estimated edge: {rating.edge_estimate*100:.1f}%")
            print(f"    💭 {rating.reasoning}")
    
    def scan(self) -> list[tuple[Market, Optional[BetRating]]]:
        """
        Scan for new interesting markets.
        Streams results as they're found and writes to CSV.
        """
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Scanning for markets...")
        
        # Initialize CSV
        self._init_csv()
        
        # Fetch all active markets
        raw_markets = self.client.get_all_active_markets()
        print(f"  Found {len(raw_markets)} total active markets")
        
        # Parse and filter
        results = []
        analyzed_count = 0
        filtered_count = 0
        max_to_analyze = CONFIG["max_markets_to_analyze"]
        llm_delay = CONFIG.get("llm_delay_seconds", 3)
        
        for raw in raw_markets:
            # Stop if we've hit the max
            if analyzed_count >= max_to_analyze:
                print(f"\n  Reached max markets limit ({max_to_analyze}). Stopping scan.")
                break
            
            market = self.client.parse_market(raw)
            if not market:
                continue
            
            # Skip already seen
            if market.id in self.seen_markets:
                continue
            
            # Apply filters
            if not self._passes_filters(market):
                filtered_count += 1
                continue
            
            analyzed_count += 1
            
            # Get LLM rating if enabled
            rating = None
            if self.llm:
                print(f"\n  [{analyzed_count}/{max_to_analyze}] Analyzing: {market.question[:50]}...")
                rating = self.llm.analyze_market(market)
                
                # Rate limiting delay for LLM API
                if analyzed_count < max_to_analyze:  # Don't delay after last one
                    time.sleep(llm_delay)
            
            # Display immediately
            self._display_single_result(market, rating, analyzed_count)
            
            # Write to CSV
            self._write_csv_row(market, rating)
            
            results.append((market, rating))
            
            # Mark as seen
            self.seen_markets.add(market.id)
        
        # Save seen markets
        self._save_seen_markets()
        
        print(f"\n{'='*80}")
        print(f"SCAN COMPLETE: Analyzed {analyzed_count} markets, filtered out {filtered_count}")
        print(f"Results saved to: {CONFIG['output_csv']}")
        
        return results
    
    def display_results(self, results: list[tuple[Market, Optional[BetRating]]]):
        """Summary display (results already shown inline during scan)"""
        if not results:
            print("  No new opportunities found.")
            return
        
        # Show quick summary sorted by score
        print("\n📋 SUMMARY (sorted by score):")
        sorted_results = sorted(results, key=lambda x: -(x[1].score if x[1] else 0))
        for market, rating in sorted_results[:10]:  # Top 10
            score = f"{rating.score}/10" if rating else "N/A"
            rec = rating.recommendation if rating else "?"
            print(f"  [{score}] [{rec}] {market.question[:60]}...")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point - run continuous scanning"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║         POLYMARKET NEW MARKET SCANNER                      ║
    ║   Finding mispriced markets before smart money arrives     ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    print(f"Configuration:")
    print(f"  - Categories: {CONFIG['watch_categories'] if CONFIG['watch_categories'] else 'ALL'}")
    print(f"  - Max market age: {CONFIG['max_market_age_hours']}h")
    print(f"  - Max volume: ${CONFIG['max_volume_usd']:,}")
    print(f"  - Max spread: {CONFIG['max_spread']*100:.0f}%")
    print(f"  - Max markets to analyze: {CONFIG['max_markets_to_analyze']}")
    print(f"  - LLM: {CONFIG['llm_provider'] if CONFIG['use_llm'] else 'Disabled'}")
    print(f"  - LLM delay: {CONFIG.get('llm_delay_seconds', 3)}s between calls")
    print(f"  - Output CSV: {CONFIG['output_csv']}")
    
    scanner = MarketScanner()
    
    while True:
        try:
            results = scanner.scan()
            scanner.display_results(results)
            
            print(f"\nNext scan in {CONFIG['poll_interval_seconds']} seconds... (Ctrl+C to exit)")
            time.sleep(CONFIG["poll_interval_seconds"])
            
        except KeyboardInterrupt:
            print("\n\nShutting down scanner...")
            break
        except Exception as e:
            print(f"\n[ERROR] Scan failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"Retrying in 60 seconds...")
            time.sleep(60)


def scan_once():
    """Run a single scan (useful for testing)"""
    scanner = MarketScanner()
    results = scanner.scan()
    scanner.display_results(results)
    return results


def list_categories():
    """Fetch markets and print unique categories"""
    print("Fetching markets to discover categories...")
    client = PolymarketClient()
    
    # Fetch active, non-closed markets
    try:
        url = f"{client.GAMMA_API_BASE}/markets"
        params = {
            "limit": 200,
            "active": "true",
            "closed": "false",
        }
        response = client.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        markets = response.json()
    except Exception as e:
        print(f"Error fetching markets: {e}")
        return
    
    from collections import Counter
    categories = Counter()
    
    # Debug: print first few markets' categories
    print("\nSample markets:")
    for i, raw in enumerate(markets[:5]):
        cat = raw.get("category", "NO_CATEGORY_FIELD")
        question = raw.get("question", "?")[:60]
        print(f"  {i+1}. [{cat}] {question}...")
    
    print("")
    
    for raw in markets:
        cat = raw.get("category")
        if cat:
            categories[cat.lower()] = categories.get(cat.lower(), 0) + 1
        else:
            categories["(no category)"] = categories.get("(no category)", 0) + 1
    
    print(f"📊 Found {len(markets)} markets across {len(categories)} categories:\n")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat:25} ({count} markets)")
    
    print("\n💡 Add categories to watch_categories in CONFIG at top of scanner.py")


if __name__ == "__main__":
    import sys
    
    if "--categories" in sys.argv:
        list_categories()
    elif "--once" in sys.argv:
        scan_once()
    else:
        main()