#!/usr/bin/env python3
"""
Whale PnL-Weighted Position Strategy for Polymarket

Hypothesis: If traders with high lifetime PnL are holding YES, the market is more 
likely to resolve YES. If they're holding NO, it's more likely to resolve NO.

This script:
1. Fetches resolved markets
2. For each market, gets top holders on YES and NO sides
3. Computes each holder's lifetime PnL from their full trade history
4. Weights positions by trader quality (lifetime PnL)
5. Generates a signal and compares to actual resolution
6. Outputs correlation statistics to validate the strategy

API Endpoints used (all from data-api.polymarket.com):
- /holders: Top 20 holders per token
- /activity: Full trade history per wallet (with timestamps)
- /positions: Current positions with PnL (for sanity checking)

Rate limits: 200 requests / 10s = 20/sec → use 0.05s delay
"""

import requests
import time
import json
import csv
import os
import math # ensure matched
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from collections import defaultdict
import statistics


# =============================================================================
# CONFIGURATION
# =============================================================================

# Keyword Fallback for missing API tags
# Modern markets often lack the 'category' field. We use these keywords to identify them.
CATEGORY_KEYWORDS = {
    "sports": ["vs", "game", "win", "score", "nfl", "nba", "premier league", 
               "champions", "f1", "ufc", "tennis", "cup", "league", "points"],
    "crypto": ["bitcoin", "ethereum", "solana", "price", "token", "coin", 
               "airdrop", "fdv", "market cap", "btc", "eth", "sol", "nft"],
    "politics": ["trump", "biden", "election", "republican", "democrat", 
                 "senate", "house", "nominee", "poll", "approval", "cabinet", "congress"],
    "us-current-affairs": ["trump", "biden", "election", "republican", "democrat", 
                 "senate", "house", "nominee", "poll", "approval", "cabinet", "congress"], # Alias for Politics
    "pop-culture": ["taylor swift", "grammy", "oscar", "movie", "box office", "music", "song", "album"],
    "science": ["nasa", "space", "launch", "temperature", "climate", "covid", "virus"],
}

CONFIG = {
    # API endpoints
    "gamma_api": "https://gamma-api.polymarket.com",
    "data_api": "https://data-api.polymarket.com",
    "clob_api": "https://clob.polymarket.com",
    
    # Rate limiting (200 req/10s = 20/sec, we'll be conservative)
    "request_delay": 0.06,  # 60ms between requests (~16/sec)
    
    # How many resolved markets to analyze
    "max_markets": 200,
    
    # How many top holders per side to analyze
    "holders_per_side": 20,
    
    # Minimum volume for a market to be included (filters out tiny markets)
    "min_volume_usd": 10000,
    
    # Cache settings
    "cache_wallet_pnl": True,  # Cache lifetime PnL per wallet to avoid re-fetching
    
    # Output
    "output_csv": "backtest_results.csv",
    "verbose": True,
    
    # Time Range Filter
    "days_back": 90,  # Only analyze markets resolved in the last X days

    # Market Category Filter
    # Leave empty [] to analyze ALL markets.
    # Supported via Keyword Match: "Sports", "Crypto", "Politics", "Pop-Culture", "Science"
    "valid_categories": ["politics"],
}

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Trade:
    """A single trade from the /activity endpoint"""
    timestamp: int
    market_condition_id: str
    side: str  # "BUY" or "SELL"
    size: float  # Number of tokens
    price: float  # Price per token
    usdc_size: float  # Total USDC value
    outcome_index: int  # 0 = YES, 1 = NO
    outcome: str  # "Yes" or "No"
    asset_id: str = "" # Added for clarity, usually token ID
    
@dataclass
class WalletStats:
    """Aggregated stats for a wallet"""
    address: str
    total_trades: int = 0
    total_volume_usdc: float = 0.0
    realized_pnl: float = 0.0  # From resolved markets
    unrealized_pnl: float = 0.0  # From open positions
    win_count: int = 0
    loss_count: int = 0
    
    @property
    def win_rate(self) -> float:
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0.5
    
    @property
    def lifetime_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

@dataclass
class HolderPosition:
    """A holder's position in a specific market"""
    wallet: str
    side: str  # "YES" or "NO"
    amount: float  # Number of tokens held
    avg_entry_price: float = 0.0
    entry_timestamp: Optional[int] = None  # When they first bought
    
@dataclass 
class MarketResult:
    """Result of analyzing a single market"""
    condition_id: str
    question: str
    resolution: str  # "YES" or "NO"
    resolution_timestamp: int
    volume_usd: float
    
    # Computed signal
    yes_weighted_pnl: float = 0.0
    no_weighted_pnl: float = 0.0
    signal: float = 0.0  # -1 to 1, positive = YES, negative = NO
    
    # Holders analyzed
    yes_holders_count: int = 0
    no_holders_count: int = 0
    
    # Did signal predict correctly?
    @property
    def signal_correct(self) -> bool:
        if self.signal > 0:
            return self.resolution == "YES"
        elif self.signal < 0:
            return self.resolution == "NO"
        return False  # signal = 0 is neutral
    
    @property
    def predicted_side(self) -> str:
        if self.signal > 0:
            return "YES"
        elif self.signal < 0:
            return "NO"
        return "NEUTRAL"

# =============================================================================
# API CLIENT
# =============================================================================

class PolymarketClient:
    """Client for Polymarket APIs with rate limiting and caching"""
    
    def __init__(self, config: dict):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "WhaleTracker/1.0"
        })
        self.last_request_time = 0
        self.request_count = 0
        
        # Cache: wallet -> WalletStats
        self.wallet_cache: dict[str, WalletStats] = {}
        # Cache: wallet -> list[Trade]
        self.wallet_history_cache: dict[str, list[Trade]] = {}
        
    def _rate_limit(self):
        """Ensure we don't exceed rate limits"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.config["request_delay"]:
            time.sleep(self.config["request_delay"] - elapsed)
        self.last_request_time = time.time()
        self.request_count += 1
        
    def _get(self, url: str, params: dict = None) -> dict:
        """Make a rate-limited GET request"""
        self._rate_limit()
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"  [ERROR] Request failed: {e}")
            return {}
    
    # -------------------------------------------------------------------------
    # Gamma API - Market metadata
    # -------------------------------------------------------------------------
    
    def get_resolved_markets(self, limit: int = 100) -> list[dict]:
        """
        Fetch resolved markets from Gamma API with pagination until we find enough valid ones.
        """
        url = f"{self.config['gamma_api']}/markets"
        
        # Calculate cutoff timestamp
        days_back = self.config.get("days_back", 90)
        cutoff_ts = int(time.time()) - (days_back * 24 * 60 * 60)
        
        valid_cats = self.config.get("valid_categories", [])
        valid_cats_lower = [c.lower() for c in valid_cats]
        
        resolved = []
        offset = 0
        batch_size = 100  # Max per request usually
        max_attempts = 50 # Prevent infinite loops
        attempts = 0
        
        print(f"  Scanning for markets (Goal: {limit}, Category Filter: {valid_cats if valid_cats else 'ALL'})...")
        
        while len(resolved) < limit and attempts < max_attempts:
            params = {
                "closed": "true",
                "limit": batch_size,
                "offset": offset,
                "order": "volume",
                "ascending": "false",
            }
            
            data = self._get(url, params)
            attempts += 1
            offset += batch_size
            
            if not data:
                break
                
            for m in data:
                if len(resolved) >= limit:
                    break
                    
                # Must have a conditionId
                if not m.get("conditionId"):
                    continue
                    
                # Check Category (Smart Filter)
                if valid_cats:
                    # 1. Try explicit API category field
                    market_cat = m.get("category", "")
                    if not market_cat and m.get("events"):
                        market_cat = m["events"][0].get("category", "")
                    market_cat = market_cat.strip().lower()
                    
                    is_match = False
                    
                    # Direct Match Attempt
                    if market_cat and any(vc in market_cat for vc in valid_cats_lower):
                        is_match = True
                    
                    # Keyword Match Fallback (if no direct match)
                    if not is_match:
                        # Construct a searchable text block
                        title = m.get("question", "")
                        desc = m.get("description", "")
                        slug = m.get("slug", "")
                        full_text = f"{title} {desc} {slug}".lower()
                        
                        for vc in valid_cats_lower:
                            # Look up keywords for this category
                            keywords = CATEGORY_KEYWORDS.get(vc, [])
                            # If keywords exist, check them
                            if keywords:
                                if any(kw in full_text for kw in keywords):
                                    is_match = True
                                    break
                            # If no specific keywords defined, try matching the category name itself in text
                            elif vc in full_text:
                                is_match = True
                                break
                    
                    if not is_match:
                        continue
                
                # Check resolution date
                # Use closedTime (actual resolution) if available, else endDate (scheduled)
                # This prevents "Future Snapshots" if a 2026 market resolves in 2025.
                resolution_iso = m.get("closedTime") or m.get("endDate", "")
                if not resolution_iso:
                    continue
                    
                try:
                    res_ts = int(datetime.fromisoformat(resolution_iso.replace("Z", "+00:00")).timestamp())
                except:
                    continue
                    
                if res_ts < cutoff_ts:
                    continue
                    
                # Check volume (Move filter here to guarantee 'limit' usable markets)
                vol = float(m.get("volume", 0) or 0)
                if vol < self.config["min_volume_usd"]:
                    continue

                # Check validity (Start Date / Duration)
                # We need start date to compute duration for snapshots
                start_iso = m.get("startDate") or m.get("createdAt")
                if not start_iso:
                    continue
                    
                try:
                    start_ts = int(datetime.fromisoformat(start_iso.replace("Z", "+00:00")).timestamp())
                except:
                    continue
                    
                duration = res_ts - start_ts
                if duration < 3600: # Skip if < 1 hour
                    continue

                # Check if market has resolution data
                outcome_prices = m.get("outcomePrices", "")
                if not outcome_prices:
                    continue
                    
                # Parse outcome prices
                try:
                    prices = json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
                    if isinstance(prices, list) and len(prices) >= 2:
                        yes_price = float(prices[0])
                        no_price = float(prices[1])
                        
                        if yes_price == 1.0 and no_price == 0.0:
                            m["_resolution"] = "YES"
                            m["_resolution_ts"] = res_ts
                            resolved.append(m)
                        elif yes_price == 0.0 and no_price == 1.0:
                            m["_resolution"] = "NO"
                            m["_resolution_ts"] = res_ts
                            resolved.append(m)
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
            
            # Optimization: If we scanned 5000 markets and found nothing, maybe stop?
            print(f"    Scanned {offset} markets, found {len(resolved)} valid...", end="\r")
            
        print(f"\n  Done. Found {len(resolved)} matching markets.")
        return resolved
    
    # -------------------------------------------------------------------------
    # Data API - Holders
    # -------------------------------------------------------------------------
    
    def get_top_holders(self, condition_id: str, limit: int = 20) -> dict:
        """
        Get top holders for a market (both YES and NO sides).
        
        Uses conditionId (market parameter), returns holders for both outcomes.
        
        Returns dict with:
            {
                "yes_holders": [...],  # outcomeIndex 0
                "no_holders": [...]    # outcomeIndex 1
            }
        """
        url = f"{self.config['data_api']}/holders"
        params = {
            "market": condition_id,  # This is the conditionId
            "limit": limit,
        }
        data = self._get(url, params)
        
        if not data or not isinstance(data, list):
            return {"yes": [], "no": []} # Changed keys to 'yes'/'no' for consistency with new snippet
        
        result = {"yes": [], "no": []}
        
        # Response is a list with one entry per token
        # Each entry has: {token: "...", holders: [{proxyWallet, amount, outcomeIndex, ...}]}
        for token_data in data:
            holders = token_data.get("holders", [])
            for holder in holders:
                outcome_idx = holder.get("outcomeIndex", 0)
                # The new snippet expects 'address' key, not 'proxyWallet'
                holder_info = {"address": holder.get("proxyWallet"), "amount": holder.get("amount")}
                if outcome_idx == 0:
                    result["yes"].append(holder_info)
                else:
                    result["no"].append(holder_info)
        
        return result
    
    # -------------------------------------------------------------------------
    # Data API - User Activity (the key endpoint for Option B)
    # -------------------------------------------------------------------------
    
    def get_user_trade_history(self, wallet: str, limit: int = 2000, 
                           end_timestamp: Optional[int] = None) -> list[Trade]:
        """
        Get full trade history for a wallet.
        
        This is Option B - we get exact timestamps for every trade,
        allowing us to compute lifetime PnL accurately.
        
        Args:
            wallet: Proxy wallet address
            limit: Max trades to fetch (paginate if needed)
            end_timestamp: Only get trades before this time (for backtesting)
            
        Returns:
            List of Trade objects
        """
        cache_key = f"{wallet}_{end_timestamp or 'current'}"
        if self.config["cache_wallet_pnl"] and cache_key in self.wallet_history_cache:
            return self.wallet_history_cache[cache_key]

        url = f"{self.config['data_api']}/activity"
        params = {
            "user": wallet,
            "type": "TRADE",  # Only trades, not splits/merges/redeems
            "limit": limit,
            "sortBy": "TIMESTAMP",
            "sortDirection": "DESC",
        }
        
        if end_timestamp:
            params["end"] = end_timestamp
            
        data = self._get(url, params)
        
        if not data or not isinstance(data, list):
            return []
            
        trades = []
        for t in data:
            try:
                trade = Trade(
                    timestamp=t.get("timestamp", 0),
                    market_condition_id=t.get("conditionId", ""),
                    side=t.get("side", ""),
                    size=float(t.get("size", 0)),
                    price=float(t.get("price", 0)),
                    usdc_size=float(t.get("usdcSize", 0)),
                    outcome_index=t.get("outcomeIndex", 0),
                    outcome=t.get("outcome", ""),
                    asset_id=t.get("assetId", "") # Added asset_id
                )
                trades.append(trade)
            except (ValueError, TypeError):
                continue
        
        self.wallet_history_cache[cache_key] = trades
        return trades
    
    # -------------------------------------------------------------------------
    # Data API - User Positions (for current PnL snapshot)
    # -------------------------------------------------------------------------
    
    def get_wallet_positions(self, wallet: str) -> list[dict]:
        """
        Get current positions for a wallet with PnL data.
        
        Returns positions with: avgPrice, cashPnl, percentPnl, realizedPnl, etc.
        """
        url = f"{self.config['data_api']}/positions"
        params = {
            "user": wallet,
        }
        data = self._get(url, params)
        
        if not data or not isinstance(data, list):
            return []
            
        return data
    
    def get_historical_position(self, wallet: str, condition_id: str, timestamp: int) -> float:
        """
        Reconstruct a user's position in a market at a specific time in the past.
        
        Args:
            wallet: User address
            condition_id: Market condition ID
            timestamp: Cutoff timestamp (Unix seconds)
            
        Returns:
            Net number of tokens held (positive)
        """
        # Fetch trades up to the snapshot time
        trades = self.get_user_trade_history(wallet, limit=1000, end_timestamp=timestamp)
        
        net_position = 0.0
        for t in trades:
            if t.market_condition_id != condition_id:
                continue
                
            if t.side == "BUY":
                net_position += t.size
            elif t.side == "SELL":
                net_position -= t.size
                
        # Return 0 if negative (shouldn't happen with valid data) or very small
        return max(0.0, net_position)
    
    # -------------------------------------------------------------------------
    # Compute Lifetime PnL for a wallet
    # -------------------------------------------------------------------------
    
    def compute_wallet_lifetime_pnl(self, wallet: str, 
                                    as_of_timestamp: Optional[int] = None,
                                    known_resolutions: dict = None,
                                    pre_fetched_history: Optional[list[Trade]] = None) -> WalletStats:
        """
        Compute a wallet's lifetime PnL from their full trade history.
        
        Args:
            wallet: Proxy wallet address
            as_of_timestamp: Only consider trades before this time
            known_resolutions: Dict of {condition_id: "YES" or "NO"} for known resolved markets
            pre_fetched_history: If provided, use this list of trades instead of fetching
            
        Returns:
            WalletStats with lifetime PnL (Realized + Synthetic)
        """
        known_resolutions = known_resolutions or {}
        
        # Check cache (include generic hash of resolutions in key if possible, 
        # but for now we assume resolutions are static for the run)
        cache_key = f"{wallet}_{as_of_timestamp or 'current'}"
        if self.config["cache_wallet_pnl"] and cache_key in self.wallet_cache:
            return self.wallet_cache[cache_key]
        
        # Fetch all trades if not pre-fetched
        trades = pre_fetched_history if pre_fetched_history is not None else \
                 self.get_user_trade_history(wallet, limit=2000, end_timestamp=as_of_timestamp)
        
        if not trades:
            stats = WalletStats(address=wallet)
            self.wallet_cache[cache_key] = stats
            return stats
        
        # Group trades by (market, outcome)
        positions: dict[tuple, list[Trade]] = defaultdict(list)
        for trade in trades:
            key = (trade.market_condition_id, trade.outcome_index)
            positions[key].append(trade)
        
        stats = WalletStats(address=wallet)
        stats.total_trades = len(trades)
        stats.total_volume_usdc = sum(t.usdc_size for t in trades)
        
        # For each position, compute realized P&L from trading
        for (condition_id, outcome_idx), position_trades in positions.items():
            net_tokens = 0.0
            cost_basis = 0.0
            
            # Sort by time
            sorted_trades = sorted(position_trades, key=lambda x: x.timestamp)
            
            for t in sorted_trades:
                if t.side == "BUY":
                    net_tokens += t.size
                    cost_basis += t.usdc_size
                elif t.side == "SELL":
                    # Proportionally reduce cost basis
                    if net_tokens > 0:
                        sell_ratio = min(t.size / net_tokens, 1.0)
                        realized = t.usdc_size - (cost_basis * sell_ratio)
                        stats.realized_pnl += realized
                        cost_basis *= (1 - sell_ratio)
                        net_tokens -= t.size
                        
                        if realized > 0:
                            stats.win_count += 1
                        else:
                            stats.loss_count += 1
            
            # Synthetic Payouts (Hold to Expiry)
            # If they still hold tokens and we know the resolution:
            # - If they held Winner: Payout = net_tokens * $1.00
            # - If they held Loser: Payout = 0 (Loss is essentially valid, cost basis is gone)
            if net_tokens > 0.1 and condition_id in known_resolutions:
                winner = known_resolutions[condition_id]  # "YES" or "NO"
                
                # Determine if this position is the winner
                # We assume outcome_index 0=YES, 1=NO (standard for binary)
                is_winner = False
                if outcome_idx == 0 and winner == "YES":
                    is_winner = True
                elif outcome_idx == 1 and winner == "NO":
                    is_winner = True
                
                if is_winner:
                    # They won! Treat as if they sold at $1.00
                    payout_value = net_tokens * 1.0
                    synthetic_profit = payout_value - cost_basis
                    stats.realized_pnl += synthetic_profit
                    stats.win_count += 1
                else:
                    # They lost. The cost_basis is sunk.
                    # Their realized PnL already reflects strictly trading PnL.
                    # But we must subtract the remaining cost basis because it's a total loss.
                    # (realized_pnl accumulator only sees 'SELL' events, so we calculate loss manually)
                    stats.realized_pnl -= cost_basis
                    stats.loss_count += 1
        
        # Update cache
        self.wallet_cache[cache_key] = stats
        return stats

# =============================================================================
# STRATEGY LOGIC
# =============================================================================

class WhaleStrategy:
    """
    The core strategy: weight positions by trader quality (lifetime PnL).
    
    For each market:
    1. Get top holders on YES and NO sides
    2. For each holder, compute their lifetime PnL
    3. Compute: weighted_yes = sum(position_size * lifetime_pnl) for YES holders
                weighted_no = sum(position_size * lifetime_pnl) for NO holders
    4. Signal = (weighted_yes - weighted_no) / (|weighted_yes| + |weighted_no|)
    
    Signal interpretation:
    - signal > 0: Smart money favors YES
    - signal < 0: Smart money favors NO
    - signal magnitude: strength of conviction
    """
    
    def __init__(self, client: PolymarketClient, config: dict):
        self.client = client
        self.config = config
        self.results: list[dict] = [] # Changed to list of dicts
        
    def analyze_market(self, market: dict, known_resolutions: dict = None) -> Optional[dict]:
        """
        Analyzes a single market at multiple time snapshots (50%, 75%, 90% duration).
        Returns a dict with signals for each timeframe.
        """
        condition_id = market.get("conditionId")
        question = market.get("question")
        resolution = market.get("_resolution") # "YES" or "NO"
        res_ts = market.get("_resolution_ts")
        
        # Get start/end time
        # Start time is often messy in API (sometimes creation date, sometimes start date)
        # We'll try "startDate" then "createdAt"
        start_iso = market.get("startDate") or market.get("createdAt")
        if not start_iso:
            return None
            
        try:
            start_ts = int(datetime.fromisoformat(start_iso.replace("Z", "+00:00")).timestamp())
        except:
            return None
            
        duration = res_ts - start_ts
        if duration < 3600: # Skip if < 1 hour (often test/weird markets)
            return None
            
        # Define Snapshots: 50%, 75% (25% remaining), 90% (10% remaining)
        # Plus 24h Pre-Resolution (User Request)
        snapshot_ratios = {
            "50pct": 0.50,
            "25pctRem": 0.75,
            "10pctRem": 0.90
        }
        
        snapshots = {}
        for name, ratio in snapshot_ratios.items():
            snapshots[name] = int(start_ts + (duration * ratio))
            
        # Add 24h Pre-Resolution (if valid)
        if duration > 86400:
             snapshots["24hPre"] = res_ts - 86400

        # 1. Fetch Top Holders (at final resolution time to cast wide net, or midpoint? 
        # Ideally we want holders valid at snapshot. 
        # Gamma API doesn't allow "holders at historical ts". 
        # We must fetch current holders (final) and verify if they held at snapshot.
        # This introduces some survivorship bias but is best we can do without massive data archival.
        holders = self.client.get_top_holders(condition_id, limit=self.config["holders_per_side"])
        all_holders = holders.get("yes", []) + holders.get("no", [])
        
        unique_wallets = list(set(h["address"] for h in all_holders if h["address"]))
        
        # Pre-fetch histories for all wallets ONCE
        # We store them to avoid re-fetching for each snapshot
        wallet_histories = {}
        # print(f"    Fetching history for {len(unique_wallets)} wallets...")
        
        for wallet in unique_wallets:
            # We fetch history up to resolution time
            history = self.client.get_user_trade_history(wallet, end_timestamp=res_ts)
            wallet_histories[wallet] = history

        # 2. Analyze at each Snapshot
        results = {
            "Question": question,
            "Resolution": resolution,
            "ConditionId": condition_id,
        }
        
        # Helpful Volume Check (Current Total Volume)
        vol = float(market.get("volume", 0) or 0)
        if vol < self.config["min_volume_usd"]:
            return None

        if self.config.get("verbose"):
            print(f"\n  Analyzing: {question[:60]}...")
            print(f"    Resolution: {resolution}, Volume: ${vol:,.0f}")
        
        # Sort snapshots for clean printing
        sorted_snapshot_names = ["50pct", "25pctRem", "10pctRem", "24hPre"]
        # Filter to those actually present
        sorted_snapshot_names = [n for n in sorted_snapshot_names if n in snapshots]
        
        for name in sorted_snapshot_names:
            ts = snapshots[name]
            
            # Calculate metrics for this snapshot
            weighted_yes_pnl = 0.0
            weighted_no_pnl = 0.0
            
            for wallet, history in wallet_histories.items():
                # Filter history for this specific timestamp
                history_at_ts = [t for t in history if t.timestamp <= ts]
                if not history_at_ts:
                    continue
                    
                # Compute PnL stats at this timestamp
                stats = self.client.compute_wallet_lifetime_pnl(
                    wallet, 
                    as_of_timestamp=ts, 
                    known_resolutions=known_resolutions,
                    pre_fetched_history=history_at_ts 
                )
                
                # Calculate Net Position dynamically at this timestamp
                # We simply sum up the tokens held for Outcome 0 (YES) and Outcome 1 (NO)
                held_yes = 0.0
                held_no = 0.0
                
                for t in history_at_ts:
                    if t.market_condition_id == condition_id:
                        change = t.size if t.side == "BUY" else -t.size
                        
                        if t.outcome_index == 0: # YES
                            held_yes += change
                        elif t.outcome_index == 1: # NO
                            held_no += change
                
                # Determine dominant side at this snapshot
                # We only count them if they have a non-trivial position
                if held_yes > 5.0 and held_yes > held_no:
                    direction = 1 # Held YES
                elif held_no > 5.0 and held_no > held_yes:
                    direction = -1 # Held NO
                else:
                    direction = 0 # Neutral / Dust / Equal hedging
                
                if direction == 0:
                    continue
                    
                # Calculate Log-Weighted PnL Contribution
                if stats.realized_pnl == 0:
                    continue
                    
                weight = math.copysign(math.log10(abs(stats.realized_pnl) + 1), stats.realized_pnl)
                            
                if direction == 1:
                    weighted_yes_pnl += weight
                elif direction == -1:
                    weighted_no_pnl += weight

            # Calculate Signal for this snapshot
            # Normalize: (Yes - No) / (abs(Yes) + abs(No)) -> -1 to 1.
            
            denom = abs(weighted_yes_pnl) + abs(weighted_no_pnl)
            if denom == 0:
                raw_signal = 0.0
            else:
                raw_signal = (weighted_yes_pnl - weighted_no_pnl) / denom
                
            results[f"Signal_{name}"] = round(raw_signal, 3)
            results[f"YesPnl_{name}"] = round(weighted_yes_pnl, 1)
            results[f"NoPnl_{name}"] = round(weighted_no_pnl, 1)
            
            # Determine prediction info based on signal threshold
            pred = "NEUTRAL"
            if raw_signal > 0.1: pred = "YES"
            if raw_signal < -0.1: pred = "NO"
            
            is_correct = False
            if pred != "NEUTRAL":
                is_correct = (pred == resolution)
                
            results[f"Pred_{name}"] = pred
            results[f"Correct_{name}"] = is_correct
            
            if self.config.get("verbose"):
                outcome_str = "✓ CORRECT" if is_correct else "✗ WRONG"
                if pred == "NEUTRAL": outcome_str = "-"
                
                print(f"    Snapshot {name:<10}: YesPnl={weighted_yes_pnl:6.1f}, NoPnl={weighted_no_pnl:6.1f}, Signal={raw_signal:6.3f} -> Predicted {pred:<4} (Actual {resolution}) {outcome_str}")

        return results
    
    def run_backtest(self, num_markets: int = 100) -> list[dict]: # Changed return type
        """
        Run backtest on N resolved markets.
        
        Fetches resolved markets, analyzes each, and records results.
        """
        print(f"\n{'='*60}")
        print("WHALE PnL STRATEGY BACKTEST")
        print(f"{'='*60}")
        print(f"Analyzing up to {num_markets} resolved markets...")
        print(f"Min volume: ${self.config['min_volume_usd']:,}")
        print(f"Holders per side: {self.config['holders_per_side']}")
        
        # Fetch resolved markets
        print("\nFetching resolved markets from Gamma API...")
        markets = self.client.get_resolved_markets(limit=num_markets)
        
        print(f"Found {len(markets)} matching markets")
        
        # Limit (should be handled by fetch, but double check)
        markets = markets[:num_markets]
        
        # Build map of known resolutions for synthetic PnL
        known_resolutions = {m["conditionId"]: m["_resolution"] for m in markets}
        
        # Analyze each market
        results = []
        for i, market in enumerate(markets):
            print(f"\n[{i+1}/{len(markets)}]", end="")
            
            try:
                result = self.analyze_market(market, known_resolutions=known_resolutions)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"  [ERROR] Failed to analyze market {market.get('conditionId', '')}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        self.results = results
        return results
    
    def compute_statistics(self) -> dict:
        """
        Compute aggregate statistics from backtest results for ALL snapshot types.
        
        Returns dict keyed by snapshot name (e.g. "10pctRem"), containing stats.
        """
        if not self.results:
            return {}
            
        # Discover available signal keys
        all_keys = self.results[0].keys()
        signal_keys = [k for k in all_keys if k.startswith("Signal_")]
        
        # Sort them for display consistency
        # We want: 50pct, 25pctRem, 10pctRem, 24hPre
        ordered_keys = []
        for exact in ["Signal_50pct", "Signal_25pctRem", "Signal_10pctRem", "Signal_24hPre"]:
            if exact in signal_keys:
                ordered_keys.append(exact)
        
        # Add any others found
        for k in signal_keys:
            if k not in ordered_keys:
                ordered_keys.append(k)

        # Helper to determine if a signal is correct
        def is_signal_correct(signal_value, resolution):
            if signal_value > 0:
                return resolution == "YES"
            elif signal_value < 0:
                return resolution == "NO"
            return False # Neutral signal is not "correct"
            
        all_stats = {}
        
        for key in ordered_keys:
            # Filter matches that have this key
            filtered_results = [r for r in self.results if key in r]
            if not filtered_results:
                continue
                
            snap_name = key.replace("Signal_", "")
            
            stats = {
                "total_markets": len(filtered_results),
                "correct_predictions": sum(1 for r in filtered_results if is_signal_correct(r[key], r["Resolution"])),
                "accuracy": 0.0,
                "yes_resolved": sum(1 for r in filtered_results if r["Resolution"] == "YES"),
                "no_resolved": sum(1 for r in filtered_results if r["Resolution"] == "NO"),
                "avg_signal": statistics.mean(r[key] for r in filtered_results),
                "signal_std": statistics.stdev(r[key] for r in filtered_results) if len(filtered_results) > 1 else 0,
            }
            
            stats["accuracy"] = stats["correct_predictions"] / stats["total_markets"]
            
            # Correlation
            outcomes = [1 if r["Resolution"] == "YES" else 0 for r in filtered_results]
            signals = [r[key] for r in filtered_results]
            
            if len(set(outcomes)) > 1:
                mean_outcome = statistics.mean(outcomes)
                mean_signal = statistics.mean(signals)
                
                numerator = sum((s - mean_signal) * (o - mean_outcome) for s, o in zip(signals, outcomes))
                denom_signal = sum((s - mean_signal) ** 2 for s in signals) ** 0.5
                denom_outcome = sum((o - mean_outcome) ** 2 for o in outcomes) ** 0.5
                
                if denom_signal > 0 and denom_outcome > 0:
                    stats["correlation"] = numerator / (denom_signal * denom_outcome)
                else:
                    stats["correlation"] = 0.0
            else:
                stats["correlation"] = 0.0
                
            all_stats[snap_name] = stats
            
        return all_stats
    
    def print_results(self):
        """Print formatted results summary for ALL snapshots"""
        all_stats = self.compute_statistics()
        
        if not all_stats:
            print("\nNo results to analyze.")
            return
        
        print(f"\n{'='*60}")
        print("BACKTEST RESULTS SUMMARY")
        print(f"{'='*60}")
        
        # We loop through stats in order
        for snap_name, stats in all_stats.items():
            print(f"\n--- SNAPSHOT: {snap_name} ---")
            print(f"Markets: {stats['total_markets']}")
            print(f"Accuracy: {stats['accuracy']:.1%} ({stats['correct_predictions']}/{stats['total_markets']})")
            print(f"Correlation: {stats['correlation']:.3f}")
            print(f"Avg Signal: {stats['avg_signal']:.3f}")
        
        # Simple Interpretation based on best performing snapshot
        best_snap = max(all_stats.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n--- BEST PERFORMING SNAPSHOT: {best_snap[0]} ---")
        acc = best_snap[1]["accuracy"]
        corr = best_snap[1]["correlation"]
        
        if acc > 0.55 and corr > 0.15:
            print("✓ PROMISING: Signal shows positive correlation with outcomes.")
        elif acc > 0.52 and corr > 0.05:
            print("⚠ MARGINAL: Weak edge detected.")
        else:
            print("✗ NOT PROFITABLE: Signal does not predict outcomes better than chance.")
            
    def save_results(self, filepath: str = None):
        """Save results to CSV for further analysis, auto-incrementing filename"""
        import os
        import csv
        
        if not self.results:
            return

        base_path = filepath or self.config["output_csv"]
        
        # Auto-increment if file exists
        final_path = base_path
        counter = 1
        name, ext = os.path.splitext(base_path)
        
        while os.path.exists(final_path):
            final_path = f"{name}_{counter}{ext}"
            counter += 1
        
        # Determine keys from the first result
        # We want to enforce a nice column order if possible
        all_keys = list(self.results[0].keys())
        preferred_order = ["Question", "Resolution", "ConditionId", "Volume"]
        
        # Group snapshot keys: 50pct, 25pctRem, 10pctRem, 24hPre
        snap_names = ["50pct", "25pctRem", "10pctRem", "24hPre"]
        
        fieldnames = []
        # 1. Base columns
        for k in preferred_order:
            if k in all_keys:
                fieldnames.append(k)
        
        # 2. Snapshot groups
        for snap in snap_names:
            # We want: Signal, YesPnl, NoPnl, Pred, Correct
            group_keys = [f"Signal_{snap}", f"YesPnl_{snap}", f"NoPnl_{snap}", f"Pred_{snap}", f"Correct_{snap}"]
            for k in group_keys:
                if k in all_keys:
                    fieldnames.append(k)
                    
        # 3. Any leftovers
        for k in all_keys:
            if k not in fieldnames:
                fieldnames.append(k)
                
        with open(final_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
                
        print(f"\nResults saved to: {final_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    # import math (Moved to top)
    
    parser = argparse.ArgumentParser(description="Whale PnL Strategy Backtester")
    parser.add_argument("--markets", type=int, default=50, help="Number of markets to analyze")
    parser.add_argument("--days", type=int, default=90, help="Days back to look for resolved markets")
    parser.add_argument("--min-volume", type=int, default=10000, help="Minimum volume filter")
    parser.add_argument("--holders", type=int, default=20, help="Holders per side to analyze")
    parser.add_argument("--category", type=str, default=None, help="Filter by category (e.g. Sports, Crypto)")
    parser.add_argument("--output", type=str, default="backtest_results.csv", help="Output CSV file")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    # Update config
    CONFIG["max_markets"] = args.markets
    CONFIG["days_back"] = args.days
    CONFIG["min_volume_usd"] = args.min_volume
    if args.category:
        CONFIG["valid_categories"] = [args.category]
    CONFIG["holders_per_side"] = args.holders
    CONFIG["output_csv"] = args.output
    CONFIG["verbose"] = not args.quiet
    
    # Create client and strategy
    client = PolymarketClient(CONFIG)
    strategy = WhaleStrategy(client, CONFIG)
    
    # Run backtest
    strategy.run_backtest(num_markets=args.markets)
    
    # Print results
    strategy.print_results()
    
    # Save to CSV
    strategy.save_results()
    
    print(f"\nTotal API requests made: {client.request_count}")

if __name__ == "__main__":
    main()
