#!/usr/bin/env python3
"""
Quick test script to verify Polymarket API endpoints work.
Run this first before running the full backtest.
"""

import requests
import json
import time

DATA_API = "https://data-api.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"

def test_gamma_markets():
    """Test fetching markets from Gamma API"""
    print("\n1. Testing Gamma API - Get Markets...")
    
    resp = requests.get(f"{GAMMA_API}/markets", params={
        "closed": "true",
        "limit": 5,
    })
    
    if resp.status_code != 200:
        print(f"   ✗ FAILED: Status {resp.status_code}")
        return None
        
    data = resp.json()
    print(f"   ✓ SUCCESS: Got {len(data)} markets")
    
    # Find one with token IDs
    for m in data:
        token_ids = m.get("clobTokenIds", "")
        if token_ids:
            if isinstance(token_ids, str):
                token_ids = json.loads(token_ids)
            if len(token_ids) >= 2:
                print(f"   Sample market: {m.get('question', 'Unknown')[:50]}...")
                print(f"   Condition ID: {m.get('conditionId')}")
                print(f"   YES token: {token_ids[0][:30]}...")
                print(f"   NO token: {token_ids[1][:30]}...")
                return {
                    "yes_token": token_ids[0],
                    "no_token": token_ids[1],
                    "condition_id": m.get("conditionId"),
                }
    
    print("   ⚠ No markets with token IDs found")
    return None

def test_holders(condition_id: str):
    """Test fetching holders for a market (using conditionId)"""
    print(f"\n2. Testing Data API - Get Holders...")
    print(f"   Using condition_id: {condition_id[:20]}...")
    
    # NOTE: The /holders endpoint uses 'market' param (conditionId), not 'tokens'
    # It returns holders for BOTH outcomes (YES and NO) in one call
    resp = requests.get(f"{DATA_API}/holders", params={
        "market": condition_id,
        "limit": 5,
    })
    
    if resp.status_code != 200:
        print(f"   ✗ FAILED: Status {resp.status_code}")
        print(f"   Response: {resp.text[:200]}")
        return None
        
    data = resp.json()
    
    if not data:
        print("   ⚠ Empty response (market may be fully resolved/redeemed)")
        return None
    
    # Response is a list with one entry per token (YES and NO)
    # Each entry has: {token: "...", holders: [...]}
    print(f"   ✓ SUCCESS: Got {len(data)} outcome tokens")
    
    for i, token_data in enumerate(data):
        holders = token_data.get("holders", [])
        outcome = "YES" if i == 0 else "NO"
        print(f"   {outcome} side: {len(holders)} holders")
        
        if holders:
            h = holders[0]
            print(f"     Top holder: {h.get('proxyWallet', 'N/A')[:20]}...")
            print(f"     Amount: {h.get('amount', 'N/A')}")
            return h.get("proxyWallet")
    
    return None

def test_activity(wallet: str):
    """Test fetching activity for a wallet"""
    print(f"\n3. Testing Data API - Get Activity...")
    
    resp = requests.get(f"{DATA_API}/activity", params={
        "user": wallet,
        "type": "TRADE",
        "limit": 10,
    })
    
    if resp.status_code != 200:
        print(f"   ✗ FAILED: Status {resp.status_code}")
        print(f"   Response: {resp.text[:200]}")
        return False
        
    data = resp.json()
    print(f"   ✓ SUCCESS: Found {len(data)} trades")
    
    if data:
        t = data[0]
        print(f"   Sample trade:")
        print(f"     Timestamp: {t.get('timestamp')}")
        print(f"     Side: {t.get('side')}")
        print(f"     Size: {t.get('size')}")
        print(f"     Price: {t.get('price')}")
        print(f"     USDC: {t.get('usdcSize')}")
        
    return True

def test_positions(wallet: str):
    """Test fetching positions for a wallet"""
    print(f"\n4. Testing Data API - Get Positions...")
    
    resp = requests.get(f"{DATA_API}/positions", params={
        "user": wallet,
    })
    
    if resp.status_code != 200:
        print(f"   ✗ FAILED: Status {resp.status_code}")
        print(f"   Response: {resp.text[:200]}")
        return False
        
    data = resp.json()
    print(f"   ✓ SUCCESS: Found {len(data)} positions")
    
    if data:
        p = data[0]
        print(f"   Sample position:")
        print(f"     Market: {p.get('title', 'N/A')[:40]}...")
        print(f"     Size: {p.get('size')}")
        print(f"     Avg Price: {p.get('avgPrice')}")
        print(f"     Cash PnL: ${p.get('cashPnl', 0):,.2f}")
        print(f"     Percent PnL: {p.get('percentPnl', 0):.1f}%")
        
    return True

def test_active_markets():
    """Find an active market with current holders"""
    print("\n5. Testing with ACTIVE market (not resolved)...")
    
    resp = requests.get(f"{GAMMA_API}/markets", params={
        "active": "true",
        "closed": "false",
        "limit": 10,
        "order": "volume",
        "ascending": "false",
    })
    
    if resp.status_code != 200:
        print(f"   ✗ FAILED to get active markets")
        return None
        
    data = resp.json()
    
    for m in data:
        condition_id = m.get("conditionId")
        if not condition_id:
            continue
            
        print(f"   Active market: {m.get('question', 'Unknown')[:50]}...")
        
        # Try to get holders using conditionId
        time.sleep(0.1)
        resp2 = requests.get(f"{DATA_API}/holders", params={
            "market": condition_id,
            "limit": 5,
        })
        
        if resp2.status_code == 200:
            holder_data = resp2.json()
            # holder_data is a list with entries for each token (YES/NO)
            for token_entry in holder_data:
                holders = token_entry.get("holders", [])
                if holders:
                    print(f"   ✓ Found {len(holders)} holders")
                    return {
                        "condition_id": condition_id,
                        "wallet": holders[0].get("proxyWallet"),
                    }
    
    return None

def main():
    print("="*60)
    print("POLYMARKET API ENDPOINT TESTS")
    print("="*60)
    
    # Test 1: Get markets
    market_data = test_gamma_markets()
    
    # Test 2: Get holders (uses conditionId, not tokenId)
    wallet = None
    if market_data:
        wallet = test_holders(market_data["condition_id"])
    
    # If no holders on resolved market, try active market
    if not wallet:
        print("\n   Trying active market instead...")
        active = test_active_markets()
        if active:
            wallet = active.get("wallet")
    
    # Test 3 & 4: Activity and positions
    if wallet:
        time.sleep(0.1)
        test_activity(wallet)
        time.sleep(0.1)
        test_positions(wallet)
    else:
        print("\n⚠ Could not find a wallet to test activity/positions")
        print("  This may mean the API structure has changed")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    
    if wallet:
        print("\n✓ All endpoints appear to be working!")
        print("  You can run the full backtest with:")
        print("  python whale_tracker.py --markets 20")
    else:
        print("\n⚠ Some endpoints may not be working as expected.")
        print("  Check the API documentation for changes.")

if __name__ == "__main__":
    main()
