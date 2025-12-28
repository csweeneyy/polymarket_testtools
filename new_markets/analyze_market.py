"""
Example: Analyze a Single Market
================================
Quick script to analyze one specific market by URL or question text.
Useful for testing the LLM analysis on markets you've already found.
"""

import os
import sys
from scanner import PolymarketClient, LLMAnalyzer, Market, CONFIG

def find_market_by_query(query: str) -> list[Market]:
    """Search for markets matching a query string"""
    client = PolymarketClient()
    all_markets = client.get_all_active_markets()
    
    matches = []
    query_lower = query.lower()
    
    for raw in all_markets:
        market = client.parse_market(raw)
        if not market:
            continue
        
        if query_lower in market.question.lower():
            matches.append(market)
    
    return matches

def analyze_single_market(market: Market):
    """Run LLM analysis on a single market"""
    print(f"\n{'='*60}")
    print(f"MARKET: {market.question}")
    print(f"{'='*60}")
    print(f"Category: {market.category}")
    print(f"YES price: {market.yes_price*100:.1f}%")
    print(f"NO price: {market.no_price*100:.1f}%")
    print(f"Volume: ${market.volume:,.0f}")
    print(f"Age: {market.age_hours:.1f} hours")
    print(f"URL: {market.url}")
    print(f"\nResolution: {market.resolution_source}")
    
    if not CONFIG.get("anthropic_api_key") and not CONFIG.get("openai_api_key"):
        print("\n⚠️  No API key set. Set ANTHROPIC_API_KEY or OPENAI_API_KEY to enable LLM analysis.")
        return
    
    print(f"\n🤖 Running LLM analysis...")
    llm = LLMAnalyzer(provider=CONFIG["llm_provider"])
    rating = llm.analyze_market(market)
    
    if rating:
        print(f"\n{'─'*60}")
        print(f"LLM ANALYSIS")
        print(f"{'─'*60}")
        print(f"Score: {rating.score}/10")
        print(f"Recommendation: {rating.recommendation}")
        print(f"Confidence: {rating.confidence}")
        if rating.edge_estimate:
            print(f"Estimated edge: {rating.edge_estimate*100:.1f}%")
        print(f"\nReasoning: {rating.reasoning}")
    else:
        print("❌ LLM analysis failed")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_market.py <search query>")
        print("Example: python analyze_market.py 'xQc speedrun'")
        print("Example: python analyze_market.py 'Bitcoin 100k'")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    print(f"Searching for: '{query}'")
    
    matches = find_market_by_query(query)
    
    if not matches:
        print(f"No markets found matching '{query}'")
        sys.exit(1)
    
    if len(matches) == 1:
        analyze_single_market(matches[0])
    else:
        print(f"\nFound {len(matches)} matches:\n")
        for i, market in enumerate(matches[:10]):  # Show max 10
            print(f"  [{i+1}] {market.question[:70]}...")
            print(f"      YES: {market.yes_price*100:.0f}% | Vol: ${market.volume:,.0f}")
        
        print("\nEnter number to analyze (or 'all' for all):")
        choice = input("> ").strip()
        
        if choice.lower() == 'all':
            for market in matches[:10]:
                analyze_single_market(market)
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(matches):
                    analyze_single_market(matches[idx])
                else:
                    print("Invalid selection")
            except ValueError:
                print("Invalid input")

if __name__ == "__main__":
    main()
