#!/usr/bin/env python3
"""
Test script for Kalshi API integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_oracle_bot import kalshi_get_markets, Config

def test_kalshi_api():
    """Test the Kalshi API integration"""
    print("🔗 Testing Kalshi API Integration")
    print("=" * 50)
    
    config = Config()
    
    if not config.kalshi_api_key or config.kalshi_api_key == 'your_api_key_here':
        print("⚠️  Kalshi API key not set in .env file")
        print("   Add: KALSHI_API_KEY=your_actual_api_key_here")
        return
    
    print(f"🔑 Using API key: {config.kalshi_api_key[:10]}...")
    print(f"🌐 API URL: {config.api_url}")
    
    # Test API call
    print("\n📡 Testing API call...")
    response = kalshi_get_markets(
        config.kalshi_api_key,
        config.api_url,
        categories=['Politics', 'Weather', 'Culture', 'Economics']
    )
    
    if response:
        markets = response.get('markets', [])
        print(f"✅ Success! Found {len(markets)} markets")
        
        if markets:
            print("\n📊 Sample markets:")
            for i, market in enumerate(markets[:3]):
                print(f"  {i+1}. {market.get('title', 'N/A')} ({market.get('category', 'N/A')})")
    else:
        print("❌ API call failed")
    
    print("\n" + "=" * 50)
    print("✅ Kalshi API test completed!")

if __name__ == "__main__":
    test_kalshi_api()
