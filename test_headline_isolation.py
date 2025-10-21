#!/usr/bin/env python3
"""
Test script to verify headline isolation and prevent mixing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_oracle_bot import generate_market_proposal_with_openai, Config

def test_headline_isolation():
    """Test that headlines don't get mixed up"""
    print("🔍 Testing Headline Isolation")
    print("=" * 50)
    
    config = Config()
    
    if not config.openai_api_key or config.openai_api_key == 'your_openai_api_key_here':
        print("⚠️  OpenAI API key not set in .env file")
        return
    
    # Test with two very different headlines
    test_headlines = [
        {
            'headline': 'Tesla stock surges 15% after Q3 earnings beat',
            'category': 'Economics',
            'sentiment': 0.8
        },
        {
            'headline': 'Hurricane Maria approaches Florida with 120mph winds',
            'category': 'Weather',
            'sentiment': 0.9
        }
    ]
    
    print("Testing headline isolation...")
    print()
    
    for i, test in enumerate(test_headlines, 1):
        print(f"📰 Test {i}: {test['headline']}")
        print(f"   Category: {test['category']}, Sentiment: {test['sentiment']}")
        
        proposal = generate_market_proposal_with_openai(
            test['headline'],
            test['category'],
            test['sentiment'],
            config
        )
        
        if proposal:
            print(f"✅ Generated: {proposal['proposed_market']}")
            print(f"   Trigger: {proposal['trigger_headline']}")
            
            # Check if the trigger headline matches exactly
            if proposal['trigger_headline'] == test['headline']:
                print("✅ Headline isolation: PASSED")
            else:
                print("❌ Headline isolation: FAILED")
                print(f"   Expected: {test['headline']}")
                print(f"   Got: {proposal['trigger_headline']}")
        else:
            print("❌ No proposal generated")
        
        print("-" * 50)
    
    print("✅ Headline isolation test completed!")

if __name__ == "__main__":
    test_headline_isolation()
