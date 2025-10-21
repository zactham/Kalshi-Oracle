#!/usr/bin/env python3
"""
Test script for dynamic AI-generated market proposals
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_oracle_bot import generate_market_proposal_with_openai, Config

def test_dynamic_proposals():
    """Test the dynamic proposal generation"""
    print("🤖 Testing Dynamic AI-Generated Market Proposals")
    print("=" * 60)
    
    config = Config()
    
    if not config.openai_api_key or config.openai_api_key == 'your_openai_api_key_here':
        print("⚠️  OpenAI API key not set in .env file")
        print("   Add: OPEN_AI_API_KEY=your_actual_openai_api_key_here")
        return
    
    # Test headlines
    test_cases = [
        {
            'headline': 'Tesla stock surges 15% after Q3 earnings beat expectations',
            'category': 'Economics',
            'sentiment': 0.8
        },
        {
            'headline': 'Hurricane Maria approaches Florida coast with 120mph winds',
            'category': 'Weather',
            'sentiment': 0.9
        },
        {
            'headline': 'Supreme Court to hear major abortion rights case next month',
            'category': 'Politics',
            'sentiment': 0.7
        },
        {
            'headline': 'Taylor Swift announces world tour with 50+ dates',
            'category': 'Culture',
            'sentiment': 0.6
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📰 Test Case {i}: {test_case['headline']}")
        print(f"   Category: {test_case['category']}, Sentiment: {test_case['sentiment']}")
        print("-" * 50)
        
        proposal = generate_market_proposal_with_openai(
            test_case['headline'],
            test_case['category'],
            test_case['sentiment'],
            config
        )
        
        if proposal:
            print(f"✅ Generated Proposal: {proposal['proposed_market']}")
            print(f"   Structure: {proposal['structure']}")
            print(f"   Audience: {proposal['audience']}")
            print(f"   Business Case: {proposal['business_case']}")
        else:
            print("❌ No proposal generated")
    
    print("\n" + "=" * 60)
    print("✅ Dynamic proposal generation test completed!")

if __name__ == "__main__":
    test_dynamic_proposals()
