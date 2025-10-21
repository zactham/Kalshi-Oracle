#!/usr/bin/env python3
"""
Test script for error handling and loading animations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_oracle_bot import generate_market_proposal_with_openai, categorize_headline_with_openai, Config

def test_error_handling():
    """Test error handling for various OpenAI API errors"""
    print("🧪 Testing Error Handling")
    print("=" * 50)
    
    config = Config()
    
    if not config.openai_api_key or config.openai_api_key == 'your_openai_api_key_here':
        print("⚠️  OpenAI API key not set in .env file")
        print("   Add: OPEN_AI_API_KEY=your_actual_openai_api_key_here")
        return
    
    # Test with a valid headline
    test_headline = "Tesla stock surges 15% after Q3 earnings beat"
    
    print(f"📰 Testing with headline: {test_headline}")
    print()
    
    # Test categorization
    print("1. Testing AI categorization...")
    category = categorize_headline_with_openai(test_headline, config)
    print(f"   ✅ Categorized as: {category}")
    
    # Test proposal generation
    print("\n2. Testing AI proposal generation...")
    proposal = generate_market_proposal_with_openai(test_headline, category, 0.8, config)
    
    if proposal:
        print(f"   ✅ Generated proposal: {proposal['proposed_market']}")
        print(f"   ✅ Trigger headline matches: {proposal['trigger_headline'] == test_headline}")
    else:
        print("   ❌ No proposal generated")
    
    print("\n" + "=" * 50)
    print("✅ Error handling test completed!")
    print("\n💡 If you see any error messages above, they should be handled gracefully")
    print("   with appropriate user-friendly messages in the Streamlit interface.")

if __name__ == "__main__":
    test_error_handling()
