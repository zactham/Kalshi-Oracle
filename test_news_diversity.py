#!/usr/bin/env python3
"""
Test script to verify news source diversity
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_oracle_bot import safe_rss_parse, Config

def test_news_diversity():
    """Test that we're getting diverse news from multiple sources"""
    print("📰 Testing News Source Diversity")
    print("=" * 60)
    
    config = Config()
    
    # Test different news sources
    test_sources = {
        'Politics': [
            'http://feeds.reuters.com/reuters/politicsNews',
            'https://feeds.bbci.co.uk/news/politics/rss.xml',
            'https://rss.cnn.com/rss/edition.rss'
        ],
        'Economics': [
            'http://feeds.reuters.com/reuters/businessNews',
            'https://feeds.bbci.co.uk/news/business/rss.xml',
            'https://rss.cnn.com/rss/money_latest.rss'
        ],
        'Culture': [
            'http://www.espn.com/espn/rss/news',
            'https://feeds.bbci.co.uk/news/entertainment/rss.xml',
            'https://rss.cnn.com/rss/edition_entertainment.rss',
            'https://feeds.bbci.co.uk/news/technology/rss.xml'
        ],
        'Weather': [
            'https://www.weather.gov/rss',
            'https://feeds.bbci.co.uk/news/science_and_environment/rss.xml'
        ]
    }
    
    total_headlines = 0
    source_counts = {}
    
    for category, urls in test_sources.items():
        print(f"\n📂 Testing {category} sources:")
        category_headlines = 0
        
        for url in urls:
            print(f"  🔗 {url}")
            try:
                entries = safe_rss_parse(url, 2)  # Get 2 entries per source
                print(f"    ✅ Got {len(entries)} headlines")
                
                for entry in entries:
                    print(f"      📰 {entry['headline'][:60]}...")
                    category_headlines += len(entries)
                    total_headlines += len(entries)
                    
                    # Track source
                    source_name = url.split('//')[1].split('/')[0]
                    source_counts[source_name] = source_counts.get(source_name, 0) + len(entries)
                    
            except Exception as e:
                print(f"    ❌ Failed: {e}")
        
        print(f"  📊 Total {category} headlines: {category_headlines}")
    
    print(f"\n📈 Summary:")
    print(f"  Total headlines fetched: {total_headlines}")
    print(f"  Sources used: {len(source_counts)}")
    
    print(f"\n📊 Headlines per source:")
    for source, count in source_counts.items():
        print(f"  {source}: {count} headlines")
    
    print(f"\n✅ News diversity test completed!")
    print(f"   Expected: 5 headlines per category from multiple sources")
    print(f"   Actual: {total_headlines} total headlines from {len(source_counts)} sources")

if __name__ == "__main__":
    test_news_diversity()
