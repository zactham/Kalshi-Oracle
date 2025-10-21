import os
import feedparser
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
import requests
import re
import streamlit as st
import logging
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
import openai
from streamlit import spinner
import pickle
import hashlib
from difflib import SequenceMatcher

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class for the Oracle Bot"""
    # API Configuration
    kalshi_api_key: str = os.getenv('KALSHI_API_KEY')
    openai_api_key: str = os.getenv('OPEN_AI_API_KEY')
    api_url: str = 'https://api.elections.kalshi.com/trade-api/v2'
    
    # AI Thresholds - Lower thresholds to capture more dramatic news (including negative)
    politics_sentiment_threshold: float = 0.2  # Lower threshold for more dramatic politics
    culture_sentiment_threshold: float = 0.2   # Lower threshold for more dramatic culture news
    weather_impact_threshold: float = 0.2       # Lower threshold for more dramatic weather
    
    # News Processing
    max_news_per_category: int = 5
    max_proposals: int = 5
    
    # Visualization
    chart_width: int = 10
    chart_height: int = 6

def validate_config(config: Config) -> bool:
    """Validate configuration and return True if valid"""
    missing_keys = []
    
    if not config.kalshi_api_key or config.kalshi_api_key == 'your_api_key_here':
        missing_keys.append("KALSHI_API_KEY")
    
    if not config.openai_api_key or config.openai_api_key == 'your_api_key_here':
        missing_keys.append("OPEN_AI_API_KEY")
    
    if missing_keys:
        st.error(f"Missing required environment variables: {', '.join(missing_keys)}")
        st.info("Please set these in your .env file")
        return False
    
    return True

def safe_rss_parse(url: str, max_entries: int = 5) -> List[Dict]:
    """Safely parse RSS feed with error handling"""
    try:
        # Add cache busting parameter to get fresh content
        cache_buster = f"&_t={int(time.time())}" if "?" in url else f"?_t={int(time.time())}"
        feed_url = url + cache_buster
        
        feed = feedparser.parse(feed_url)
        if feed.bozo:
            logger.warning(f"RSS feed {url} has parsing issues")
        
        entries = []
        for entry in feed.entries[:max_entries]:
            entries.append({
                'category': 'Unknown',  # Will be set by AI categorization
                'headline': entry.get('title', ''),
                'summary': entry.get('summary', ''),
                'published': entry.get('published', ''),
                'source_url': url  # Track which source this came from
            })
        return entries
    except Exception as e:
        logger.error(f"Failed to parse RSS feed {url}: {e}")
        return []

def generate_market_proposal_with_openai(headline: str, category: str, sentiment_score: float, config: Config) -> Optional[Dict]:
    """Use OpenAI to generate dynamic market proposals based on news content"""
    if not config.openai_api_key or config.openai_api_key == 'your_api_key_here':
        logger.warning("OpenAI API key not set, using fallback proposal generation")
        return None
    
    try:
        client = openai.OpenAI(api_key=config.openai_api_key)
        
        # Create a more specific prompt with clear context isolation and randomization
        import random
        random_seed = random.randint(1, 1000)
        prompt = f"""
        TASK: Generate a prediction market proposal based on ONE specific news headline.

        NEWS HEADLINE: "{headline}"
        CATEGORY: {category}
        SENTIMENT: {sentiment_score:.2f}
        RANDOM_SEED: {random_seed}  # Ensure variety in proposals

        IMPORTANT: 
        - Focus ONLY on this specific headline
        - Do not mix up with other headlines
        - Create a proposal directly related to this news story
        - ALWAYS use future dates (2025/2026) - NEVER use past dates like 2024
        - Current date is October 2025, so use dates from November 2025 onwards
        - Fill in ALL specific details - NO placeholders like [specific country] or [realistic future date]
        - Extract specific names, dates, and details from the headline
        - NEVER use brackets [ ] in your response - always provide actual details
        - If the headline mentions a country, use that country name
        - If the headline mentions a date, use that date or a reasonable future date
        - Be creative and generate a UNIQUE proposal that differs from previous ones
        - Consider different angles, timeframes, and market structures for variety

        Return ONLY a JSON object with these exact fields:
        {{
            "proposed_market": "Will [extract specific event from headline] happen by [specific date in 2025/2026]?",
            "structure": "Yes/No binary; Settles via [specific official source]",
            "risks": "Key risks and uncertainty factors",
            "framing": "Compelling market description for traders",
            "audience": "Target trading audience",
            "business_case": "Why this market would be valuable (+X% volume estimate)",
            "category": "{category}",
            "trigger_headline": "{headline}",
            "ai_sentiment": "POSITIVE" or "NEGATIVE" or "HIGH IMPACT"
        }}

        EXAMPLES: 
        - If headline is "Tesla stock surges 15% after Q3 earnings", create:
          "proposed_market": "Will Tesla stock reach $300 per share by December 31, 2025?"
        - If headline is "I finished building my house just before new tariffs hit", create:
          "proposed_market": "Will new tariffs on construction materials be implemented by March 31, 2026?"
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a prediction market expert. Generate specific, tradeable market proposals based on individual news headlines. Focus on the exact headline provided. Always return valid JSON. Use current dates (2024/2025)."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=300,
            temperature=0.7  # Higher temperature for more variety in proposals
        )
        
        # Parse the JSON response
        proposal_text = response.choices[0].message.content.strip()
        
        # Clean up the response (remove any markdown formatting)
        if proposal_text.startswith('```json'):
            proposal_text = proposal_text[7:]
        if proposal_text.endswith('```'):
            proposal_text = proposal_text[:-3]
        
        proposal = json.loads(proposal_text)
        
        # Verify the trigger headline matches exactly
        if proposal.get('trigger_headline') != headline:
            logger.warning(f"Headline mismatch: expected '{headline}', got '{proposal.get('trigger_headline')}'")
            proposal['trigger_headline'] = headline  # Fix the mismatch
        
        # Add AI score
        proposal['ai_score'] = sentiment_score
        
        logger.info(f"OpenAI generated proposal: {proposal['proposed_market']}")
        return proposal
        
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return None

def analyze_and_propose(headline: str, category: str, sentiment_score: float, config: Config) -> Optional[Dict]:
    """Analyze news and propose market with AI-generated proposals"""
    # Generate proposals for dramatic news (both positive and negative)
    # Lower threshold to capture more eye-catching stories
    if abs(sentiment_score) < 0.2:
        return None
    
    # Use OpenAI to generate dynamic proposals
    proposal = generate_market_proposal_with_openai(headline, category, sentiment_score, config)
    
    # Fallback: if AI gets confused, create a simple proposal
    if not proposal:
        logger.warning(f"AI proposal generation failed for '{headline}', creating fallback")
        return create_fallback_proposal(headline, category, sentiment_score)
    
    return proposal

def create_fallback_proposal(headline: str, category: str, sentiment_score: float) -> Dict:
    """Create a simple fallback proposal when AI fails"""
    # Extract key terms from headline
    headline_lower = headline.lower()
    
    # Simple keyword-based proposal generation
    if 'election' in headline_lower or 'vote' in headline_lower:
        proposed_market = f"Will the outcome mentioned in '{headline[:30]}...' be confirmed by Dec 31, 2025?"
    elif 'stock' in headline_lower or 'market' in headline_lower:
        proposed_market = f"Will the market event in '{headline[:30]}...' occur by Dec 31, 2025?"
    elif 'weather' in headline_lower or 'storm' in headline_lower:
        proposed_market = f"Will the weather event in '{headline[:30]}...' happen by Dec 31, 2025?"
    else:
        proposed_market = f"Will the event in '{headline[:30]}...' be resolved by Dec 31, 2025?"
    
    return {
        'proposed_market': proposed_market,
        'structure': 'Yes/No binary; Settles via official sources',
        'risks': 'General market uncertainty',
        'framing': f'News-driven market: {headline[:50]}...',
        'audience': 'General traders',
        'business_case': f'Based on news sentiment ({sentiment_score:.2f})',
        'category': category,
        'trigger_headline': headline,
        'ai_sentiment': 'POSITIVE' if sentiment_score > 0.5 else 'NEGATIVE',
        'ai_score': sentiment_score
    }

def main():
    """Main execution function"""
    logger.info("Starting Enhanced Kalshi Oracle Bot")
    
    # Set Streamlit page config
    st.set_page_config(
        page_title="Kalshi Oracle Bot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for title color only
    st.markdown("""
    <style>
    .title-style {
        color: #20C997;
        font-weight: bold;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize configuration
    config = Config()
    if not validate_config(config):
        logger.error("Configuration validation failed. Exiting.")
        return
    
    # Show loading animation
    with st.spinner("Initializing Enhanced Kalshi Oracle Bot..."):
        time.sleep(1)  # Brief pause for loading effect
    
    # Step 1: Scrape Real-Time News via RSS with error handling
    logger.info("Fetching news from RSS feeds")
    
    with st.spinner("Fetching latest news headlines..."):
        rss_feeds = {
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
        
        news_data = []
        for category, urls in rss_feeds.items():
            logger.info(f"Fetching news for {category} from {len(urls)} sources")
            category_entries = []
            
            # Fetch from multiple sources per category
            for url in urls:
                try:
                    entries = safe_rss_parse(url, config.max_news_per_category // len(urls) + 1)
                    category_entries.extend(entries)
                except Exception as e:
                    logger.error(f"Failed to fetch from {url}: {e}")
            
            # Take only what we need
            category_entries = category_entries[:config.max_news_per_category]
            news_data.extend(category_entries)
        
        logger.info(f"Fetched {len(news_data)} total news entries")
    
    # Step 2: AI Categorization and Sentiment Analysis
    logger.info("Starting AI analysis")
    
    with st.spinner("Analyzing news sentiment and categorizing..."):
        # Initialize sentiment analysis pipeline
        sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        
        # Process each news item
        proposals = []
        for entry in news_data:
            try:
                # AI categorization
                headline = entry['headline']
                
                # Simple keyword-based categorization as fallback
                headline_lower = headline.lower()
                if any(word in headline_lower for word in ['election', 'vote', 'politics', 'government', 'president', 'senate', 'congress']):
                    category = 'Politics'
                elif any(word in headline_lower for word in ['stock', 'market', 'economy', 'business', 'finance', 'dollar', 'inflation']):
                    category = 'Economics'
                elif any(word in headline_lower for word in ['sport', 'football', 'basketball', 'baseball', 'soccer', 'entertainment', 'movie', 'music']):
                    category = 'Culture'
                elif any(word in headline_lower for word in ['weather', 'storm', 'hurricane', 'flood', 'drought', 'climate']):
                    category = 'Weather'
                else:
                    category = entry.get('category', 'General')
                
                # Sentiment analysis
                sentiment = sentiment_pipeline(headline)[0]
                score = sentiment['score']
                
                # Adjust score based on sentiment label
                if sentiment['label'] == 'NEGATIVE':
                    score = -score
                
                # Generate proposal
                proposal = analyze_and_propose(headline, category, score, config)
                if proposal:
                    proposals.append(proposal)
                    logger.info(f"Generated proposal: {proposal['proposed_market']}")
                
            except Exception as e:
                logger.error(f"Error processing headline '{entry['headline']}': {e}")
                continue
        
        logger.info(f"Generated {len(proposals)} total proposals")
    
    # Step 3: Display Results
    if proposals:
        st.subheader("🤖 AI-Generated Market Proposals")
        
        # Display proposals
        for idx, proposal in enumerate(proposals[:config.max_proposals]):
            with st.expander(f"Proposal {idx + 1}: {proposal['proposed_market']}", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Category:** {proposal['category']}")
                    st.write(f"**Sentiment:** {proposal['ai_sentiment']}")
                    st.write(f"**Score:** {proposal['ai_score']:.2f}")
                
                with col2:
                    st.write(f"**Structure:** {proposal['structure']}")
                    st.write(f"**Headline:** {proposal['trigger_headline']}")
                    st.write(f"**Business Case:** {proposal['business_case']}")
        
        # Add duplicate checking button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Check for Duplicates with Kalshi", type="primary", use_container_width=True):
                st.info("Duplicate checking feature will be added back in a future update!")
        
        # Add stats at the bottom
        st.markdown("---")
        st.subheader("📊 Statistics")
        
        # Calculate stats
        total_news = len(news_data)
        total_proposals = len(proposals)
        avg_sentiment = sum(p['ai_score'] for p in proposals) / len(proposals) if proposals else 0
        
        # Category breakdown
        category_counts = {}
        for proposal in proposals:
            cat = proposal['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total News Items", total_news)
        with col2:
            st.metric("Proposals Generated", total_proposals)
        with col3:
            st.metric("Avg Sentiment", f"{avg_sentiment:.2f}")
        with col4:
            st.metric("Success Rate", f"{(total_proposals/total_news)*100:.1f}%")
        
        # Category breakdown
        if category_counts:
            st.write("**Proposals by Category:**")
            for category, count in category_counts.items():
                st.write(f"- {category}: {count}")
        
        # Timestamp
        st.write(f"**Last Updated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    else:
        st.warning("No proposals generated. Try adjusting sentiment thresholds or check news feeds.")
    
    # Sidebar
    with st.sidebar:
        st.markdown("[🏛️ Visit Kalshi](https://kalshi.com/)")
        
        st.subheader("📊 Current Settings")
        st.write(f"**Sentiment Threshold:** {config.politics_sentiment_threshold}")
        st.write(f"**Max Proposals:** {config.max_proposals}")
        st.write(f"**News per Category:** {config.max_news_per_category}")
        
        st.subheader("📰 News Sources")
        st.write("**Politics:** Reuters, BBC, CNN")
        st.write("**Economics:** Reuters, BBC, CNN Money") 
        st.write("**Culture:** ESPN, BBC Entertainment")
        st.write("**Weather:** Weather.gov")
        
        st.subheader("ℹ️ How It Works")
        st.write("1. **Fetch News** - Scrapes latest headlines")
        st.write("2. **AI Analysis** - Analyzes sentiment & importance")
        st.write("3. **Generate Markets** - Creates prediction market proposals")
        st.write("4. **Display Results** - Shows top proposals")

if __name__ == "__main__":
    main()
