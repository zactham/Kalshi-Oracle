import os
import feedparser
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
# from kalshi_py import Client  # Official Kalshi SDK - recommended for production
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

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('oracle_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class for the Oracle Bot"""
    # API Configuration
    kalshi_api_key: str = os.getenv('KALSHI_API_KEY')
    openai_api_key: str = os.getenv('OPEN_AI_API_KEY')
    api_url: str = 'https://api.elections.kalshi.com/v1'
    
    # AI Thresholds
    politics_sentiment_threshold: float = 0.3
    culture_sentiment_threshold: float = 0.3
    weather_impact_threshold: float = 0.3
    
    # News Processing
    max_news_per_category: int = 5
    max_proposals: int = 5
    
    # Visualization
    chart_width: int = 10
    chart_height: int = 6

def validate_config(config: Config) -> bool:
    """Validate configuration and return True if valid"""
    if not config.kalshi_api_key or config.kalshi_api_key == 'your_api_key_here':
        logger.error("KALSHI_API_KEY environment variable not set or invalid")
        return False
    return True

def safe_api_call(func, *args, **kwargs):
    """Safely execute API calls with proper error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"API call failed: {e}")
        return None

def kalshi_get_markets(api_key: str, api_url: str, categories: List[str] = None):
    """Kalshi API client using proper authentication"""
    try:
        import time
        import hashlib
        import hmac
        import base64
        
        # Get current timestamp in milliseconds
        timestamp = str(int(time.time() * 1000))
        
        # Build URL with categories if provided
        url = f"{api_url}/markets"
        if categories:
            category_params = '&'.join([f'categories={cat}' for cat in categories])
            url += f"?{category_params}"
        
        # For now, use simple Bearer token approach since we don't have private key
        # In production, you'd need the full Kalshi authentication with RSA-PSS signing
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'Kalshi-Oracle-Bot/1.0'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Handle empty responses
        if not response.text.strip():
            logger.warning("Kalshi API returned empty response")
            return {'markets': []}
        
        data = response.json()
        logger.info(f"Kalshi API response: {len(data.get('markets', []))} markets found")
        return data
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Kalshi API request failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Kalshi API error: {e}")
        return None

def categorize_headline_with_openai(headline: str, config: Config) -> str:
    """Use OpenAI to categorize headlines into Politics, Weather, Culture, or Economics"""
    if not config.openai_api_key or config.openai_api_key == 'your_openai_api_key_here':
        logger.warning("OpenAI API key not set, using fallback categorization")
        return 'Culture'  # Default fallback
    
    try:
        client = openai.OpenAI(api_key=config.openai_api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a news categorization expert. Categorize news headlines into exactly one of these categories: Politics, Weather, Culture, Economics. Return only the category name, nothing else."
                },
                {
                    "role": "user", 
                    "content": f"Categorize this headline: '{headline}'"
                }
            ],
            max_tokens=10,
            temperature=0.1
        )
        
        category = response.choices[0].message.content.strip()
        # Map Sports to Culture for our proposal system
        if category.lower() == 'sports':
            category = 'Culture'
        logger.info(f"OpenAI categorized '{headline}' as: {category}")
        return category
        
    except openai.RateLimitError as e:
        logger.error(f"OpenAI API rate limit exceeded during categorization: {e}")
        st.warning("🚫 OpenAI API rate limit exceeded. Using fallback categorization.")
        return 'Culture'
    except openai.APIConnectionError as e:
        logger.error(f"OpenAI API connection error during categorization: {e}")
        st.warning("🌐 OpenAI API connection failed. Using fallback categorization.")
        return 'Culture'
    except openai.AuthenticationError as e:
        logger.error(f"OpenAI API authentication error during categorization: {e}")
        st.warning("🔑 OpenAI API authentication failed. Using fallback categorization.")
        return 'Culture'
    except openai.APITimeoutError as e:
        logger.error(f"OpenAI API timeout during categorization: {e}")
        st.warning("⏰ OpenAI API request timed out. Using fallback categorization.")
        return 'Culture'
    except openai.InternalServerError as e:
        logger.error(f"OpenAI API internal server error during categorization: {e}")
        st.warning("🔧 OpenAI API internal error. Using fallback categorization.")
        return 'Culture'
    except openai.APIError as e:
        if "insufficient_quota" in str(e).lower() or "billing" in str(e).lower():
            logger.error(f"OpenAI API credits exhausted during categorization: {e}")
            st.error("💳 OpenAI API credits exhausted. Please add credits to your account.")
            return 'Culture'
        else:
            logger.error(f"OpenAI API error during categorization: {e}")
            st.warning(f"🤖 OpenAI API error. Using fallback categorization.")
            return 'Culture'
    except Exception as e:
        logger.error(f"OpenAI categorization failed: {e}")
        st.warning(f"⚠️ AI categorization failed: {e}. Using fallback.")
        return 'Culture'  # Fallback

def safe_rss_parse(url: str, max_entries: int = 3) -> List[Dict]:
    """Safely parse RSS feeds with error handling and cache busting"""
    try:
        # Add cache busting parameter to get fresh content
        import time
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
    if not config.openai_api_key or config.openai_api_key == 'your_openai_api_key_here':
        logger.warning("OpenAI API key not set, using fallback proposal generation")
        return None
    
    try:
        client = openai.OpenAI(api_key=config.openai_api_key)
        
        # Create a more specific prompt with clear context isolation
        prompt = f"""
        TASK: Generate a prediction market proposal based on ONE specific news headline.

        NEWS HEADLINE: "{headline}"
        CATEGORY: {category}
        SENTIMENT: {sentiment_score:.2f}

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
            temperature=0.3  # Lower temperature for more consistent results
        )
        
        # Parse the JSON response
        import json
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
        
    except openai.RateLimitError as e:
        logger.error(f"OpenAI API rate limit exceeded: {e}")
        st.error("🚫 OpenAI API rate limit exceeded. Please wait a moment and try again.")
        return None
    except openai.APIConnectionError as e:
        logger.error(f"OpenAI API connection error: {e}")
        st.error("🌐 OpenAI API connection failed. Please check your internet connection.")
        return None
    except openai.AuthenticationError as e:
        logger.error(f"OpenAI API authentication error: {e}")
        st.error("🔑 OpenAI API authentication failed. Please check your API key.")
        return None
    except openai.APITimeoutError as e:
        logger.error(f"OpenAI API timeout: {e}")
        st.error("⏰ OpenAI API request timed out. Please try again.")
        return None
    except openai.InternalServerError as e:
        logger.error(f"OpenAI API internal server error: {e}")
        st.error("🔧 OpenAI API internal error. Please try again later.")
        return None
    except openai.APIError as e:
        if "insufficient_quota" in str(e).lower() or "billing" in str(e).lower():
            logger.error(f"OpenAI API credits exhausted: {e}")
            st.error("💳 OpenAI API credits exhausted. Please add credits to your account.")
            return None
        else:
            logger.error(f"OpenAI API error: {e}")
            st.error(f"🤖 OpenAI API error: {e}")
            return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse OpenAI JSON response: {e}")
        st.warning("⚠️ Failed to parse AI response. Using fallback proposal.")
        return None
    except Exception as e:
        logger.error(f"OpenAI proposal generation failed: {e}")
        st.warning(f"⚠️ AI proposal generation failed: {e}")
        return None

def analyze_and_propose(headline: str, category: str, sentiment_score: float, config: Config) -> Optional[Dict]:
    """Analyze news and propose market with AI-generated proposals"""
    # Only generate proposals for headlines with sufficient sentiment impact
    if abs(sentiment_score) < 0.3:
        return None
    
    # Use OpenAI to generate dynamic proposals
    proposal = generate_market_proposal_with_openai(headline, category, sentiment_score, config)
    
    # Fallback: if AI gets confused, create a simple proposal
    if not proposal:
        logger.warning(f"AI proposal generation failed for '{headline}', creating fallback")
        return create_fallback_proposal(headline, category, sentiment_score)
    
    return proposal

def rank_proposals_by_importance(proposals: List[Dict], config: Config) -> List[Dict]:
    """Use AI to rank proposals by importance and return top 5"""
    if not config.openai_api_key or config.openai_api_key == 'your_openai_api_key_here':
        logger.warning("OpenAI API key not set, using simple ranking")
        # Simple fallback: sort by absolute sentiment score
        return sorted(proposals, key=lambda x: abs(x.get('ai_score', 0)), reverse=True)[:config.max_proposals]
    
    try:
        client = openai.OpenAI(api_key=config.openai_api_key)
        
        # Create prompt for importance ranking
        proposals_text = "\n".join([f"{i+1}. {p['proposed_market']} (Sentiment: {p.get('ai_score', 0):.2f})" 
                                   for i, p in enumerate(proposals)])
        
        prompt = f"""
        Rank these prediction market proposals by importance for trading volume and market impact.
        Consider: news relevance, public interest, market potential, and trading appeal.
        
        PROPOSALS:
        {proposals_text}
        
        Return ONLY a JSON array with the top 5 most important proposal numbers in order of importance:
        [1, 3, 2, 5, 4]
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a prediction market expert. Rank proposals by trading importance and market appeal. Return only a JSON array of numbers."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        # Parse ranking
        import json
        ranking_text = response.choices[0].message.content.strip()
        if ranking_text.startswith('```json'):
            ranking_text = ranking_text[7:]
        if ranking_text.endswith('```'):
            ranking_text = ranking_text[:-3]
        
        ranking = json.loads(ranking_text)
        
        # Reorder proposals based on AI ranking
        ranked_proposals = []
        for rank in ranking[:config.max_proposals]:
            if 1 <= rank <= len(proposals):
                ranked_proposals.append(proposals[rank-1])
        
        logger.info(f"AI ranked {len(ranked_proposals)} proposals by importance")
        return ranked_proposals
        
    except Exception as e:
        logger.error(f"AI ranking failed: {e}")
        # Fallback to simple ranking
        return sorted(proposals, key=lambda x: abs(x.get('ai_score', 0)), reverse=True)[:config.max_proposals]

def create_fallback_proposal(headline: str, category: str, sentiment_score: float) -> Dict:
    """Create a simple fallback proposal when AI fails"""
    # Extract key terms from headline
    headline_lower = headline.lower()
    
    # Simple keyword-based proposal generation
    if 'election' in headline_lower or 'vote' in headline_lower:
        proposed_market = f"Will the outcome mentioned in '{headline[:30]}...' be confirmed by Dec 31, 2024?"
    elif 'stock' in headline_lower or 'market' in headline_lower:
        proposed_market = f"Will the market event in '{headline[:30]}...' occur by Dec 31, 2024?"
    elif 'weather' in headline_lower or 'storm' in headline_lower:
        proposed_market = f"Will the weather event in '{headline[:30]}...' happen by Dec 31, 2024?"
    else:
        proposed_market = f"Will the event in '{headline[:30]}...' be resolved by Dec 31, 2024?"
    
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
    
    # Set Streamlit page config with custom background
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
    
    # Step 1: Initialize Kalshi Client with API Key
    logger.info("Initializing Kalshi client")
    
    with st.spinner("Connecting to Kalshi API..."):
        # Fetch Kalshi markets with error handling
        df_kalshi = pd.DataFrame()
        if config.kalshi_api_key and config.kalshi_api_key != 'your_api_key_here':
            try:
                markets_response = kalshi_get_markets(
                    config.kalshi_api_key,
                    config.api_url,
                    categories=['Politics', 'Weather', 'Culture', 'Economics']
                )
                if markets_response and 'markets' in markets_response:
                    df_kalshi = pd.DataFrame(markets_response['markets'])
                    df_kalshi['market_title'] = df_kalshi['title']
                    df_kalshi['category'] = df_kalshi['category']
                    logger.info(f"Fetched {len(df_kalshi)} real Kalshi markets")
                else:
                    logger.warning("No markets data received from Kalshi API")
            except Exception as e:
                logger.error(f"Kalshi API error: {e}")
                df_kalshi = pd.DataFrame()
        else:
            logger.warning("Kalshi API key not set - running in demo mode without duplicate checking")
            # Create some sample markets for demo purposes
            df_kalshi = pd.DataFrame({
                'market_title': [
                    'Will Trump win the 2024 election?',
                    'Will there be a recession in 2024?',
                    'Will Bitcoin reach $100k in 2024?'
                ],
                'category': ['Politics', 'Economics', 'Culture']
            })

    # Step 2: Scrape Real-Time News via RSS with error handling
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
                    logger.info(f"  Fetched {len(entries)} entries from {url}")
                except Exception as e:
                    logger.warning(f"  Failed to fetch from {url}: {e}")
                    continue
            
            # Shuffle and limit to max_news_per_category
            import random
            random.shuffle(category_entries)
            category_entries = category_entries[:config.max_news_per_category]
            
            for entry in category_entries:
                # Use OpenAI to categorize the headline
                ai_category = categorize_headline_with_openai(entry['headline'], config)
                entry['category'] = ai_category
                entry['original_category'] = category  # Keep original for reference
                news_data.append(entry)
        
        df_news = pd.DataFrame(news_data)
        logger.info(f"Scraped {len(df_news)} news headlines")
        
        # Log category distribution
        category_counts = df_news['category'].value_counts()
        logger.info(f"AI-categorized headlines: {dict(category_counts)}")

    # Step 3: External AI Analysis with error handling
    logger.info("Initializing AI sentiment analysis")
    
    with st.spinner("Initializing AI sentiment analysis..."):
        try:
            sentiment_pipeline = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
            logger.info("AI pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI pipeline: {e}")
            st.error(f"❌ Failed to initialize AI pipeline: {e}")
            return

    # Generate proposals with error handling
    logger.info("Generating market proposals")
    
    with st.spinner("Generating AI-powered market proposals..."):
        proposals = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, row in df_news.iterrows():
            try:
                # Update progress
                progress = (idx + 1) / len(df_news)
                progress_bar.progress(progress)
                status_text.text(f"Analyzing headline {idx + 1}/{len(df_news)}: {row['headline'][:50]}...")
                
                logger.info(f"Analyzing headline {idx + 1}/{len(df_news)}: '{row['headline']}' (Category: {row['category']})")
                sentiment = sentiment_pipeline(row['headline'])[0]
                score = sentiment['score'] if sentiment['label'] == 'POSITIVE' else -sentiment['score']
                logger.info(f"Sentiment: {sentiment['label']} (score: {score:.3f})")
                
                # Add small delay to prevent API context bleeding
                if idx > 0:
                    time.sleep(0.5)
                
                proposal = analyze_and_propose(row['headline'], row['category'], score, config)
                if proposal:
                    proposal['ai_score'] = score
                    proposals.append(proposal)
                    logger.info(f"✅ Generated proposal: {proposal['proposed_market']}")
                    logger.info(f"   Trigger headline: {proposal['trigger_headline']}")
                else:
                    logger.info("❌ No proposal generated for this headline")
            except Exception as e:
                logger.error(f"Failed to analyze headline '{row['headline']}': {e}")
                continue
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        df_proposals = pd.DataFrame(proposals)
        logger.info(f"Generated {len(df_proposals)} proposals")
        
        # AI-based importance ranking to get top 5
        if len(proposals) > config.max_proposals:
            with st.spinner("AI ranking proposals by importance..."):
                proposals = rank_proposals_by_importance(proposals, config)
                df_proposals = pd.DataFrame(proposals)
                logger.info(f"AI selected top {len(proposals)} most important proposals")

    # Step 4: Validate Against Real Kalshi Markets
    with st.spinner("Checking for duplicate markets..."):
        if not df_proposals.empty and not df_kalshi.empty:
            logger.info("Checking for duplicate markets")
            df_proposals['is_duplicate'] = False
            
            for idx, proposal in df_proposals.iterrows():
                market_title = proposal['proposed_market']
                # Check for similar markets using fuzzy matching
                is_duplicate = False
                for _, existing_market in df_kalshi.iterrows():
                    existing_title = existing_market['market_title']
                    # Check if the proposed market is too similar to existing ones
                    if (market_title.lower() in existing_title.lower() or 
                        existing_title.lower() in market_title.lower() or
                        len(set(market_title.lower().split()) & set(existing_title.lower().split())) >= 3):
                        is_duplicate = True
                        logger.info(f"Found duplicate: '{market_title}' similar to '{existing_title}'")
                        break
                
                df_proposals.at[idx, 'is_duplicate'] = is_duplicate
            
            unique_proposals = df_proposals[~df_proposals['is_duplicate']]
            logger.info(f"Found {len(unique_proposals)} unique proposals after duplicate check")
        else:
            unique_proposals = df_proposals
            if df_proposals.empty:
                logger.warning("No proposals generated; skipping duplicate check")
            else:
                logger.warning("No Kalshi markets fetched; skipping duplicate check")

# Step 5: Output & Export
    logger.info("Generating output and exports")
    
    with st.spinner("Generating reports and visualizations..."):
        print("=== Enhanced Kalshi Oracle Bot: AI-Proposed Markets (Oct 20, 2025) ===")
        if not unique_proposals.empty:
            display_cols = ['proposed_market', 'category', 'framing', 'business_case', 'ai_sentiment', 'trigger_headline']
            print(unique_proposals[display_cols].to_string(index=False))
            
            # Export to CSV
            try:
                unique_proposals.to_csv('oracle_proposals.csv', index=False)
                df_kalshi.to_csv('oracle_existing_markets.csv', index=False)
                logger.info("Exported to 'oracle_proposals.csv' and 'oracle_existing_markets.csv'")
            except Exception as e:
                logger.error(f"Failed to export CSV files: {e}")
        else:
            print("No unique AI proposals—check news feeds or AI thresholds")
            logger.warning("No unique proposals generated")

    # Step 6: AI Visualization
    if not unique_proposals.empty:
        logger.info("Generating visualization")
        try:
            impact = unique_proposals['business_case'].str.extract('(\d+)%').astype(float)[0].fillna(10)
            colors = ['green' if s == 'POSITIVE' else 'red' for s in unique_proposals['ai_sentiment']]
            plt.figure(figsize=(config.chart_width, config.chart_height))
            bars = plt.bar(unique_proposals['proposed_market'], impact, color=colors)
            plt.title('AI-Driven Market Proposals: Projected Volume Boost')
            plt.ylabel('Est. Liquidity Increase (%)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('oracle_ai_viz.png')
            plt.show()
            logger.info("AI visualization saved as 'oracle_ai_viz.png'")
        except Exception as e:
            logger.error(f"Failed to generate visualization: {e}")

# Step 7: Streamlit Dashboard
    logger.info("Initializing Streamlit dashboard")
    
    # Success message
    if not unique_proposals.empty:
        st.success(f"Successfully generated {len(unique_proposals)} unique market proposals!")
    else:
        st.warning("No unique proposals generated. Check configuration or news feeds.")
    
    st.markdown('<h1 class="title-style">🤖 Kalshi Oracle Bot: AI-Powered Market Proposals</h1>', unsafe_allow_html=True)
    st.write("Fetches live Kalshi markets, analyzes news with DistilBERT, and proposes new contracts.")
    
    # Show last updated time
    from datetime import datetime
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not unique_proposals.empty:
        st.subheader("Market Proposals")
        
        # Display proposals in a nice format
        for idx, proposal in unique_proposals.iterrows():
            with st.expander(f"Proposal {idx + 1}: {proposal['proposed_market']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Category:** {proposal['category']}")
                    st.write(f"**AI Sentiment:** {proposal['ai_sentiment']}")
                    st.write(f"**Trigger Headline:** {proposal['trigger_headline']}")
                
                with col2:
                    st.write(f"**Structure:** {proposal['structure']}")
                    st.write(f"**Risks:** {proposal['risks']}")
                    st.write(f"**Audience:** {proposal['audience']}")
                
                st.write(f"**Framing:** {proposal['framing']}")
                st.write(f"**Business Case:** {proposal['business_case']}")
        
        # Summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Proposals", len(unique_proposals))
        with col2:
            st.metric("Categories", unique_proposals['category'].nunique())
        with col3:
            st.metric("Positive Sentiment", len(unique_proposals[unique_proposals['ai_sentiment'] == 'POSITIVE']))
        
        # Category breakdown
        st.subheader("Proposals by Category")
        category_counts = unique_proposals['category'].value_counts()
        st.bar_chart(category_counts)
        
    else:
        st.warning("No unique AI proposals generated. Check news feeds or adjust AI thresholds.")
        st.info("Try adjusting the sentiment thresholds in the configuration or check if news feeds are accessible.")
    
    # Configuration display
    with st.sidebar:
        st.subheader("Configuration")
        st.write(f"Politics Sentiment Threshold: {config.politics_sentiment_threshold}")
        st.write(f"Culture Sentiment Threshold: {config.culture_sentiment_threshold}")
        st.write(f"Max News per Category: {config.max_news_per_category}")
        st.write(f"Max Proposals: {config.max_proposals}")
        
        st.subheader("Actions")
        if st.button("Refresh Data", type="primary"):
            # Clear any cached data and force refresh
            st.cache_data.clear()
            st.rerun()
        
        st.subheader("News Sources")
        st.write("**Politics:** Reuters, BBC, CNN")
        st.write("**Economics:** Reuters, BBC, CNN Money")
        st.write("**Culture:** ESPN, BBC Entertainment, CNN Entertainment, BBC Technology")
        st.write("**Weather:** Weather.gov, BBC Science")
        
        st.subheader("Downloads")
        if st.button("Download Proposals CSV"):
            if not unique_proposals.empty:
                csv = unique_proposals.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="oracle_proposals.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No proposals to download")

if __name__ == "__main__":
    main()