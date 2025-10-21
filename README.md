# Enhanced Kalshi Oracle Bot 🤖📈

An AI-powered prediction market proposal generator that analyzes real-time news and creates unique Kalshi market proposals using OpenAI and sentiment analysis.

## 🎯 What This Bot Does

The Enhanced Kalshi Oracle Bot is a sophisticated system that:

1. **📡 Scrapes Real-Time News** - Fetches headlines from RSS feeds (Reuters, ESPN, Weather.gov)
2. **🤖 AI Categorization** - Uses OpenAI to intelligently categorize news into Politics, Weather, Culture, or Economics
3. **💭 AI Sentiment Analysis** - Analyzes emotional tone using DistilBERT (positive/negative sentiment)
4. **🎲 Dynamic Market Generation** - Creates unique prediction market proposals based on actual news content
5. **🔍 Duplicate Detection** - Checks against existing Kalshi markets to avoid duplicates
6. **📊 Interactive Dashboard** - Displays results in a professional Streamlit interface

## 🧠 AI Sentiment Analysis Explained

**AI Sentiment** is a measure of the emotional tone and impact of news headlines:

- **POSITIVE Sentiment** (0.0 to 1.0): Headlines with optimistic, favorable, or beneficial tone
- **NEGATIVE Sentiment** (-1.0 to 0.0): Headlines with pessimistic, concerning, or harmful tone
- **HIGH IMPACT** (|score| > 0.7): Headlines with significant emotional intensity

### How It Works:
- Uses **DistilBERT** (a lightweight BERT model) trained on sentiment analysis
- Processes each headline through a neural network
- Returns a confidence score between -1.0 and 1.0
- Only headlines with |sentiment| > 0.3 generate market proposals

### Example:
- "Tesla stock surges 15% after earnings beat" → **POSITIVE (0.8)**
- "Hurricane Maria approaches with 120mph winds" → **HIGH IMPACT (0.9)**
- "Company announces massive layoffs" → **NEGATIVE (-0.7)**

## 🏗️ Architecture

```
News RSS Feeds → AI Categorization → Sentiment Analysis → Market Generation → Duplicate Check → Dashboard
     ↓              ↓                    ↓                    ↓                ↓           ↓
Reuters/ESPN    OpenAI GPT-3.5      DistilBERT         OpenAI GPT-3.5    Fuzzy Match   Streamlit
Weather.gov     (Politics/Weather/  (Positive/Negative) (Dynamic Proposals) (Similarity)  (Web UI)
                Culture/Economics)  (Confidence Score)  (JSON Response)   (Prevent Dups)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key
- Kalshi API key (optional, for duplicate checking)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Kalshi
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   # Create .env file
   echo "OPEN_AI_API_KEY=your_openai_api_key_here" > .env
   echo "KALSHI_API_KEY=your_kalshi_api_key_here" >> .env
   ```

4. **Run the bot:**
   ```bash
   python3 -m streamlit run enhanced_oracle_bot.py
   ```

5. **Access the dashboard:**
   - Local: http://localhost:8501
   - Network: http://192.168.1.176:8501

## 📋 Configuration

The bot uses a `Config` class for all settings:

```python
@dataclass
class Config:
    # API Configuration
    kalshi_api_key: str = os.getenv('KALSHI_API_KEY')
    openai_api_key: str = os.getenv('OPEN_AI_API_KEY')
    api_url: str = 'https://trading-api.kalshi.com/v2'
    
    # AI Thresholds
    politics_sentiment_threshold: float = 0.3
    culture_sentiment_threshold: float = 0.3
    weather_impact_threshold: float = 0.3
    
    # News Processing
    max_news_per_category: int = 3
    max_proposals: int = 10
```

### Adjustable Parameters:
- **Sentiment Thresholds**: Minimum sentiment score to generate proposals
- **Max News per Category**: How many headlines to fetch from each RSS feed
- **Max Proposals**: Maximum number of proposals to generate

## 🔧 How It Works

### Step 1: News Scraping
- Fetches headlines from multiple RSS feeds
- Handles parsing errors gracefully
- Stores headlines with metadata (title, summary, published date)

### Step 2: AI Categorization
- Sends each headline to OpenAI GPT-3.5
- Categorizes into: Politics, Weather, Culture, Economics
- Maps Sports → Culture for proposal generation
- Falls back to 'Culture' if API fails

### Step 3: Sentiment Analysis
- Uses DistilBERT for fast sentiment analysis
- Processes headlines through neural network
- Returns confidence scores (-1.0 to 1.0)
- Only processes headlines with |sentiment| > 0.3

### Step 4: Market Proposal Generation
- Sends headline + category + sentiment to OpenAI
- Generates unique market proposals in JSON format
- Creates specific, measurable, tradeable markets
- Includes settlement sources and risk factors

### Step 5: Duplicate Detection
- Fetches existing Kalshi markets (if API key available)
- Uses fuzzy matching to detect similar proposals
- Prevents duplicate market creation
- Falls back to demo mode if no API access

### Step 6: Dashboard Display
- Shows proposals in expandable cards
- Displays summary statistics
- Category breakdown charts
- Configuration sidebar

## 📊 Output Files

The bot generates several output files:

- **`oracle_proposals.csv`** - Generated market proposals
- **`oracle_existing_markets.csv`** - Existing Kalshi markets (for reference)
- **`oracle_ai_viz.png`** - Visualization chart
- **`oracle_bot.log`** - Detailed execution logs

## 🧪 Testing

### Test AI Categorization:
```bash
python3 test_ai_categorization.py
```

### Test Dynamic Proposals:
```bash
python3 test_dynamic_proposals.py
```

### Test Kalshi API:
```bash
python3 test_kalshi_api.py
```

## 🔍 Troubleshooting

### Common Issues:

1. **"OpenAI API key not set"**
   - Add `OPEN_AI_API_KEY=your_key` to `.env` file

2. **"Kalshi API request failed: 401 Unauthorized"**
   - Add `KALSHI_API_KEY=your_key` to `.env` file
   - Bot will run in demo mode without duplicate checking

3. **"RSS feed has parsing issues"**
   - Normal warning, bot continues with available feeds
   - Some RSS feeds may be temporarily unavailable

4. **"No proposals generated"**
   - Check sentiment thresholds in config
   - Verify news feeds are accessible
   - Ensure OpenAI API key is valid

### Debug Mode:
- Check `oracle_bot.log` for detailed execution logs
- Use test scripts to verify individual components
- Monitor console output for real-time status

## 🎨 Customization

### Adding New RSS Feeds:
```python
rss_feeds = {
    'Politics': 'http://feeds.reuters.com/reuters/politicsNews',
    'Economics': 'http://feeds.reuters.com/reuters/businessNews',
    'Culture': 'http://www.espn.com/espn/rss/news',
    'Weather': 'https://www.weather.gov/rss',
    'Technology': 'https://feeds.feedburner.com/oreilly/radar'  # Add new feed
}
```

### Adjusting AI Prompts:
Modify the prompt in `generate_market_proposal_with_openai()` to change proposal style.

### Changing Sentiment Thresholds:
Update the `Config` class to adjust when proposals are generated.

## 📈 Performance

- **Processing Speed**: ~2-3 seconds per headline
- **API Calls**: 2 OpenAI calls per headline (categorization + proposal)
- **Memory Usage**: ~500MB (includes DistilBERT model)
- **Accuracy**: High for categorization, moderate for sentiment

## 🔒 Security

- API keys stored in `.env` file (not committed to git)
- No sensitive data logged
- HTTPS for all API calls
- Graceful error handling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI** for GPT-3.5 API
- **Hugging Face** for DistilBERT model
- **Kalshi** for prediction market platform
- **Streamlit** for dashboard framework
- **Reuters/ESPN/Weather.gov** for news feeds

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the logs in `oracle_bot.log`
3. Run the test scripts
4. Create an issue on GitHub

---

**Built with ❤️ for the prediction market community**
