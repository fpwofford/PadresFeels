import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import feedparser
import os
from textblob import TextBlob
import time

# Page configuration
st.set_page_config(
    page_title="Padres Sentiment Tracker",
    page_icon="⚾",
    layout="wide"
)

# Custom CSS for professional Padres-themed design
st.markdown("""
    <style>
    /* Import professional font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global font */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main dashboard background - Padres brown/sand */
    .main {
        background-color: #A0AAB2;
        padding: 2rem;
    }
    
    /* Professional title styling */
    .main h1 {
        color: #2F241D !important;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Subtitle styling */
    .main h3 {
        color: #2F241D !important;
        font-weight: 400 !important;
        font-size: 1.1rem !important;
        margin-bottom: 2rem !important;
        opacity: 0.8;
    }
    
    /* Section headers */
    .main h2 {
        color: #2F241D !important;
        font-weight: 600 !important;
        font-size: 1.8rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1.5rem !important;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #FFC425;
    }
    
    /* Subheaders */
    .main .stSubheader {
        color: #2F241D !important;
        font-weight: 600 !important;
    }
    
    /* Metric containers - card style */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #ffffff, #f0f0f0);
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 2px 4px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(47, 36, 29, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15), 0 3px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric labels */
    [data-testid="stMetricLabel"] {
        color: #2F241D !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Metric values */
    [data-testid="stMetricValue"] {
        color: #2F241D !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
    }
    
    /* Metric deltas */
    [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }
    
    /* Chart containers */
    .js-plotly-plot {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(47, 36, 29, 0.1);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2F241D 0%, #1a1612 100%);
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #FFC425 !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #FFC425;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(145deg, #FFC425, #e6b022);
        color: #2F241D !important;
        font-weight: 600 !important;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        transition: all 0.2s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.9rem;
    }
    
    .stButton > button:hover {
        background: linear-gradient(145deg, #e6b022, #FFC425);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        transform: translateY(-1px);
    }
    
    /* Info boxes */
    .stAlert {
        background: white;
        border-radius: 8px;
        border-left: 4px solid #FFC425;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 8px;
        font-weight: 600;
        color: #2F241D !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Horizontal rule styling */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #FFC425, transparent);
        margin: 2rem 0;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Status badges */
    .main p strong {
        background: #FFC425;
        color: #2F241D;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    
    /* Slider styling */
    .stSlider [data-baseweb="slider"] {
        background: rgba(255, 196, 37, 0.2);
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: #2F241D;
        color: white !important;
        border-radius: 6px;
        font-weight: 500;
    }
    
    .stDownloadButton > button:hover {
        background: #1a1612;
    }
    </style>
    """, unsafe_allow_html=True)

# File to store historical data
DATA_FILE = 'padres_sentiment_history.csv'

# Function to get sentiment from text using TextBlob
def analyze_sentiment(text):
    """Analyze sentiment of text. Returns score from -1 to 1"""
    try:
        blob = TextBlob(text)
        # TextBlob returns polarity from -1 to 1
        return blob.sentiment.polarity
    except:
        return 0

# Function to fetch Padres news from RSS feeds
def fetch_padres_news(max_articles=50):
    """Fetch recent news about Padres from multiple sources"""
    
    news_items = []
    
    # RSS feeds to check (these are public and don't require authentication)
    feeds = [
        'https://www.mlb.com/feeds/news/rss.xml',  # MLB general news
        'https://www.espn.com/espn/rss/mlb/news',   # ESPN MLB news
    ]
    
    st.info("🔄 Fetching Padres news...")
    
    for feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:max_articles]:  # Get articles from each source
                title = entry.get('title', '')
                # Only include if it mentions Padres or related keywords
                if any(keyword in title.lower() for keyword in ['padres', 'san diego', 'petco park', 'machado', 'tatis', 'king', 'michael king']):
                    # Parse the published date - handle multiple formats
                    pub_date = entry.get('published_parsed', None)
                    if pub_date:
                        try:
                            # Convert time struct to simple date string
                            date_str = f"{pub_date.tm_year}-{pub_date.tm_mon:02d}-{pub_date.tm_mday:02d}"
                        except:
                            date_str = datetime.now().strftime('%Y-%m-%d')
                    else:
                        date_str = datetime.now().strftime('%Y-%m-%d')
                    
                    news_items.append({
                        'title': title,
                        'date': date_str,
                        'source': feed_url
                    })
        except Exception as e:
            st.warning(f"Could not fetch from {feed_url}: {str(e)}")
            continue
    
    return news_items

# Function to categorize news items
def categorize_news(title):
    """Determine what category a news item belongs to"""
    title_lower = title.lower()
    
    categories = {
        'team': False,
        'machado': False,
        'tatis': False,
        'king': False,
        'gm': False,
        'fans': False,
        'stadium': False
    }
    
    # Team overall - general team news
    if any(word in title_lower for word in ['padres', 'team', 'roster', 'season', 'win', 'loss', 'game']):
        categories['team'] = True
    
    # Players
    if 'machado' in title_lower or 'manny' in title_lower:
        categories['machado'] = True
    if 'tatis' in title_lower or 'fernando' in title_lower:
        categories['tatis'] = True
    if 'king' in title_lower or 'michael' in title_lower:
        categories['king'] = True
    
    # GM/Front Office
    if any(word in title_lower for word in ['preller', 'trade', 'sign', 'deal', 'contract', 'front office', 'management']):
        categories['gm'] = True
    
    # Fans
    if any(word in title_lower for word in ['fan', 'crowd', 'attendance', 'support']):
        categories['fans'] = True
    
    # Stadium
    if any(word in title_lower for word in ['petco', 'stadium', 'park', 'venue']):
        categories['stadium'] = True
    
    return categories

# Function to process news and update sentiment data
def process_news_sentiment():
    """Fetch news, analyze sentiment, and return current scores"""
    
    news_items = fetch_padres_news(max_articles=30)
    
    if not news_items:
        st.warning("⚠️ Could not fetch news. Using previous data.")
        return None
    
    st.success(f"✅ Found {len(news_items)} Padres-related articles!")
    
    # Initialize sentiment accumulators
    sentiment_totals = {
        'team': [],
        'machado': [],
        'tatis': [],
        'king': [],
        'gm': [],
        'fans': [],
        'stadium': []
    }
    
    # Store headlines with their sentiment for display
    headlines_with_sentiment = []
    
    # Analyze each news item
    for item in news_items:
        sentiment = analyze_sentiment(item['title'])
        categories = categorize_news(item['title'])
        
        headlines_with_sentiment.append({
            'title': item['title'],
            'sentiment': sentiment,
            'date': item['date']
        })
        
        # Add sentiment to relevant categories
        for category, is_relevant in categories.items():
            if is_relevant:
                sentiment_totals[category].append(sentiment)
    
    # Calculate average sentiment for each category
    current_sentiment = {}
    for category, scores in sentiment_totals.items():
        if scores:
            current_sentiment[category] = np.mean(scores)
        else:
            current_sentiment[category] = 0  # Neutral if no data
    
    # Save headlines to a separate file for display
    headlines_df = pd.DataFrame(headlines_with_sentiment)
    # Ensure dates are simple strings
    headlines_df['date'] = headlines_df['date'].astype(str)
    headlines_df.to_csv('padres_recent_headlines.csv', index=False)
    
    return current_sentiment

# Function to backfill historical data
def backfill_historical_data(days=30):
    """Fetch and analyze historical news to populate past 30 days"""
    
    st.info(f"🔄 Backfilling {days} days of historical data... This may take a minute.")
    progress_bar = st.progress(0)
    
    # Fetch a large batch of news articles
    news_items = fetch_padres_news(max_articles=200)
    
    if not news_items:
        st.error("❌ Could not fetch historical news")
        return False
    
    # Group news by date
    news_by_date = {}
    for item in news_items:
        date = item['date']
        if date not in news_by_date:
            news_by_date[date] = []
        news_by_date[date].append(item)
    
    # Get the last 30 days
    end_date = datetime.now()
    date_list = [(end_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days-1, -1, -1)]
    
    # Process each date
    df = load_historical_data()
    records_added = 0
    
    for idx, date in enumerate(date_list):
        progress_bar.progress((idx + 1) / len(date_list))
        
        # Skip if we already have data for this date
        if not df.empty and date in df['Date'].astype(str).values:
            continue
        
        # Get news for this date
        day_news = news_by_date.get(date, [])
        
        if not day_news:
            # If no news for this date, use neutral sentiment or skip
            continue
        
        # Calculate sentiment for this date
        sentiment_totals = {
            'team': [],
            'machado': [],
            'tatis': [],
            'king': [],
            'gm': [],
            'fans': [],
            'stadium': []
        }
        
        for item in day_news:
            sentiment = analyze_sentiment(item['title'])
            categories = categorize_news(item['title'])
            
            for category, is_relevant in categories.items():
                if is_relevant:
                    sentiment_totals[category].append(sentiment)
        
        # Calculate averages
        day_sentiment = {}
        for category, scores in sentiment_totals.items():
            if scores:
                day_sentiment[category] = np.mean(scores)
            else:
                day_sentiment[category] = 0
        
        # Create record
        new_row = {
            'Date': date,
            'Team_Overall': day_sentiment.get('team', 0),
            'Players_Machado': day_sentiment.get('machado', 0),
            'Players_Tatis': day_sentiment.get('tatis', 0),
            'Players_King': day_sentiment.get('king', 0),
            'GM_Front_Office': day_sentiment.get('gm', 0),
            'Fan_Morale': day_sentiment.get('fans', 0),
            'Stadium_Experience': day_sentiment.get('stadium', 0)
        }
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        records_added += 1
    
    # Sort by date and save
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.to_csv(DATA_FILE, index=False)
    
    progress_bar.empty()
    st.success(f"✅ Backfilled {records_added} days of historical data!")
    return True

# Function to load historical data
def load_historical_data():
    """Load sentiment history from CSV file"""
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    else:
        # Create empty dataframe with proper structure
        return pd.DataFrame(columns=['Date', 'Team_Overall', 'Players_Machado', 'Players_Tatis', 
                                    'Players_King', 'GM_Front_Office', 'Fan_Morale', 'Stadium_Experience'])

# Function to generate sample data for demo
def generate_sample_data_demo():
    """Generate 30 days of realistic sample data for demo purposes"""
    
    # DELETE any old files first to ensure clean slate
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    if os.path.exists('padres_recent_headlines.csv'):
        os.remove('padres_recent_headlines.csv')
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=29)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    data = {
        'Date': dates,
        'Team_Overall': [],
        'Players_Machado': [],
        'Players_Tatis': [],
        'Players_King': [],
        'GM_Front_Office': [],
        'Fan_Morale': [],
        'Stadium_Experience': []
    }
    
    np.random.seed(42)
    
    for i in range(len(dates)):
        # Team overall - simulating ups and downs with wins/losses
        base_team = 0.25 + 0.35 * np.sin(i / 8) + np.random.normal(0, 0.12)
        data['Team_Overall'].append(np.clip(base_team, -1, 1))
        
        # Machado - consistent positive sentiment
        machado = 0.55 + np.random.normal(0, 0.08)
        data['Players_Machado'].append(np.clip(machado, -1, 1))
        
        # Tatis - more volatile, recovering from injury news
        tatis = 0.15 + 0.4 * np.sin(i / 12) + np.random.normal(0, 0.15)
        data['Players_Tatis'].append(np.clip(tatis, -1, 1))
        
        # King - steady pitcher performance
        king = 0.40 + np.random.normal(0, 0.10)
        data['Players_King'].append(np.clip(king, -1, 1))
        
        # GM/Front Office - varying based on trade deadline moves
        gm = 0.05 + 0.3 * np.sin(i / 15) + np.random.normal(0, 0.14)
        data['GM_Front_Office'].append(np.clip(gm, -1, 1))
        
        # Fan morale - follows team performance
        fan_morale = 0.30 + 0.3 * np.sin(i / 10) + np.random.normal(0, 0.13)
        data['Fan_Morale'].append(np.clip(fan_morale, -1, 1))
        
        # Stadium experience - consistently positive
        stadium = 0.65 + 0.1 * np.sin(i / 20) + np.random.normal(0, 0.07)
        data['Stadium_Experience'].append(np.clip(stadium, -1, 1))
    
    df = pd.DataFrame(data)
    df.to_csv(DATA_FILE, index=False)
    
    # Also generate sample headlines for demo - SIMPLE DATE FORMAT
    sample_headlines = [
        {"title": "Padres secure crucial series win against division rivals", "sentiment": 0.65, "date": datetime.now().strftime('%Y-%m-%d')},
        {"title": "Machado's hot streak continues with two-homer game", "sentiment": 0.72, "date": (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')},
        {"title": "Tatis returns to lineup after injury scare", "sentiment": 0.35, "date": (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')},
        {"title": "King strikes out 10 in dominant pitching performance", "sentiment": 0.68, "date": (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')},
        {"title": "Padres bullpen struggles in late innings", "sentiment": -0.45, "date": (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')},
        {"title": "Front office explores trade options before deadline", "sentiment": 0.15, "date": (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')},
        {"title": "Petco Park attendance hits season high", "sentiment": 0.55, "date": (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')},
        {"title": "Injury concerns mount as key players hit IL", "sentiment": -0.52, "date": (datetime.now() - timedelta(days=4)).strftime('%Y-%m-%d')},
        {"title": "Padres announce roster moves ahead of road trip", "sentiment": 0.08, "date": (datetime.now() - timedelta(days=4)).strftime('%Y-%m-%d')},
        {"title": "Fans excited for upcoming homestand promotions", "sentiment": 0.48, "date": (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')},
    ]
    
    headlines_df = pd.DataFrame(sample_headlines)
    # Ensure date is string format, no timestamp confusion
    headlines_df['date'] = headlines_df['date'].astype(str)
    headlines_df.to_csv('padres_recent_headlines.csv', index=False)
    
    return df

# Function to save new sentiment data
def save_sentiment_data(sentiment_dict):
    """Save new sentiment data point to history"""
    
    df = load_historical_data()
    
    # Create new row
    new_row = {
        'Date': datetime.now().strftime('%Y-%m-%d'),
        'Team_Overall': sentiment_dict.get('team', 0),
        'Players_Machado': sentiment_dict.get('machado', 0),
        'Players_Tatis': sentiment_dict.get('tatis', 0),
        'Players_King': sentiment_dict.get('king', 0),
        'GM_Front_Office': sentiment_dict.get('gm', 0),
        'Fan_Morale': sentiment_dict.get('fans', 0),
        'Stadium_Experience': sentiment_dict.get('stadium', 0)
    }
    
    # Check if we already have data for today
    today = datetime.now().strftime('%Y-%m-%d')
    if not df.empty and today in df['Date'].astype(str).values:
        # Update today's row
        df.loc[df['Date'].astype(str) == today, list(new_row.keys())] = list(new_row.values())
    else:
        # Append new row
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save to file
    df.to_csv(DATA_FILE, index=False)
    return df

# Helper functions
def get_sentiment_color(value):
    if value > 0.3:
        return 'green'
    elif value < -0.1:
        return 'red'
    else:
        return 'orange'

def get_sentiment_label(value):
    if value > 0.3:
        return 'POSITIVE'
    elif value < -0.1:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'

# Main app
st.title("⚾ San Diego Padres - Sentiment Tracker Dashboard")
st.markdown("### Real-time sentiment analysis from sports news headlines")

# Professional info banner
st.markdown("""
<div style='background: linear-gradient(135deg, #2F241D 0%, #1a1612 100%); 
            padding: 1.5rem; 
            border-radius: 10px; 
            margin: 1.5rem 0 2rem 0;
            border-left: 5px solid #FFC425;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);'>
    <p style='color: white; 
              margin: 0; 
              font-size: 0.95rem;
              line-height: 1.6;
              font-family: Inter;'>
        <strong style='color: #FFC425;'>🎯 About:</strong> 
        This dashboard tracks public sentiment about the San Diego Padres by analyzing headlines 
        from major sports news outlets. Sentiment scores range from <strong style='color: #ff6b6b;'>-1.0 (negative)</strong> 
        to <strong style='color: #51cf66;'>+1.0 (positive)</strong>, with 0.0 being neutral.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Dashboard Controls")

# Update data button
if st.sidebar.button("🔄 Update with Latest News", type="primary"):
    with st.spinner("Fetching and analyzing news..."):
        current_sentiment = process_news_sentiment()
        if current_sentiment:
            df = save_sentiment_data(current_sentiment)
            st.sidebar.success("✅ Data updated successfully!")
            st.rerun()
        else:
            st.sidebar.error("❌ Could not update data")

# Backfill historical data button
if st.sidebar.button("📅 Backfill Last 30 Days", help="Fetch historical news to populate past 30 days"):
    with st.spinner("Analyzing historical news... This may take 1-2 minutes."):
        success = backfill_historical_data(days=30)
        if success:
            st.rerun()

# Demo data button
st.sidebar.markdown("---")
st.sidebar.markdown("**Demo Mode:**")
if st.sidebar.button("🎨 Generate Sample Data", help="Populate dashboard with 30 days of realistic sample data for demo"):
    generate_sample_data_demo()
    st.sidebar.success("✅ Sample data generated!")
    st.rerun()

st.sidebar.markdown("---")

# Load current data
df = load_historical_data()

if df.empty:
    st.info("⚠️ **No data yet!** You have three options:")
    st.markdown("""
    1. **🎨 Generate Sample Data** - Click the button in the sidebar to see the dashboard with 30 days of demo data
    2. **🔄 Update with Latest News** - Fetch real Padres news and start tracking actual sentiment
    3. **📅 Backfill Last 30 Days** - Pull historical news (if available) to populate past data
    """)
    st.stop()

# Date range selector (only show if we have enough data)
max_days = min(90, len(df))
if max_days >= 7:
    date_range = st.sidebar.slider(
        "Select Date Range (Days)",
        min_value=7,
        max_value=max_days,
        value=max_days,
        step=7
    )
else:
    date_range = max_days
    st.sidebar.info(f"Showing all {max_days} day(s) of data. Collect more data to use the date range slider.")

# Filter data
df_filtered = df.tail(date_range)

# Category selector
categories_to_show = st.sidebar.multiselect(
    "Select Categories to Display",
    ['Team Overall', 'Players', 'GM/Front Office', 'Fan Morale', 'Stadium Experience'],
    default=['Team Overall', 'Players', 'GM/Front Office', 'Fan Morale', 'Stadium Experience']
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Last Updated:** {df['Date'].iloc[-1].strftime('%Y-%m-%d') if not df.empty else 'Never'}")
st.sidebar.markdown(f"**Total Data Points:** {len(df)}")

# Top metrics
st.markdown("## Current Sentiment Snapshot")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    current_team = df['Team_Overall'].iloc[-1]
    delta_team = (df['Team_Overall'].iloc[-1] - df['Team_Overall'].iloc[-min(8, len(df))]) if len(df) >= 8 else 0
    st.metric(
        label="Team Overall",
        value=f"{current_team:.2f}",
        delta=f"{delta_team:.2f} (7d)" if len(df) >= 8 else "N/A"
    )
    st.markdown(f"**Status:** {get_sentiment_label(current_team)}")

with col2:
    current_machado = df['Players_Machado'].iloc[-1]
    delta_machado = (df['Players_Machado'].iloc[-1] - df['Players_Machado'].iloc[-min(8, len(df))]) if len(df) >= 8 else 0
    st.metric(
        label="Machado",
        value=f"{current_machado:.2f}",
        delta=f"{delta_machado:.2f} (7d)" if len(df) >= 8 else "N/A"
    )
    st.markdown(f"**Status:** {get_sentiment_label(current_machado)}")

with col3:
    current_gm = df['GM_Front_Office'].iloc[-1]
    delta_gm = (df['GM_Front_Office'].iloc[-1] - df['GM_Front_Office'].iloc[-min(8, len(df))]) if len(df) >= 8 else 0
    st.metric(
        label="GM/Front Office",
        value=f"{current_gm:.2f}",
        delta=f"{delta_gm:.2f} (7d)" if len(df) >= 8 else "N/A"
    )
    st.markdown(f"**Status:** {get_sentiment_label(current_gm)}")

with col4:
    current_fans = df['Fan_Morale'].iloc[-1]
    delta_fans = (df['Fan_Morale'].iloc[-1] - df['Fan_Morale'].iloc[-min(8, len(df))]) if len(df) >= 8 else 0
    st.metric(
        label="Fan Morale",
        value=f"{current_fans:.2f}",
        delta=f"{delta_fans:.2f} (7d)" if len(df) >= 8 else "N/A"
    )
    st.markdown(f"**Status:** {get_sentiment_label(current_fans)}")

with col5:
    current_stadium = df['Stadium_Experience'].iloc[-1]
    delta_stadium = (df['Stadium_Experience'].iloc[-1] - df['Stadium_Experience'].iloc[-min(8, len(df))]) if len(df) >= 8 else 0
    st.metric(
        label="Stadium Experience",
        value=f"{current_stadium:.2f}",
        delta=f"{delta_stadium:.2f} (7d)" if len(df) >= 8 else "N/A"
    )
    st.markdown(f"**Status:** {get_sentiment_label(current_stadium)}")

st.markdown("---")

# Top Headlines Section
st.markdown("## 📰 Recent Padres Headlines")

# Load recent headlines if they exist
if os.path.exists('padres_recent_headlines.csv'):
    try:
        headlines_df = pd.read_csv('padres_recent_headlines.csv')
        
        # BULLETPROOF date parsing - try multiple methods
        try:
            # Try simple date format first
            headlines_df['date'] = pd.to_datetime(headlines_df['date'], format='%Y-%m-%d')
        except:
            try:
                # Try letting pandas figure it out
                headlines_df['date'] = pd.to_datetime(headlines_df['date'], errors='coerce')
            except:
                # Last resort: just use string dates
                headlines_df['date'] = headlines_df['date'].astype(str)
        
        # Remove any rows with invalid dates
        headlines_df = headlines_df.dropna(subset=['date'])
        
        # Sort and get top 10
        try:
            headlines_df = headlines_df.sort_values('date', ascending=False).head(10)
        except:
            # If sorting fails, just take first 10
            headlines_df = headlines_df.head(10)
            
    except Exception as e:
        st.warning(f"⚠️ Could not load headlines properly. Refresh data to fix. (Error: {str(e)})")
        headlines_df = pd.DataFrame()  # Empty dataframe if error
    
    if not headlines_df.empty:
        # Display headlines in a nice card format
        for idx, row in headlines_df.iterrows():
            sentiment = row['sentiment']
            
            # Determine sentiment color and label
            if sentiment > 0.3:
                sentiment_color = "#51cf66"
                sentiment_emoji = "📈"
                sentiment_text = "Positive"
            elif sentiment < -0.1:
                sentiment_color = "#ff6b6b"
                sentiment_emoji = "📉"
                sentiment_text = "Negative"
            else:
                sentiment_color = "#ffd43b"
                sentiment_emoji = "➡️"
                sentiment_text = "Neutral"
            
            # Create headline card
            st.markdown(f"""
            <div style='background: white; 
                        padding: 1rem 1.5rem; 
                        border-radius: 8px; 
                        margin-bottom: 0.8rem;
                        border-left: 4px solid {sentiment_color};
                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div style='flex: 1;'>
                        <p style='color: #2F241D; 
                                  margin: 0; 
                                  font-size: 0.95rem; 
                                  line-height: 1.5;
                                  font-weight: 500;'>
                            {row['title']}
                        </p>
                        <p style='color: #666; 
                                  margin: 0.3rem 0 0 0; 
                                  font-size: 0.8rem;'>
                            {row['date'].strftime('%B %d, %Y') if hasattr(row['date'], 'strftime') else str(row['date'])}
                        </p>
                    </div>
                    <div style='margin-left: 1rem; 
                                text-align: center; 
                                min-width: 80px;'>
                        <div style='font-size: 1.5rem;'>{sentiment_emoji}</div>
                        <div style='color: {sentiment_color}; 
                                    font-weight: 600; 
                                    font-size: 0.75rem;'>
                            {sentiment_text}
                        </div>
                        <div style='color: #666; 
                                    font-size: 0.7rem;'>
                            {sentiment:.2f}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📰 No valid headlines available. Click 'Update with Latest News' to fetch fresh headlines.")
else:
    st.info("📰 No recent headlines available. Click 'Update with Latest News' to fetch headlines.")

st.markdown("---")

# Charts section
st.markdown("## Sentiment Trends Over Time")

# Team Overall Chart
if 'Team Overall' in categories_to_show and len(df_filtered) > 0:
    st.subheader("📊 Team Overall Sentiment")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df_filtered['Date'],
        y=df_filtered['Team_Overall'],
        mode='lines+markers',
        name='Team Sentiment',
        line=dict(color='#2F241D', width=3),
        marker=dict(size=8, color='#FFC425', line=dict(color='#2F241D', width=2)),
        fill='tozeroy',
        fillcolor='rgba(47, 36, 29, 0.1)',
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Sentiment: %{y:.3f}<extra></extra>'
    ))
    fig1.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.3)", line_width=1)
    fig1.update_layout(
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        yaxis=dict(
            range=[-1, 1], 
            title="Sentiment Score",
            gridcolor='rgba(0,0,0,0.05)',
            title_font=dict(size=14, color='#2F241D', family='Inter')
        ),
        xaxis=dict(
            title="Date",
            gridcolor='rgba(0,0,0,0.05)',
            title_font=dict(size=14, color='#2F241D', family='Inter')
        ),
        hovermode='x unified',
        font=dict(family='Inter', color='#2F241D'),
        margin=dict(l=60, r=20, t=40, b=60)
    )
    st.plotly_chart(fig1, use_container_width=True)

# Players Comparison Chart
if 'Players' in categories_to_show and len(df_filtered) > 0:
    st.subheader("👥 Individual Players Sentiment")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df_filtered['Date'],
        y=df_filtered['Players_Machado'],
        mode='lines+markers',
        name='Manny Machado',
        line=dict(color='#2F241D', width=3),
        marker=dict(size=7, color='#2F241D'),
        hovertemplate='<b>Machado</b><br>%{x|%b %d}<br>Sentiment: %{y:.3f}<extra></extra>'
    ))
    fig2.add_trace(go.Scatter(
        x=df_filtered['Date'],
        y=df_filtered['Players_Tatis'],
        mode='lines+markers',
        name='Fernando Tatis Jr.',
        line=dict(color='#8B4513', width=3),
        marker=dict(size=7, color='#8B4513'),
        hovertemplate='<b>Tatis Jr.</b><br>%{x|%b %d}<br>Sentiment: %{y:.3f}<extra></extra>'
    ))
    fig2.add_trace(go.Scatter(
        x=df_filtered['Date'],
        y=df_filtered['Players_King'],
        mode='lines+markers',
        name='Michael King',
        line=dict(color='#FFC425', width=3),
        marker=dict(size=7, color='#FFC425', line=dict(color='#2F241D', width=1)),
        hovertemplate='<b>King</b><br>%{x|%b %d}<br>Sentiment: %{y:.3f}<extra></extra>'
    ))
    fig2.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.3)", line_width=1)
    fig2.update_layout(
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        yaxis=dict(
            range=[-1, 1], 
            title="Sentiment Score",
            gridcolor='rgba(0,0,0,0.05)',
            title_font=dict(size=14, color='#2F241D', family='Inter')
        ),
        xaxis=dict(
            title="Date",
            gridcolor='rgba(0,0,0,0.05)',
            title_font=dict(size=14, color='#2F241D', family='Inter')
        ),
        hovermode='x unified',
        font=dict(family='Inter', color='#2F241D'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#2F241D',
            borderwidth=1
        ),
        margin=dict(l=60, r=20, t=80, b=60)
    )
    st.plotly_chart(fig2, use_container_width=True)

# Two column layout for remaining charts
col_left, col_right = st.columns(2)

with col_left:
    if 'GM/Front Office' in categories_to_show and len(df_filtered) > 0:
        st.subheader("🏢 GM/Front Office Sentiment")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df_filtered['Date'],
            y=df_filtered['GM_Front_Office'],
            mode='lines+markers',
            name='GM/Front Office',
            line=dict(color='#2F241D', width=3),
            marker=dict(size=7, color='#8B4513'),
            fill='tozeroy',
            fillcolor='rgba(47, 36, 29, 0.1)',
            hovertemplate='%{x|%b %d}<br>Sentiment: %{y:.3f}<extra></extra>'
        ))
        fig3.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.3)", line_width=1)
        fig3.update_layout(
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white',
            yaxis=dict(
                range=[-1, 1], 
                title="Sentiment Score",
                gridcolor='rgba(0,0,0,0.05)',
                title_font=dict(size=12, color='#2F241D', family='Inter')
            ),
            xaxis=dict(
                title="Date",
                gridcolor='rgba(0,0,0,0.05)',
                title_font=dict(size=12, color='#2F241D', family='Inter')
            ),
            hovermode='x unified',
            font=dict(family='Inter', color='#2F241D'),
            margin=dict(l=50, r=20, t=30, b=50)
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    if 'Fan Morale' in categories_to_show and len(df_filtered) > 0:
        st.subheader("🎉 Fan Morale")
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=df_filtered['Date'],
            y=df_filtered['Fan_Morale'],
            mode='lines+markers',
            name='Fan Morale',
            line=dict(color='#FFC425', width=3),
            marker=dict(size=7, color='#FFC425', line=dict(color='#2F241D', width=1)),
            fill='tozeroy',
            fillcolor='rgba(255, 196, 37, 0.15)',
            hovertemplate='%{x|%b %d}<br>Sentiment: %{y:.3f}<extra></extra>'
        ))
        fig4.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.3)", line_width=1)
        fig4.update_layout(
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white',
            yaxis=dict(
                range=[-1, 1], 
                title="Sentiment Score",
                gridcolor='rgba(0,0,0,0.05)',
                title_font=dict(size=12, color='#2F241D', family='Inter')
            ),
            xaxis=dict(
                title="Date",
                gridcolor='rgba(0,0,0,0.05)',
                title_font=dict(size=12, color='#2F241D', family='Inter')
            ),
            hovermode='x unified',
            font=dict(family='Inter', color='#2F241D'),
            margin=dict(l=50, r=20, t=30, b=50)
        )
        st.plotly_chart(fig4, use_container_width=True)

with col_right:
    if 'Stadium Experience' in categories_to_show and len(df_filtered) > 0:
        st.subheader("🏟️ Stadium Experience")
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=df_filtered['Date'],
            y=df_filtered['Stadium_Experience'],
            mode='lines+markers',
            name='Stadium Experience',
            line=dict(color='#8B4513', width=3),
            marker=dict(size=7, color='#8B4513'),
            fill='tozeroy',
            fillcolor='rgba(139, 69, 19, 0.1)',
            hovertemplate='%{x|%b %d}<br>Sentiment: %{y:.3f}<extra></extra>'
        ))
        fig5.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.3)", line_width=1)
        fig5.update_layout(
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white',
            yaxis=dict(
                range=[-1, 1], 
                title="Sentiment Score",
                gridcolor='rgba(0,0,0,0.05)',
                title_font=dict(size=12, color='#2F241D', family='Inter')
            ),
            xaxis=dict(
                title="Date",
                gridcolor='rgba(0,0,0,0.05)',
                title_font=dict(size=12, color='#2F241D', family='Inter')
            ),
            hovermode='x unified',
            font=dict(family='Inter', color='#2F241D'),
            margin=dict(l=50, r=20, t=30, b=50)
        )
        st.plotly_chart(fig5, use_container_width=True)
    
    # Summary bar chart
    st.subheader("📈 Current Comparison")
    categories = ['Team', 'Machado', 'Tatis', 'King', 'GM/FO', 'Fans', 'Stadium']
    values = [
        df['Team_Overall'].iloc[-1],
        df['Players_Machado'].iloc[-1],
        df['Players_Tatis'].iloc[-1],
        df['Players_King'].iloc[-1],
        df['GM_Front_Office'].iloc[-1],
        df['Fan_Morale'].iloc[-1],
        df['Stadium_Experience'].iloc[-1]
    ]
    
    # Use Padres colors for bars
    bar_colors = []
    for v in values:
        if v > 0.3:
            bar_colors.append('#FFC425')  # Gold for positive
        elif v < -0.1:
            bar_colors.append('#8B4513')  # Brown for negative
        else:
            bar_colors.append('#A0AAB2')  # Sand for neutral
    
    fig6 = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker=dict(
                color=bar_colors,
                line=dict(color='#2F241D', width=2)
            ),
            text=[f'{v:.2f}' for v in values],
            textposition='outside',
            textfont=dict(size=13, color='#2F241D', family='Inter', weight='bold'),
            hovertemplate='<b>%{x}</b><br>Sentiment: %{y:.3f}<extra></extra>'
        )
    ])
    fig6.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.3)", line_width=1)
    fig6.update_layout(
        height=350,
        plot_bgcolor='white',
        paper_bgcolor='white',
        yaxis=dict(
            range=[-1, 1], 
            title="Sentiment Score",
            gridcolor='rgba(0,0,0,0.05)',
            title_font=dict(size=12, color='#2F241D', family='Inter')
        ),
        xaxis=dict(
            title="Category",
            title_font=dict(size=12, color='#2F241D', family='Inter')
        ),
        showlegend=False,
        font=dict(family='Inter', color='#2F241D'),
        margin=dict(l=50, r=20, t=30, b=50)
    )
    st.plotly_chart(fig6, use_container_width=True)

# Data table
st.markdown("---")
with st.expander("📋 View Raw Data"):
    st.dataframe(df_filtered, use_container_width=True)
    
    csv = df_filtered.to_csv(index=False)
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name=f"padres_sentiment_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='background: white; 
            padding: 1.5rem; 
            border-radius: 10px; 
            border-left: 4px solid #FFC425;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-top: 2rem;'>
    <h4 style='color: #2F241D; margin-top: 0;'>📊 About This Dashboard</h4>
    <ul style='color: #2F241D; line-height: 1.8;'>
        <li><strong>Sentiment Range:</strong> -1.0 (Very Negative) to +1.0 (Very Positive), with 0.0 being neutral</li>
        <li><strong>Data Sources:</strong> MLB.com and ESPN news headlines</li>
        <li><strong>Analysis Method:</strong> TextBlob sentiment analysis engine</li>
        <li><strong>Update Frequency:</strong> Click "Update with Latest News" daily for fresh data</li>
        <li><strong>Demo Mode:</strong> Use "Generate Sample Data" to see the dashboard with realistic demo data</li>
    </ul>
    <p style='color: #666; font-size: 0.9rem; margin-bottom: 0;'>
        <em>💡 Tip: Build up 2+ weeks of data for meaningful trend analysis</em>
    </p>
</div>
""", unsafe_allow_html=True)
