import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(
    page_title="Nifty 500 Advanced Screener (TradingView Style)",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Nifty 500 Advanced Technical Screener")
st.markdown("Screening live Nifty 500 stocks with **Price > 20**, **10D Avg Vol > 100K**, **Close within 0-20% of All-Time High**, **Price > 50 SMA**, alongside **Tech, MA, and Oscillator Ratings**.")

@st.cache_data(ttl=86400)
def get_nifty500_tickers():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            if 'Symbol' in df.columns:
                return [str(symbol).strip() + ".NS" for symbol in df['Symbol'].dropna().tolist()]
    except Exception as e:
        st.warning(f"Could not load official NSE list: {e}. Using fallback sample tickers.")
    
    return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
            "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LT.NS"]

def calculate_ratings(df):
    """Computes MA rating, Oscillator rating, and Combined Tech Rating similar to TradingView logic"""
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # 1. Moving Averages Calculation
    sma_10 = close.rolling(10).mean().iloc[-1]
    sma_20 = close.rolling(20).mean().iloc[-1]
    sma_50 = close.rolling(50).mean().iloc[-1]
    sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else sma_50
    curr_close = close.iloc[-1]
    
    ma_buy_count = sum([curr_close > sma_10, curr_close > sma_20, curr_close > sma_50, curr_close > sma_200])
    
    if ma_buy_count >= 3:
        ma_rating = "Strong Buy"
    elif ma_buy_count == 2:
        ma_rating = "Buy"
    elif ma_buy_count == 1:
        ma_rating = "Neutral"
    else:
        id_val = "Sell" if ma_buy_count == 0 else "Strong Sell"
        ma_rating = id_val

    # 2. Oscillators Calculation (RSI & Stochastic Proxy)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    curr_rsi = rsi.iloc[-1]
    
    # Stochastic %K (14)
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    stoch_k = 100 * (close - low_14) / (high_14 - low_14)
    curr_stoch = stoch_k.iloc[-1]
    
    os_signals = 0
    if 50 < curr_rsi < 70: os_signals += 1
    elif curr_rsi >= 70: os_signals += 2
    elif curr_rsi < 30: os_signals -= 1
    
    if 20 < curr_stoch < 80: os_signals += 1
    elif curr_stoch >= 80: os_signals += 2
    
    if os_signals >= 2:
        os_rating = "Strong Buy"
    elif os_signals == 1:
        os_rating = "Buy"
    elif os_signals == 0:
        os_rating = "Neutral"
    else:
        os_rating = "Sell"

    # 3. Combined Tech Rating
    total_score = ma_buy_count + os_signals
    if total_score >= 5:
        tech_rating = "Strong Buy"
    elif total_score >= 3:
        tech_rating = "Buy"
    elif total_score >= 1:
        tech_rating = "Neutral"
    else:
        tech_rating = "Sell"
        
    return tech_rating, ma_rating, os_rating, round(curr_rsi, 2)

@st.cache_data(ttl=3600)
def fetch_and_screen_stocks(tickers):
    matched_stocks = []
    end_date = datetime.today()
    start_date = end_date - timedelta(days=500) # Deep window to capture all-time high & 200 SMA
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(tickers)
    
    for i, ticker in enumerate(tickers):
        status_text.text(f"Scanning ({i+1}/{total}): {ticker.replace('.NS', '')}")
        progress_bar.progress((i + 1) / total)
        
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty or len(df) < 100:
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            close = df['Close']
            volume = df['Volume']
            high = df['High']
            
            curr_close = close.iloc[-1]

            # Filter 1: Price >= 20 INR
            if curr_close < 20:
                continue

            # Filter 2: Avg vol, 10D > 100K
            vol_sma_10 = volume.rolling(window=10).mean().iloc[-1]
            if vol_sma_10 <= 100000:
                continue

            # Filter 3: High, All Time above Price by 0% to 20%
            # (Close >= ATH * 0.80 and Close <= ATH)
            all_time_high = high.max()
            if curr_close < (all_time_high * 0.80) or curr_close > all_time_high:
                continue

            # Filter 4: Price > SMA, 50
            sma_50 = close.rolling(window=50).mean().iloc[-1]
            if curr_close <= sma_50:
                continue

            # Calculate technical ratings
            tech_rating, ma_rating, os_rating, rsi_val = calculate_ratings(df)

            matched_stocks.append({
                "Ticker": ticker.replace(".NS", ""),
                "Close Price": round(float(curr_close), 2),
                "Tech Rating": tech_rating,
                "MA Rating": ma_rating,
                "Os Rating": os_rating,
                "RSI (14)": rsi_val,
                "10D Vol SMA": int(vol_sma_10),
                "All-Time High": round(float(all_time_high), 2),
                "% of ATH": round(float((curr_close / all_time_high) * 100), 2),
                "50D SMA": round(float(sma_50), 2)
            })
        except Exception:
            continue

    status_text.empty()
    progress_bar.empty()
    return pd.DataFrame(matched_stocks)

# Sidebar UI
st.sidebar.header("Screener Settings")
universe_choice = st.sidebar.selectbox(
    "Stock Universe", 
    ["Official Live Nifty 500 (NSE)", "Quick Sample Test (Top 30)"]
)

run_button = st.sidebar.button("Run Screener", type="primary")

if run_button:
    with st.spinner("Downloading data, computing ATH boundaries, moving averages, and technical ratings..."):
        tickers_list = get_nifty500_tickers() if "Nifty 500" in universe_choice else [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
            "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LT.NS",
            "AXISPAINT.NS", "SUNPHARMA.NS", "TITAN.NS", "BAJFINANCE.NS"
        ]
        
        results_df = fetch_and_screen_stocks(tickers_list)
        
    if not results_df.empty:
        st.success(f"Scan complete! Found {len(results_df)} stocks matching your exact criteria.")
        
        # Display DataFrame with color highlight for ratings if desired
        st.dataframe(results_df, use_container_width=True)
        
        # CSV Export
        csv_export = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Screened Results (CSV)",
            data=csv_export,
            file_name=f"nifty500_technical_screener_{datetime.today().strftime('%Y-%m-%d')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No stocks matched all specified criteria concurrently.")
else:
    st.info("Click **Run Screener** in the sidebar to execute the live market scan.")
