import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(
    page_title="Nifty 500 Dynamic Momentum Screener",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Live Nifty 500 Momentum & Trend Screener")
st.markdown("Dynamically pulls the official **Nifty 500** universe from NSE, fetches live price history via `yfinance`, and filters based on your criteria.")

@st.cache_data(ttl=86400) # Cache the Nifty 500 ticker list for 24 hours
def get_nifty500_tickers():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            if 'Symbol' in df.columns:
                tickers = [str(symbol).strip() + ".NS" for symbol in df['Symbol'].dropna().tolist()]
                return tickers
    except Exception as e:
        st.warning(f"Could not automatically fetch live NSE list directly: {e}. Falling back to sample symbols.")
    
    # Fallback default tickers if connection drops or NSE blocks direct request
    return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
            "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LT.NS"]

@st.cache_data(ttl=3600)
def fetch_and_screen_stocks(tickers):
    matched_stocks = []
    
    end_date = datetime.today()
    start_date = end_date - timedelta(days=400) # Ensure ~250 trading sessions
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(tickers)
    
    # Batch processing or loop with error handling
    for i, ticker in enumerate(tickers):
        status_text.text(f"Scanning ({i+1}/{total}): {ticker.replace('.NS', '')}")
        progress_bar.progress((i + 1) / total)
        
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty or len(df) < 250:
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            close = df['Close'].iloc[-1]
            volume = df['Volume']
            high = df['High']

            # Filter 1: Close > 20
            if close <= 20:
                continue

            # Filter 2: Volume (SMA, 10) > 100,000
            vol_sma_10 = volume.rolling(window=10).mean().iloc[-1]
            if vol_sma_10 <= 100000:
                continue

            # Filter 3: Close > Max(250, High) * 0.80
            max_250_high = high.iloc[-250:].max()
            if close <= (max_250_high * 0.80):
                continue

            # Filter 4: Close > SMA(Close, 50)
            sma_50 = df['Close'].rolling(window=50).mean().iloc[-1]
            if close <= sma_50:
                continue

            matched_stocks.append({
                "Ticker": ticker.replace(".NS", ""),
                "Close Price": round(float(close), 2),
                "10D Vol SMA": int(vol_sma_10),
                "250D High": round(float(max_250_high), 2),
                "50D SMA": round(float(sma_50), 2),
                "% of 250D High": round(float((close / max_250_high) * 100), 2)
            })
        except Exception:
            continue

    status_text.empty()
    progress_bar.empty()
    return pd.DataFrame(matched_stocks)

# Sidebar controls
st.sidebar.header("Screener Configuration")
universe_option = st.sidebar.selectbox(
    "Select Universe", 
    ["Official Live Nifty 500 (NSE)", "Quick Sample Test (Top 40)"]
)

run_scan = st.sidebar.button("Run Live Scan", type="primary")

if run_scan:
    with st.spinner("Downloading live Nifty 500 constituents and evaluating technical filters..."):
        if "Nifty 500" in universe_option:
            tickers_to_scan = get_nifty500_tickers()
        else:
            tickers_to_scan = [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
                "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LICI.NS",
                "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS", "HCLTECH.NS", "ASIANPAINT.NS",
                "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "BAJFINANCE.NS", "ULTRACEMCO.NS"
            ]
        
        st.info(f"Loaded {len(tickers_to_scan)} symbols into the processing queue.")
        results_df = fetch_and_screen_stocks(tickers_to_scan)
        
    if not results_df.empty:
        st.success(f"Scan complete! Found {len(results_df)} matching stocks.")
        st.dataframe(results_df, use_container_width=True)
        
        csv_data = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv_data,
            file_name=f"nifty500_momentum_screener_{datetime.today().strftime('%Y-%m-%d')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No stocks matched all criteria under the selected pool.")
else:
    st.info("Select your preference in the sidebar and click **Run Live Scan**.")
