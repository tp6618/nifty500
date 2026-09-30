import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="Nifty 500 Advanced Screener & Charts",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Nifty 500 Technical Screener & Interactive Charts")
st.markdown("Screening live Nifty 500 stocks with **Price > 20**, **10D Avg Vol > 100K**, **Close within 0-20% of All-Time High**, and **Price > 50 SMA**. Click any stock to view its 1-day chart.")

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
    close = df['Close']
    high = df['High']
    low = df['Low']
    
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
        ma_rating = "Sell"

    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    curr_rsi = rsi.iloc[-1]
    
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
    start_date = end_date - timedelta(days=500)
    
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

            if curr_close < 20:
                continue

            vol_sma_10 = volume.rolling(window=10).mean().iloc[-1]
            if vol_sma_10 <= 100000:
                continue

            all_time_high = high.max()
            if curr_close < (all_time_high * 0.80) or curr_close > all_time_high:
                continue

            sma_50 = close.rolling(window=50).mean().iloc[-1]
            if curr_close <= sma_50:
                continue

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

# Session state to store scan results
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

if run_button:
    with st.spinner("Scanning market and computing metrics..."):
        tickers_list = get_nifty500_tickers() if "Nifty 500" in universe_choice else [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
            "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LT.NS"
        ]
        st.session_state.results_df = fetch_and_screen_stocks(tickers_list)

if not st.session_state.results_df.empty:
    df_res = st.session_state.results_df
    st.success(f"Found {len(df_res)} matching stocks.")
    
    # Interactive Table with clickable TradingView links
    st.markdown("### 📋 Screened Results (Click Ticker to open TradingView in a new tab)")
    
    # Add direct TradingView URL column
    display_df = df_res.copy()
    display_df['TradingView Chart'] = display_df['Ticker'].apply(
        lambda t: f"https://www.tradingview.com/chart/?symbol=NSE:{t}"
    )
    
    st.dataframe(
        display_df,
        column_config={
            "TradingView Chart": st.column_config.LinkColumn(
                "Open Chart (New Tab)", 
                help="Click to open full TradingView chart", 
                display_text="📈 View Chart"
            )
        },
        use_container_width=True
    )
    
    st.markdown("---")
    st.markdown("### 📊 Interactive 1-Day Candlestick Chart Viewer")
    
    selected_ticker = st.selectbox("Select a stock to inspect its 1-Day chart:", df_res['Ticker'].tolist())
    
    if selected_ticker:
        with st.spinner(f"Loading 1-day chart for {selected_ticker}..."):
            chart_df = yf.download(f"{selected_ticker}.NS", period="1y", interval="1d", progress=False)
            if isinstance(chart_df.columns, pd.MultiIndex):
                chart_df.columns = chart_df.columns.get_level_values(0)
                
            fig = go.Figure(data=[go.Candlestick(
                x=chart_df.index,
                open=chart_df['Open'],
                high=chart_df['High'],
                low=chart_df['Low'],
                close=chart_df['Close'],
                name=selected_ticker
            )])
            
            # Add 50 SMA
            chart_df['SMA_50'] = chart_df['Close'].rolling(50).mean()
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['SMA_50'], line=dict(color='orange', width=1.5), name='50 SMA'))
            
            fig.update_layout(
                title=f"{selected_ticker} - Daily Chart & 50 SMA",
                yaxis_title="Price (INR)",
                xaxis_rangeslider_visible=False,
                height=500,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Direct link button below chart
            tv_url = f"https://www.tradingview.com/chart/?symbol=NSE:{selected_ticker}"
            st.markdown(f"🔗 [Click here to open {selected_ticker} directly on TradingView in a new tab]({tv_url})", unsafe_allow_html=True)

else:
    st.info("Click **Run Screener** in the sidebar to execute the scan.")
