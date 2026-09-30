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
    page_title="Nifty 500 Pro Screener with Telegram Alerts",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Nifty 500 Pro Screener: RS, EMAs & Telegram Alerts")
st.markdown("Screening with **Price > 20**, **10D Vol > 100K**, **0-20% of ATH**, **50 SMA**, **9/20 EMA Confluence**, **RS vs Nifty 50**, and automated **Telegram Alerts** for **Strong Buy** Tech Ratings.")

# Telegram Alert Function
def send_telegram_alert(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload, timeout=5)
        return response.status_code == 200
    except Exception:
        return False

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
    
    return [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
        "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LT.NS"
    ]

@st.cache_data(ttl=3600)
def get_nifty50_benchmark(start_date, end_date):
    try:
        nifty_df = yf.download("^NSEI", start=start_date, end=end_date, progress=False)
        if isinstance(nifty_df.columns, pd.MultiIndex):
            nifty_df.columns = nifty_df.columns.get_level_values(0)
        return nifty_df['Close']
    except Exception:
        return None

def calculate_ratings_and_emas(df):
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    ema_9 = close.ewm(span=9, adjust=False).mean().iloc[-1]
    ema_20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    sma_50 = close.rolling(50).mean().iloc[-1]
    sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else sma_50
    curr_close = close.iloc[-1]
    
    ma_buy_count = sum([curr_close > ema_9, curr_close > ema_20, curr_close > sma_50, curr_close > sma_200])
    
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
    rs_val = gain / loss
    rsi = 100 - (100 / (1 + rs_val))
    curr_rsi = rsi.iloc[-1]
    
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    stoch_k = 100 * (close - low_14) / (high_14 - low_14)
    curr_stoch = stoch_k.iloc[-1]
    
    os_signals = 0
    if 50 < curr_rsi < 70: 
        os_signals += 1
    elif curr_rsi >= 70: 
        os_signals += 2
    elif curr_rsi < 30: 
        os_signals -= 1
    
    if 20 < curr_stoch < 80: 
        os_signals += 1
    elif curr_stoch >= 80: 
        os_signals += 2
    
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
        
    return tech_rating, ma_rating, os_rating, round(curr_rsi, 2), round(ema_9, 2), round(ema_20, 2)

@st.cache_data(ttl=3600)
def fetch_and_screen_stocks(tickers):
    matched_stocks = []
    end_date = datetime.today()
    start_date = end_date - timedelta(days=500)
    
    nifty_close = get_nifty50_benchmark(start_date, end_date)
    nifty_return_3m = 0.0
    if nifty_close is not None and len(nifty_close) >= 60:
        nifty_return_3m = (nifty_close.iloc[-1] - nifty_close.iloc[-60]) / nifty_close.iloc[-60]

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

            tech_rating, ma_rating, os_rating, rsi_val, ema_9, ema_20 = calculate_ratings_and_emas(df)

            stock_return_3m = 0.0
            rs_vs_nifty = 0.0
            if len(close) >= 60:
                stock_return_3m = (close.iloc[-1] - close.iloc[-60]) / close.iloc[-60]
                rs_vs_nifty = (stock_return_3m - nifty_return_3m) * 100

            matched_stocks.append({
                "Ticker": ticker.replace(".NS", ""),
                "Close Price": round(float(curr_close), 2),
                "RS vs Nifty (%)": round(float(rs_vs_nifty), 2),
                "Tech Rating": tech_rating,
                "MA Rating": ma_rating,
                "Os Rating": os_rating,
                "RSI (14)": rsi_val,
                "9 EMA": ema_9,
                "20 EMA": ema_20,
                "50 SMA": round(float(sma_50), 2),
                "10D Vol SMA": int(vol_sma_10),
                "% of ATH": round(float((curr_close / all_time_high) * 100), 2)
            })
        except Exception:
            continue

    status_text.empty()
    progress_bar.empty()
    
    res_df = pd.DataFrame(matched_stocks)
    
    if not res_df.empty:
        sectors = []
        for ticker in res_df['Ticker']:
            try:
                info = yf.Ticker(f"{ticker}.NS").info
                sectors.append(info.get('sector', 'N/A'))
            except:
                sectors.append('N/A')
        res_df.insert(1, "Sector", sectors)
        res_df = res_df.sort_values(by="RS vs Nifty (%)", ascending=False).reset_index(drop=True)
        
    return res_df

# Sidebar UI
st.sidebar.header("Screener Settings")
universe_choice = st.sidebar.selectbox(
    "Stock Universe", 
    ["Official Live Nifty 500 (NSE)", "Quick Sample Test (Top 30)"]
)

st.sidebar.markdown("---")
st.sidebar.header("📱 Telegram Notification Settings")
enable_telegram = st.sidebar.checkbox("Enable Telegram Alerts", value=False)
bot_token_input = st.sidebar.text_input("Bot Token", type="password", help="Enter your Telegram Bot API Token")
chat_id_input = st.sidebar.text_input("Chat ID", help="Enter your Telegram Chat ID or Channel ID")

run_button = st.sidebar.button("Run Pro Scan & Alert", type="primary")

if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

if run_button:
    with st.spinner("Running live scan, evaluating ratings, and checking alerts..."):
        if "Nifty 500" in universe_choice:
            tickers_list = get_nifty500_tickers()
        else:
            tickers_list = [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
                "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LT.NS"
            ]
        st.session_state.results_df = fetch_and_screen_stocks(tickers_list)
        
        # Filter and send Telegram alerts for "Strong Buy" Tech Rating only
        if enable_telegram and bot_token_input and chat_id_input and not st.session_state.results_df.empty:
            strong_buys = st.session_state.results_df[st.session_state.results_df['Tech Rating'] == 'Strong Buy']
            
            if not strong_buys.empty:
                alert_count = 0
                for _, row in strong_buys.iterrows():
                    msg = (
                        f"🚨 *STRONG BUY BREAKOUT ALERT* 🚨\n\n"
                        f"📌 *Ticker:* `{row['Ticker']}`\n"
                        f"🏭 *Sector:* {row['Sector']}\n"
                        f"💰 *Price:* ₹{row['Close Price']}\n"
                        f"⚡ *RS vs Nifty:* `+{row['RS vs Nifty (%)']}%`\n"
                        f"📈 *Tech Rating:* `Strong Buy`\n"
                        f"📊 *RSI (14):* {row['RSI (14)']}\n"
                        f"🔗 [View Chart](https://www.tradingview.com/chart/?symbol=NSE:{row['Ticker']})"
                    )
                    success = send_telegram_alert(bot_token_input, chat_id_input, msg)
                    if success:
                        alert_count += 1
                st.sidebar.success(f"Successfully sent {alert_count} Telegram alerts for Strong Buy stocks!")
            else:
                st.sidebar.info("Scan completed, but no stocks matched the 'Strong Buy' Tech Rating for alerts.")

if not st.session_state.results_df.empty:
    df_res = st.session_state.results_df
    st.success(f"Scan complete! Showing all {len(df_res)} matching stocks in Streamlit.")
    
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
    st.markdown("### 📊 Interactive Daily Chart & Moving Averages Viewer")
    
    selected_ticker = st.selectbox("Select stock for technical inspection:", df_res['Ticker'].tolist())
    
    if selected_ticker:
        with st.spinner(f"Loading chart for {selected_ticker}..."):
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
            
            chart_df['EMA_9'] = chart_df['Close'].ewm(span=9, adjust=False).mean()
            chart_df['EMA_20'] = chart_df['Close'].ewm(span=20, adjust=False).mean()
            chart_df['SMA_50'] = chart_df['Close'].rolling(50).mean()
            
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA_9'], line=dict(color='cyan', width=1), name='9 EMA'))
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA_20'], line=dict(color='magenta', width=1), name='20 EMA'))
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['SMA_50'], line=dict(color='orange', width=1.5), name='50 SMA'))
            
            fig.update_layout(
                title=f"{selected_ticker} - Daily Price with 9/20 EMA & 50 SMA Confluence",
                yaxis_title="Price (INR)",
                xaxis_rangeslider_visible=False,
                height=550,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            tv_url = f"https://www.tradingview.com/chart/?symbol=NSE:{selected_ticker}"
            st.markdown(f"🔗 [Open {selected_ticker} on TradingView in a new tab]({tv_url})", unsafe_allow_html=True)

else:
    st.info("Configure your Telegram settings in the sidebar (optional) and click **Run Pro Scan & Alert**.")
