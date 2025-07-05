import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="📈 Stock Intelligence Terminal", layout="wide")

st.markdown("# 📊 Stock Intelligence Terminal")
st.caption("Built for Quantitative Analysts | Real-time Market Data | ARIMA Forecasting")

# ---------------- Sidebar ----------------
market = st.sidebar.selectbox("Market", ["India (NSE)", "USA (NASDAQ)"])

@st.cache_data
def get_tickers_auto(market):
    if market == "India (NSE)":
        # NSE blocks web scraping, use hardcoded top Nifty stocks
        return sorted([
            "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
            "ITC.NS", "LT.NS", "SBIN.NS", "AXISBANK.NS", "BHARTIARTL.NS"
        ])
    else:
        url = "https://raw.githubusercontent.com/datasets/nasdaq-listings/master/data/nasdaq-listed-symbols.csv"
        df = pd.read_csv(url)
        return df['Symbol'].dropna().sort_values().tolist()

stock_list = get_tickers_auto(market)
ticker = st.sidebar.selectbox("Select Stock", stock_list)
period = st.sidebar.selectbox("Time Frame", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
chart_type = st.sidebar.selectbox("Chart Type", ["Candlestick", "Line", "Bar"])
show_ma = st.sidebar.checkbox("Show 20-Day MA", True)
use_forecast = st.sidebar.checkbox("🔮 Predict Future with ARIMA")
forecast_days = st.sidebar.slider("Days to Forecast", 1, 30, 7) if use_forecast else 0

# ---------------- Fetch Data ----------------
try:
    stock = yf.Ticker(ticker)
    info = stock.info
    data = stock.history(period=period)
    intraday = yf.download(ticker, period="1d", interval="1m")
    latest_price = intraday["Close"].iloc[-1]
    st.sidebar.metric("📍 Live Price (1m)", f"{latest_price:.2f}")
except Exception:
    st.sidebar.error("❌ Error fetching data")
    st.stop()

# Clean data
data = data.asfreq('B')
data["Close"].interpolate(method='linear', inplace=True)
data["MA20"] = data["Close"].rolling(20).mean()

# ---------------- Forecast ----------------
def forecast_arima(series, days=7):
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=days)
    last_date = series.index[-1]
    future_dates = [last_date + timedelta(days=i + 1) for i in range(days)]
    return pd.Series(forecast.values, index=future_dates)

forecast_series = None
if use_forecast:
    try:
        forecast_series = forecast_arima(data["Close"], forecast_days)
    except Exception as e:
        st.warning(f"Forecasting failed: {e}")

# ---------------- Layout Tabs ----------------
tab1, tab2, tab3 = st.tabs(["📈 Chart", "📊 Intraday", "📋 Financials"])

# Tab 1: Price Chart
with tab1:
    fig = go.Figure()
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=data.index, open=data['Open'], high=data['High'],
            low=data['Low'], close=data['Close'], name='Candlestick'))
    elif chart_type == "Line":
        fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close"))
    elif chart_type == "Bar":
        fig.add_trace(go.Bar(x=data.index, y=data["Close"], name="Bar"))

    if show_ma:
        fig.add_trace(go.Scatter(
            x=data.index, y=data["MA20"], mode="lines", name="MA20",
            line=dict(color='orange', dash='dash')))

    if forecast_series is not None:
        fig.add_trace(go.Scatter(
            x=forecast_series.index, y=forecast_series.values,
            mode="lines", name="ARIMA Forecast",
            line=dict(color='green', dash='dot')))

    fig.update_layout(
        hovermode="x unified",
        xaxis_rangeslider_visible=True,
        height=500,
        plot_bgcolor="#f4f4f4",
        paper_bgcolor="#f4f4f4"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(data.tail(10))

# Tab 2: Intraday Movement
with tab2:
    st.subheader("📅 Today's Intraday Price")
    st.line_chart(intraday["Close"])
    st.dataframe(intraday.tail())

# Tab 3: Financials
with tab3:
    def format_value(val):
        if isinstance(val, (int, float)):
            symbol = "₹" if ticker.endswith(".NS") else "$"
            return f"{symbol}{val:,.0f}" if abs(val) > 1 else f"{symbol}{val:.2f}"
        return "--"

    def format_percent(val):
        return f"{val:.2%}" if isinstance(val, (int, float)) else "--"

    def safe_div(a, b):
        return a / b if b else None

    col1, col2 = st.columns(2)
    col1.metric("Market Cap", format_value(info.get("marketCap")))
    col2.metric("PE Ratio", info.get("trailingPE"))
    col1.metric("PB Ratio", info.get("priceToBook"))
    col2.metric("EPS (TTM)", info.get("trailingEps"))
    col1.metric("ROE", format_percent(info.get("returnOnEquity")))
    col2.metric("ROA", format_percent(info.get("returnOnAssets")))
    col1.metric("Dividend Yield", format_percent(info.get("dividendYield")))

    st.markdown("---")
    st.markdown("**📖 Company Overview**")
    st.info(info.get("longBusinessSummary", "No summary available."))
