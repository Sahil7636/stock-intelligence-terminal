import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
from datetime import timedelta
import requests
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="📈 Stock Intelligence Terminal", layout="wide")

st.markdown("""
# 📊 Stock Intelligence Terminal
**Quant-Grade Dashboard** | Real-time Price | Financial Metrics
""")

# ---------------- Sidebar ----------------
market = st.sidebar.selectbox("Market", ["India (NSE)", "USA (NASDAQ)"])

@st.cache_data
def get_tickers_auto(market):
    if market == "India (NSE)":
        url = "https://www1.nseindia.com/content/equities/EQUITY_L.csv"
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            response = requests.get(url, headers=headers)
            df = pd.read_csv(pd.compat.StringIO(response.text))
            df.columns = df.columns.str.strip()
            return sorted(df['SYMBOL'].dropna().astype(str).apply(lambda x: x + ".NS"))
        except:
            return ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    else:
        url = "https://raw.githubusercontent.com/datasets/nasdaq-listings/master/data/nasdaq-listed-symbols.csv"
        df = pd.read_csv(url)
        return df['Symbol'].dropna().sort_values().tolist()

stock_list = get_tickers_auto(market)
ticker = st.sidebar.selectbox("Select Stock", stock_list)
period = st.sidebar.selectbox("Time Frame", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
chart_type = st.sidebar.selectbox("Chart Type", ["Candlestick", "Line", "Bar"])
show_ma = st.sidebar.checkbox("Show 20-Day MA", True)

# ---------------- Fetch Data ----------------
try:
    stock = yf.Ticker(ticker)
    info = stock.info
    data = stock.history(period=period)
    intraday = yf.download(ticker, period="1d", interval="1m")
    latest_price = intraday["Close"].iloc[-1]
    st.sidebar.metric("📍 Live Price (1m)", f"{latest_price:.2f}")
except Exception as e:
    st.error("Failed to load data.")
    st.stop()

data = data.asfreq('B')
data["Close"].interpolate(method='linear', inplace=True)
data["MA20"] = data["Close"].rolling(20).mean()

# ---------------- Formatters ----------------
def format_value(val):
    if isinstance(val, (int, float)):
        symbol = "₹" if ticker.endswith(".NS") else "$"
        return f"{symbol}{val:,.0f}" if abs(val) > 1 else f"{symbol}{val:.2f}"
    return "--"

def format_ratio(val):
    return f"{val:.2f}" if isinstance(val, (int, float)) else "--"

def format_percent(val):
    return f"{val:.2%}" if isinstance(val, (int, float)) else "--"

def safe_div(a, b):
    return a / b if b else None

# ---------------- Layout ----------------
left, right = st.columns((2, 1))

with left:
    st.subheader("📈 Price Chart")

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

    fig.update_layout(
        hovermode="x unified",
        xaxis_rangeslider_visible=True,
        height=500,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="#f4f4f4",
        paper_bgcolor="#f4f4f4",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🔍 Data Preview")
    st.dataframe(data.tail(10), use_container_width=True, height=200)

    st.download_button(
        "📥 Download CSV",
        data.to_csv().encode('utf-8'),
        file_name=f"{ticker}_data.csv",
        mime='text/csv'
    )

with right:
    st.subheader("📋 Key Financial Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Market Cap", format_value(info.get("marketCap")))
    col2.metric("PE Ratio", format_ratio(info.get("trailingPE")))
    col1.metric("PB Ratio", format_ratio(info.get("priceToBook")))
    col2.metric("EPS (TTM)", format_ratio(info.get("trailingEps")))
    col1.metric("ROE", format_percent(info.get("returnOnEquity")))
    col2.metric("ROA", format_percent(info.get("returnOnAssets")))
    col1.metric("Dividend Yield", format_percent(info.get("dividendYield")))

    st.markdown("---")
    st.markdown("**📖 Company Overview**")
    st.info(info.get("longBusinessSummary", "No summary available."))

    if info.get("website"):
        st.markdown(f"🌐 [Visit Website]({info.get('website')})", unsafe_allow_html=True)

    st.markdown("### 📌 M&A / Valuation Metrics")
    try:
        bs = stock.balance_sheet
        cf = stock.cashflow
        income = stock.financials

        total_debt = bs.loc["Total Liab"].iloc[0]
        total_equity = bs.loc["Total Stockholder Equity"].iloc[0]
        current_assets = bs.loc["Total Current Assets"].iloc[0]
        current_liabilities = bs.loc["Total Current Liabilities"].iloc[0]
        total_revenue = income.loc["Total Revenue"].iloc[0]
        net_income = income.loc["Net Income"].iloc[0]
        free_cash_flow = cf.loc["Total Cash From Operating Activities"].iloc[0] - cf.loc["Capital Expenditures"].iloc[0]

        ev_ebitda = info.get("enterpriseToEbitda")
        ev_rev = info.get("enterpriseToRevenue")

        col3, col4 = st.columns(2)
        col3.metric("EV / EBITDA", format_ratio(ev_ebitda))
        col4.metric("EV / Revenue", format_ratio(ev_rev))
        col3.metric("Debt / Equity", format_ratio(safe_div(total_debt, total_equity)))
        col4.metric("Current Ratio", format_ratio(safe_div(current_assets, current_liabilities)))
        col3.metric("Net Profit Margin", format_percent(safe_div(net_income, total_revenue)))
        col4.metric("Free Cash Flow", format_value(free_cash_flow))

    except Exception:
        st.warning("⚠️ Advanced financial metrics not available for this stock.")
