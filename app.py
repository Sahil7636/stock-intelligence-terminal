import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

st.set_page_config(page_title="\ud83d\udcc8 Stock Intelligence Terminal", layout="wide")

st.markdown("""
# \ud83d\udcca Stock Intelligence Terminal
**Live Market Data | Technical Indicators | Financial Insights | Forecasting**
""")

# ---------------- Sidebar ----------------
market = st.sidebar.selectbox("Market", ["India (NSE)", "USA (NASDAQ)"])

@st.cache_data
def get_tickers_auto(market):
    try:
        if market == "India (NSE)":
            return sorted([
                "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
                "ICICIBANK.NS", "ITC.NS", "LT.NS", "SBIN.NS",
                "AXISBANK.NS", "BHARTIARTL.NS"
            ])
        else:
            url = "https://raw.githubusercontent.com/datasets/nasdaq-listings/master/data/nasdaq-listed-symbols.csv"
            df = pd.read_csv(url)
            return df['Symbol'].dropna().sort_values().tolist()
    except:
        return []

stock_list = get_tickers_auto(market)
ticker = st.sidebar.selectbox("Select Stock", stock_list)
period = st.sidebar.selectbox("Time Frame", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
chart_type = st.sidebar.selectbox("Chart Type", ["Candlestick", "Line", "Bar"])
show_ma = st.sidebar.checkbox("Show 20-Day MA", True)

# ---------------- Fetch Data ----------------
stock = yf.Ticker(ticker)
info = stock.info

try:
    data = stock.history(period=period)
    intraday = yf.download(ticker, period="1d", interval="1m")
    latest_price = intraday["Close"].iloc[-1]
    st.sidebar.metric("\ud83d\udccd Live Price (1m)", f"{latest_price:.2f}")
except:
    st.sidebar.warning("\u26a0\ufe0f Data unavailable.")
    data = pd.DataFrame()

if not data.empty:
    data["MA20"] = data["Close"].rolling(20).mean()

# ---------------- Formatters ----------------
def format_value(val):
    symbol = "\u20b9" if ticker.endswith(".NS") else "$"
    if isinstance(val, (int, float)):
        return f"{symbol}{val:,.0f}" if abs(val) > 1 else f"{symbol}{val:.2f}"
    return "--"

def format_ratio(val):
    return f"{val:.2f}" if isinstance(val, (int, float)) else "--"

def format_percent(val):
    return f"{val:.2%}" if isinstance(val, (int, float)) else "--"

def safe_div(a, b):
    return a / b if b else None

# ---------------- Layout ----------------
if not data.empty:
    left, right = st.columns((2, 1))

    with left:
        st.subheader("\ud83d\udcc8 Price Chart")

        fig = go.Figure()
        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Candlestick'))
        elif chart_type == "Line":
            fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close"))
        elif chart_type == "Bar":
            fig.add_trace(go.Bar(x=data.index, y=data["Close"], name="Bar"))

        if show_ma:
            fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], mode="lines", name="MA20", line=dict(color='orange', dash='dash')))

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

        st.markdown("#### \ud83d\udd0d Data Preview")
        st.dataframe(data.tail(10), use_container_width=True, height=200)

        st.download_button("\ud83d\udcc5 Download CSV", data.to_csv().encode('utf-8'), file_name=f"{ticker}_data.csv", mime='text/csv')

        # ARIMA Forecasting
        if len(data) > 30:
            st.markdown("### \ud83d\udd2e ARIMA Forecast (Next 10 Days)")
            close_series = data['Close'].dropna()
            model = ARIMA(close_series, order=(5,1,0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=10)
            forecast_index = pd.date_range(start=close_series.index[-1] + timedelta(days=1), periods=10)
            fig_arima = go.Figure()
            fig_arima.add_trace(go.Scatter(x=close_series.index, y=close_series, name='Historical'))
            fig_arima.add_trace(go.Scatter(x=forecast_index, y=forecast, name='Forecast'))
            fig_arima.update_layout(title="ARIMA Forecast", xaxis_title="Date", yaxis_title="Price", height=400)
            st.plotly_chart(fig_arima, use_container_width=True)

    with right:
        st.subheader("\ud83d\udccb Key Financial Metrics")
        col1, col2 = st.columns(2)
        col1.metric("Market Cap", format_value(info.get("marketCap")))
        col2.metric("PE Ratio", format_ratio(info.get("trailingPE")))
        col1.metric("PB Ratio", format_ratio(info.get("priceToBook")))
        col2.metric("EPS (TTM)", format_ratio(info.get("trailingEps")))
        col1.metric("ROE", format_percent(info.get("returnOnEquity")))
        col2.metric("ROA", format_percent(info.get("returnOnAssets")))
        col1.metric("Dividend Yield", format_percent(info.get("dividendYield")))

        st.markdown("---")
        st.markdown("**\ud83d\udcd6 Company Overview**")
        st.info(info.get("longBusinessSummary", "No summary available."))

        if info.get("website"):
            st.markdown(f"\ud83c\udf10 [Visit Website]({info.get('website')})", unsafe_allow_html=True)

        st.markdown("### \ud83d\udccc M&A / Valuation Metrics")
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
            st.warning("\u26a0\ufe0f Advanced financial metrics not available for this stock.")

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("<p style='text-align:center;'>Built with \u2764\ufe0f by [Your Name] | Powered by yFinance, ARIMA & Plotly</p>", unsafe_allow_html=True)
