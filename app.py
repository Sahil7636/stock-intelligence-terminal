import streamlit as st
import yfinance as yf
# Replaced Plotly with TradingView Lightweight Charts
import json
try:
    import talib
except ImportError:
    talib = None  # TALib might not be installed in dev env
import streamlit.components.v1 as components
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
import asyncio
import aiohttp
import os
from pathlib import Path

st.set_page_config(page_title="📈 Stock Intelligence Terminal", layout="wide")

# Optional styling
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

st.title("📊 Stock Intelligence Terminal")

# ---------------- Sidebar ----------------
market = st.sidebar.selectbox("Market", ["India (NSE)", "USA (NASDAQ)"])

@st.cache_data
def get_tickers(market):
    if market == "India (NSE)":
        return sorted([
            "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
            "ICICIBANK.NS", "ITC.NS", "LT.NS", "SBIN.NS", "AXISBANK.NS", "BHARTIARTL.NS"
        ])
    else:
        url = "https://raw.githubusercontent.com/datasets/nasdaq-listings/master/data/nasdaq-listed-symbols.csv"
        df = pd.read_csv(url)
        return df['Symbol'].dropna().sort_values().tolist()

stock_list = get_tickers(market)
ticker = st.sidebar.selectbox("Select Stock", stock_list)
period = st.sidebar.selectbox("Time Frame", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
chart_type = st.sidebar.selectbox("Chart Type", ["Candlestick", "Line", "Bar"])
show_ma = st.sidebar.checkbox("Show 20-Day MA", True)
use_forecast = st.sidebar.checkbox("🔮 Predict Future with ARIMA")
forecast_days = st.sidebar.slider("Days to Forecast", 1, 30, 7) if use_forecast else 0

# Technical indicators selection
available_indicators = ["RSI", "MACD", "Bollinger Bands"]
indicators_selected = st.sidebar.multiselect("Technical Indicators", available_indicators)

# Real-time WebSocket options
enable_realtime = st.sidebar.checkbox("🔌 Real-Time Updates (Finnhub WS)")
finnhub_token = ""
if enable_realtime:
    finnhub_token = st.sidebar.text_input("Finnhub API Token", value="demo", type="password")

# ---------------- Cached historical data ----------------
@st.cache_data(ttl=3600)
def get_history_data(ticker: str, period: str) -> pd.DataFrame:
    """Return historical price data, cached on disk as parquet (fastparquet engine)."""
    cache_dir = Path("data_cache")
    cache_dir.mkdir(exist_ok=True)
    file_path = cache_dir / f"{ticker}_{period}.parquet"

    if file_path.exists():
        try:
            return pd.read_parquet(file_path)
        except Exception:
            file_path.unlink(missing_ok=True)  # remove corrupt cache and refetch

    df = yf.download(ticker, period=period, progress=False)
    if not df.empty:
        df.to_parquet(file_path, engine="fastparquet")
    return df

# ---------------- Fetch Data ----------------
stock = yf.Ticker(ticker)
info = stock.info
# Replace direct yfinance history call with cached version
data = get_history_data(ticker, period)
data = data.asfreq('B')
data["Close"].interpolate(method='linear', inplace=True)
data["MA20"] = data["Close"].rolling(20).mean()

# ---------------- TA-Lib Indicators ----------------
indicator_payloads = []

if talib:
    close_np = data["Close"].values
    idx_ts = data.index

    if "RSI" in indicators_selected:
        rsi = talib.RSI(close_np, timeperiod=14)
        rsi_data = [{"time": int(ts.timestamp()), "value": float(val)} for ts, val in zip(idx_ts, rsi) if not pd.isna(val)]
        indicator_payloads.append({"name": "RSI", "type": "line", "color": "purple", "data": rsi_data})

    if "MACD" in indicators_selected:
        macd, macd_signal, macd_hist = talib.MACD(close_np, fastperiod=12, slowperiod=26, signalperiod=9)
        macd_data = [{"time": int(ts.timestamp()), "value": float(val)} for ts, val in zip(idx_ts, macd) if not pd.isna(val)]
        indicator_payloads.append({"name": "MACD", "type": "line", "color": "teal", "data": macd_data})

    if "Bollinger Bands" in indicators_selected:
        upper, middle, lower = talib.BBANDS(close_np, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        upper_data = [{"time": int(ts.timestamp()), "value": float(val)} for ts, val in zip(idx_ts, upper) if not pd.isna(val)]
        lower_data = [{"time": int(ts.timestamp()), "value": float(val)} for ts, val in zip(idx_ts, lower) if not pd.isna(val)]
        indicator_payloads.append({"name": "Upper Band", "type": "line", "color": "#FFB6C1", "data": upper_data})
        indicator_payloads.append({"name": "Lower Band", "type": "line", "color": "#87CEFA", "data": lower_data})

# else no talib -> warn user
elif indicators_selected:
    st.warning("TA-Lib not available. Install dependencies to enable indicators.")

# ---------------- Async price fetch helpers ----------------
async def _fetch_quote(session, ticker: str):
    """Fetch a single quote from Yahoo Finance REST endpoint."""
    url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={ticker}"
    async with session.get(url) as resp:
        data = await resp.json()
    result = data.get("quoteResponse", {}).get("result", [])
    return ticker, (result[0] if result else {})

async def _fetch_quotes_async(tickers):
    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_quote(session, t) for t in tickers]
        return await asyncio.gather(*tasks)

@st.cache_data(ttl=60)
def get_current_prices(tickers):
    """Synchronous wrapper so Streamlit can cache the async quote fetch."""
    results = asyncio.run(_fetch_quotes_async(tickers))
    # Convert list[tuple] -> dict[str, dict]
    return {t: r for t, r in results if r}

# Current price using async fetch
try:
    current_price = get_current_prices([ticker])[ticker]["regularMarketPrice"]
    st.sidebar.metric("📍 Current Price", f"{current_price:.2f}")
except Exception:
    st.sidebar.warning("⚠️ Current price unavailable.")

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
        st.error(f"Forecasting failed: {e}")

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

    # -------- TradingView Lightweight Charts integration --------
    # Build data payloads depending on selected chart type
    if chart_type in ("Candlestick", "Bar"):
        chart_data = [
            {"time": int(ts.timestamp()), "open": float(o), "high": float(h), "low": float(l), "close": float(c)}
            for ts, o, h, l, c in zip(data.index, data["Open"], data["High"], data["Low"], data["Close"])
        ]
    else:  # Line chart
        chart_data = [
            {"time": int(ts.timestamp()), "value": float(v)}
            for ts, v in zip(data.index, data["Close"])
        ]

    ma_data = None
    if show_ma and chart_type != "Bar":
        ma_data = [
            {"time": int(ts.timestamp()), "value": float(v)}
            for ts, v in zip(data.index, data["MA20"])
        ]

    forecast_data = None
    if forecast_series is not None:
        forecast_data = [
            {"time": int(ts.timestamp()), "value": float(v)}
            for ts, v in forecast_series.items()
        ]

    payload = json.dumps(
        {
            "series_type": chart_type.lower(),
            "chart_data": chart_data,
            "ma_data": ma_data,
            "forecast_data": forecast_data,
            "enable_realtime": enable_realtime,
            "ws_token": finnhub_token,
            "ticker": ticker.split(".")[0],
            "indicators": indicator_payloads,
        }
    )

    html_content = f"""
    <style>
    .toolbar {{
        position:absolute;
        top:10px;
        left:10px;
        z-index:1000;
        background:rgba(255,255,255,0.85);
        border-radius:4px;
        padding:4px;
        font-family:sans-serif;
    }}
    .toolbar button {{
        margin:2px;
        padding:4px 6px;
        font-size:12px;
        cursor:pointer;
    }}
    </style>
    <div style='position:relative;width:100%;height:500px;'>
        <div id='tv_chart' style='width:100%;height:500px;'></div>
        <div id='toolbar' class='toolbar'>
            <button id='btnLine'>Trend</button>
            <button id='btnHLine'>H&nbsp;Line</button>
            <button id='btnFib'>Fib</button>
            <button id='btnClear'>Clear</button>
        </div>
    </div>
    <script src='https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js'></script>
    <script>
    const container = document.getElementById('tv_chart');
    const chart = LightweightCharts.createChart(container, {{
        layout: {{ backgroundColor: '#f4f4f4', textColor: '#333' }},
        grid: {{ vertLines: {{ color: '#e1e3e8' }}, horzLines: {{ color: '#e1e3e8' }} }},
        width: container.clientWidth,
        height: 500,
        timeScale: {{ borderVisible: false }},
        rightPriceScale: {{ borderVisible: false }},
    }});

    const payload = {payload};
    const enableRealtime = payload.enable_realtime && payload.ws_token;

    let mainSeries;
    switch(payload.series_type) {{
        case 'candlestick':
            mainSeries = chart.addCandlestickSeries();
            break;
        case 'bar':
            mainSeries = chart.addBarSeries();
            break;
        default:
            mainSeries = chart.addLineSeries();
    }}
    mainSeries.setData(payload.chart_data);

    // ---------- Drawing tools ----------
    const drawings = [];
    let drawingMode = null;
    let firstPoint = null;
    document.getElementById('btnLine').onclick = () => {{ drawingMode = 'line'; firstPoint=null; }};
    document.getElementById('btnHLine').onclick = () => {{ drawingMode = 'hline'; firstPoint=null; }};
    document.getElementById('btnFib').onclick = () => {{ drawingMode = 'fib'; firstPoint=null; }};
    document.getElementById('btnClear').onclick = () => {{ drawings.forEach(s => chart.removeSeries(s)); drawings.length = 0; }};

    chart.subscribeClick(param => {{
        if (!drawingMode || param.time === undefined || param.price === undefined) return;
        if (!firstPoint) {{
            firstPoint = param;
            return;
        }}
        const leftTime = payload.chart_data[0].time;
        const rightTime = payload.chart_data[payload.chart_data.length-1].time;
        switch(drawingMode) {{
            case 'line': {{
                const s = chart.addLineSeries({{ color: 'red', lineWidth: 1 }});
                s.setData([{{ time:firstPoint.time, value:firstPoint.price }}, {{ time:param.time, value:param.price }}]);
                drawings.push(s);
                break;
            }}
            case 'hline': {{
                const s = chart.addLineSeries({{ color: 'blue', lineWidth: 1 }});
                s.setData([{{ time:leftTime, value:firstPoint.price }}, {{ time:rightTime, value:firstPoint.price }}]);
                drawings.push(s);
                break;
            }}
            case 'fib': {{
                const minP = Math.min(firstPoint.price, param.price);
                const maxP = Math.max(firstPoint.price, param.price);
                const levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
                levels.forEach(lvl => {{
                    const p = maxP - (maxP - minP) * lvl;
                    const s = chart.addLineSeries({{ color: '#FFA500', lineWidth: 1, lineStyle: 2 }});
                    s.setData([{{ time:firstPoint.time, value:p }}, {{ time:param.time, value:p }}]);
                    drawings.push(s);
                }});
                break;
            }}
        }}
        firstPoint = null;
        drawingMode = null;
    }});
    // ---------- End Drawing tools ----------

    // Add technical indicators as separate line series
    if (payload.indicators) {{
        payload.indicators.forEach(ind => {{
            const series = chart.addLineSeries({{ color: ind.color || 'grey', lineWidth: 1 }});
            series.setData(ind.data);
        }});
    }}

    if (enableRealtime) {{
        const ws = new WebSocket(`wss://ws.finnhub.io?token=${{payload.ws_token}}`);
        ws.addEventListener('open', () => {{
            ws.send(JSON.stringify({{ type: 'subscribe', symbol: payload.ticker }}));
        }});
        ws.addEventListener('message', (event) => {{
            const msg = JSON.parse(event.data);
            if (msg.data) {{
                msg.data.forEach(pt => {{
                    // Finnhub returns trade ticks; update last value
                    const updateObj = payload.series_type === 'candlestick' || payload.series_type === 'bar'
                        ? {{ time: Math.floor(pt.t/1000), open: pt.p, high: pt.p, low: pt.p, close: pt.p }}
                        : {{ time: Math.floor(pt.t/1000), value: pt.p }};
                    mainSeries.update(updateObj);
                }});
            }}
        }});
    }}

    if (payload.ma_data) {{
        const maSeries = chart.addLineSeries({{ color: 'orange', lineWidth: 1, lineStyle: 1 }});
        maSeries.setData(payload.ma_data);
    }}

    if (payload.forecast_data) {{
        const forecastSeries = chart.addLineSeries({{ color: 'green', lineWidth: 1, lineStyle: 2 }});
        forecastSeries.setData(payload.forecast_data);
    }}

    window.addEventListener('resize', () => {{
        chart.applyOptions({{ width: container.clientWidth }});
    }});
    </script>
    """

    components.html(html_content, height=520)

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
