# 📈 Stock Intelligence Terminal

A powerful and interactive **Multi-Asset Financial Terminal** built with **Python**, **Streamlit**, and **yFinance**. This professional-grade platform provides Bloomberg Terminal and TradingView-like functionality.

## 🚀 Live Demo

🔗 [Click here to open the app](https://stock-intelligence-terminal-jw4mnzh4es7nty7dkuganf.streamlit.app/)

## ✨ Key Features

### 🏛️ **Professional Terminal Interface**
- Bloomberg-style dark theme with orange accents
- Command-driven interface (`AAPL FA`, `ES=F CH`, `BTC-USD OP`)
- Fuzzy search with intelligent suggestions
- Multi-asset class support

### 📊 **Advanced Charting**
- TradingView Lightweight Charts integration
- Real-time price updates via WebSocket
- Technical indicators (RSI, MACD, Bollinger Bands)
- Drawing tools (trend lines, Fibonacci retracements)
- Multi-timeframe analysis

### 🌍 **Multi-Asset Coverage**
- **Equities**: Stocks from NSE, NASDAQ, NYSE
- **Futures**: ES=F, NQ=F, YM=F, GC=F, CL=F
- **Forex**: EURUSD=X, GBPUSD=X, USDJPY=X
- **Crypto**: BTC-USD, ETH-USD, ADA-USD
- **Commodities**: Gold, Oil, Natural Gas

### 💼 **Portfolio Analytics**
- Position tracking and risk metrics
- Sharpe ratio, Sortino ratio, VaR calculations
- Allocation analysis and performance charts
- Multi-asset portfolio optimization

### 🔍 **Professional Analysis Tools**
- Options chain analysis with Greeks
- Stock screener with advanced filters
- Real-time news integration
- Fundamental analysis (P/E, ROE, debt ratios)
- Financial statements (Income, Balance Sheet, Cash Flow)

### ⚡ **Performance & Reliability**
- Async data fetching with retry mechanisms
- Parquet caching for fast data access
- Error handling and data validation
- Docker containerization with nginx

## �️ Tech Stack

- **Frontend**: Streamlit with custom CSS styling
- **Charts**: TradingView Lightweight Charts
- **Data**: yFinance, Finnhub API
- **Analytics**: TA-Lib, NumPy, Pandas
- **Caching**: FastParquet, Redis-like caching
- **Deployment**: Docker, nginx, Poetry

## 📦 Installation

### Option 1: Poetry (Recommended)
```bash
git clone https://github.com/Sahil7636/stock-intelligence-terminal.git
cd stock-intelligence-terminal
poetry install
poetry run streamlit run app.py
```

### Option 2: pip
```bash
git clone https://github.com/Sahil7636/stock-intelligence-terminal.git
cd stock-intelligence-terminal
pip install -r requirements.txt
streamlit run app.py
```

### Option 3: Docker
```bash
git clone https://github.com/Sahil7636/stock-intelligence-terminal.git
cd stock-intelligence-terminal
docker build -t stock-terminal .
docker run -p 8501:8501 stock-terminal
```

## 🎯 Command Reference

### Basic Commands
```bash
AAPL FA          # Apple fundamentals
MSFT CH          # Microsoft chart
GOOGL IS         # Google income statement
TSLA BS          # Tesla balance sheet
AMZN CF          # Amazon cash flow
```

### Multi-Asset Commands
```bash
ES=F CH          # S&P 500 futures chart
EURUSD=X GO      # Euro/USD forex overview
BTC-USD OP       # Bitcoin options
GC=F FA          # Gold futures fundamentals
```

### Utility Commands
```bash
SC               # Stock screener
PF               # Portfolio analytics
WL               # Multi-asset watchlist
NW               # News feed
```

## ⚠️ Compliance & Disclaimers

- **Educational Purpose**: This platform is for educational and research purposes only
- **Not Investment Advice**: No recommendations or investment advice provided
- **Delayed Data**: All data feeds are delayed (15-20 minutes)
- **SEBI Compliance**: Follows SEBI and exchange board guidelines
- **No Trading**: No real-time trading execution capabilities

## 🔧 Configuration

### API Keys
1. **Finnhub**: Get free API key from [finnhub.io](https://finnhub.io)
2. **Real-time Data**: Enter API key in sidebar for live updates

### Environment Variables
```bash
FINNHUB_API_KEY=your_api_key_here
STREAMLIT_SERVER_PORT=8501
```

## 🚀 Advanced Features

### Custom Indicators
Add your own technical indicators by modifying the TA-Lib section:
```python
# Add custom indicator
custom_indicator = talib.YOUR_INDICATOR(close_np, timeperiod=14)
```

### Portfolio Optimization
Upload CSV with format:
```csv
Ticker,Quantity
AAPL,100
MSFT,50
GOOGL,25
```

### Drawing Tools
- **Trend Lines**: Click two points to draw trend lines
- **Horizontal Lines**: Click to draw support/resistance levels
- **Fibonacci**: Click two points for Fibonacci retracements

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TradingView** for Lightweight Charts library
- **Yahoo Finance** for market data
- **Finnhub** for real-time data feeds
- **TA-Lib** for technical analysis
- **Streamlit** for the web framework

---

**⚡ Built with ❤️ for traders, analysts, and financial professionals**
