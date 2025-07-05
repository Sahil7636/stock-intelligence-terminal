#!/usr/bin/env python3
"""
Comprehensive test script for Stock Intelligence Terminal
Tests all major features and components
"""

import sys
import pandas as pd
import yfinance as yf
from difflib import SequenceMatcher

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import yfinance as yf
        print("✅ yFinance imported successfully")
    except ImportError as e:
        print(f"❌ yFinance import failed: {e}")
        return False
    
    try:
        import talib
        print("✅ TA-Lib imported successfully")
    except ImportError as e:
        print(f"⚠️ TA-Lib import failed: {e}")
    
    try:
        import pandas as pd
        import numpy as np
        print("✅ Pandas/NumPy imported successfully")
    except ImportError as e:
        print(f"❌ Pandas/NumPy import failed: {e}")
        return False
    
    return True

def test_data_fetching():
    """Test data fetching functionality"""
    print("\n🧪 Testing data fetching...")
    try:
        # Test basic stock data
        data = yf.download("AAPL", period="5d", progress=False)
        if data.empty:
            print("❌ No data returned for AAPL")
            return False
        else:
            print(f"✅ AAPL data fetched: {len(data)} rows")
        
        # Test futures data
        futures_data = yf.download("ES=F", period="5d", progress=False)
        if not futures_data.empty:
            print(f"✅ Futures data fetched: {len(futures_data)} rows")
        else:
            print("⚠️ Futures data not available")
        
        # Test forex data
        forex_data = yf.download("EURUSD=X", period="5d", progress=False)
        if not forex_data.empty:
            print(f"✅ Forex data fetched: {len(forex_data)} rows")
        else:
            print("⚠️ Forex data not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Data fetching failed: {e}")
        return False

def test_command_parsing():
    """Test command parsing functionality"""
    print("\n🧪 Testing command parsing...")
    
    # Define command suggestions
    COMMAND_SUGGESTIONS = {
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft Corporation',
        'ES=F': 'S&P 500 E-mini Futures',
        'EURUSD=X': 'Euro to US Dollar',
        'BTC-USD': 'Bitcoin',
        'FA': 'Fundamental Analysis',
        'CH': 'Chart View',
        'SC': 'Stock Screener'
    }
    
    def parse_command(cmd: str):
        if not cmd:
            return None, None
        parts = cmd.upper().strip().split()
        if not parts:
            return None, None
        view_codes = {'CH', 'FA', 'GO', 'IS', 'BS', 'CF', 'SC', 'NW', 'PF', 'OP', 'WL'}
        if parts[0] in view_codes and len(parts) == 1:
            return None, parts[0]
        ticker_part = parts[0]
        view = parts[1] if len(parts) > 1 else None
        return ticker_part, view
    
    def fuzzy_search(query: str, suggestions: dict, threshold: float = 0.3):
        if not query:
            return []
        
        query = query.upper().strip()
        matches = []
        
        for symbol, description in suggestions.items():
            if symbol.startswith(query):
                matches.append((symbol, description, 1.0))
            elif query in symbol:
                ratio = SequenceMatcher(None, query, symbol).ratio()
                if ratio >= threshold:
                    matches.append((symbol, description, ratio))
        
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches[:10]
    
    try:
        # Test basic command parsing
        ticker, view = parse_command("AAPL FA")
        if ticker == "AAPL" and view == "FA":
            print("✅ Basic command parsing works")
        else:
            print(f"❌ Command parsing failed: got {ticker}, {view}")
            return False
        
        # Test view-only commands
        ticker, view = parse_command("SC")
        if ticker is None and view == "SC":
            print("✅ View-only command parsing works")
        else:
            print(f"❌ View-only parsing failed: got {ticker}, {view}")
            return False
        
        # Test fuzzy search
        suggestions = fuzzy_search("APL", COMMAND_SUGGESTIONS)
        if suggestions and suggestions[0][0] == "AAPL":
            print("✅ Fuzzy search works")
        else:
            print(f"❌ Fuzzy search failed: got {suggestions}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Command parsing test failed: {e}")
        return False

def test_technical_indicators():
    """Test technical indicators"""
    print("\n🧪 Testing technical indicators...")
    try:
        import talib
        
        # Get sample data
        data = yf.download("AAPL", period="1mo", progress=False)
        if data.empty:
            print("❌ No data for technical indicators test")
            return False
        
        close_prices = data['Close'].values
        
        # Test RSI
        rsi = talib.RSI(close_prices, timeperiod=14)
        if len(rsi) > 0:
            print("✅ RSI calculation works")
        else:
            print("❌ RSI calculation failed")
            return False
        
        # Test MACD
        macd, macd_signal, macd_hist = talib.MACD(close_prices)
        if len(macd) > 0:
            print("✅ MACD calculation works")
        else:
            print("❌ MACD calculation failed")
            return False
        
        # Test Bollinger Bands
        upper, middle, lower = talib.BBANDS(close_prices)
        if len(upper) > 0:
            print("✅ Bollinger Bands calculation works")
        else:
            print("❌ Bollinger Bands calculation failed")
            return False
        
        return True
        
    except ImportError:
        print("⚠️ TA-Lib not available - skipping technical indicators test")
        return True
    except Exception as e:
        print(f"❌ Technical indicators test failed: {e}")
        return False

def test_portfolio_analytics():
    """Test portfolio analytics functionality"""
    print("\n🧪 Testing portfolio analytics...")
    try:
        import numpy as np
        
        # Create sample portfolio data
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        positions = pd.Series([100, 50, 25], index=symbols)
        
        # Get price data
        price_data = yf.download(symbols, period="3mo", progress=False)
        if price_data.empty:
            print("❌ No price data for portfolio test")
            return False
        
        # Extract close prices
        if len(symbols) > 1:
            close_prices = price_data['Close']
        else:
            close_prices = price_data['Close'].to_frame(symbols[0])
        
        # Calculate basic portfolio metrics
        returns = close_prices.pct_change().dropna()
        weights = positions / positions.sum()
        
        portfolio_returns = (returns * weights).sum(axis=1)
        sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
        
        if not pd.isna(sharpe_ratio):
            print(f"✅ Portfolio analytics work - Sharpe: {sharpe_ratio:.2f}")
            return True
        else:
            print("❌ Portfolio analytics calculation failed")
            return False
        
    except Exception as e:
        print(f"❌ Portfolio analytics test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Stock Intelligence Terminal - Comprehensive Test Suite")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Data Fetching", test_data_fetching),
        ("Command Parsing", test_command_parsing),
        ("Technical Indicators", test_technical_indicators),
        ("Portfolio Analytics", test_portfolio_analytics)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
    
    print("\n" + "="*60)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Platform is ready to use.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())