import yfinance as yf
import pandas as pd

def get_stock_data(ticker="AAPL", start="2015-01-01", end="2023-12-31"):
    df = yf.download(ticker, start=start, end=end)
    df = df[['Close']].dropna()
    return df