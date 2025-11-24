import yfinance as yf
import pandas as pd

import matplotlib.pyplot as plt

def fetch_history(ticker: str, start: str = None, end: str = None, interval: str = "1d") -> pd.DataFrame:
    """Fetch historical OHLCV data for a ticker. Dates like '2023-01-01'."""
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=end, interval=interval)
    return df

def fetch_latest_price(ticker: str) -> float:
    """Get the latest available price (uses last row of 1d history)."""
    df = yf.Ticker(ticker).history(period="1d", interval="1m")
    if df.empty:
        # fallback to previous close
        df = yf.Ticker(ticker).history(period="5d", interval="1d")
    return float(df['Close'].iloc[-1])

def fetch_multiple(tickers: list, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Download multiple tickers; returns a MultiIndex DataFrame when more than one ticker."""
    return yf.download(tickers, period=period, interval=interval, group_by='ticker', threads=True)

if __name__ == "__main__":
    # Example: get Apple daily history and save
    df_aapl = fetch_history("AAPL", start="2023-01-01", end=None, interval="1d")
    print(df_aapl.tail())
    df_aapl.to_csv("AAPL_history.csv")

    # Latest price example
    price = fetch_latest_price("AAPL")
    print(f"AAPL latest price: {price}")

    # Multiple tickers
    tickers = ["AAPL", "MSFT", "GOOGL"]
    multi = fetch_multiple(tickers, period="6mo")
    print(multi.head())

    # Simple plot of Close
    df_aapl['Close'].plot(title="AAPL Close Price")
    plt.show()
