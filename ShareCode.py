import logging
import time
from typing import List, Optional

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class DataFetchError(Exception):
    """Custom exception for data fetching errors."""


def _retry(fn, max_retries: int = 3, backoff_factor: float = 0.5, *args, **kwargs):
    """Helper to retry a function with exponential backoff on exceptions."""
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                logger.exception("Max retries reached. Last error: %s", e)
                raise
            sleep_time = backoff_factor * (2 ** (attempt - 1))
            logger.warning("Error calling %s: %s. Retrying in %.1f seconds (attempt %d/%d)...",
                           getattr(fn, "__name__", str(fn)), e, sleep_time, attempt, max_retries)
            time.sleep(sleep_time)


def fetch_history(ticker: str, start: str = None, end: str = None, interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a ticker. Returns an empty DataFrame on failure.
    Dates like '2023-01-01'.
    """
    try:
        t = yf.Ticker(ticker)
        df = _retry(t.history, max_retries=3, backoff_factor=0.5, start=start, end=end, interval=interval)
        if df is None:
            logger.error("Received None when fetching history for %s", ticker)
            return pd.DataFrame()
        if df.empty:
            logger.info("No historical data returned for %s (start=%s end=%s interval=%s).",
                        ticker, start, end, interval)
        return df
    except Exception:
        logger.exception("Failed to fetch history for %s", ticker)
        return pd.DataFrame()


def fetch_latest_price(ticker: str) -> Optional[float]:
    """
    Get the latest available price.
    Returns a float price or None if it can't be retrieved.
    """
    try:
        t = yf.Ticker(ticker)
        # Try intraday first
        try:
            df = _retry(t.history, max_retries=2, backoff_factor=0.3, period="1d", interval="1m")
        except Exception:
            logger.warning("Intraday fetch failed for %s; will attempt fallback.", ticker)
            df = pd.DataFrame()

        if df is None or df.empty:
            # fallback to a small daily window
            try:
                df = _retry(t.history, max_retries=2, backoff_factor=0.3, period="5d", interval="1d")
            except Exception:
                logger.exception("Fallback history fetch failed for %s", ticker)
                return None

        if df is None or df.empty:
            logger.error("No price data available for %s", ticker)
            return None

        # Ensure 'Close' exists
        if 'Close' not in df.columns:
            logger.error("No 'Close' column in data for %s", ticker)
            return None

        price = float(df['Close'].iloc[-1])
        logger.info("Latest price for %s: %s", ticker, price)
        return price
    except Exception:
        logger.exception("Unhandled error while fetching latest price for %s", ticker)
        return None


def fetch_multiple(tickers: List[str], period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Download multiple tickers; returns a DataFrame (possibly MultiIndex).
    Returns an empty DataFrame on failure.
    """
    try:
        df = _retry(yf.download, max_retries=3, backoff_factor=0.5,
                    tickers=tickers, period=period, interval=interval, group_by='ticker', threads=True)
        if df is None:
            logger.error("yf.download returned None for tickers: %s", tickers)
            return pd.DataFrame()
        if df.empty:
            logger.info("No data returned for tickers %s (period=%s interval=%s).", tickers, period, interval)
        return df
    except Exception:
        logger.exception("Failed to download data for tickers: %s", tickers)
        return pd.DataFrame()


if __name__ == "__main__":
    try:
        # Example: get Apple daily history and save
        df_aapl = fetch_history("AAPL", start="2023-01-01", end=None, interval="1d")
        if not df_aapl.empty:
            logger.info("AAPL history last rows:\n%s", df_aapl.tail())
            try:
                df_aapl.to_csv("AAPL_history.csv")
                logger.info("Saved AAPL_history.csv")
            except Exception:
                logger.exception("Failed to save AAPL_history.csv")

        # Latest price example
        price = fetch_latest_price("AAPL")
        if price is not None:
            print(f"AAPL latest price: {price}")
        else:
            print("Failed to fetch AAPL latest price. See logs for details.")

        # Multiple tickers
        tickers = ["AAPL", "MSFT", "GOOGL"]
        multi = fetch_multiple(tickers, period="6mo")
        if not multi.empty:
            logger.info("Downloaded multiple tickers head:\n%s", multi.head())
        else:
            logger.warning("No data downloaded for multiple tickers.")

        # Simple plot of Close (if available)
        try:
            if not df_aapl.empty and 'Close' in df_aapl.columns:
                ax = df_aapl['Close'].plot(title="AAPL Close Price")
                fig = ax.get_figure()
                # Try to show; if running headless consider saving instead
                try:
                    plt.show()
                except Exception:
                    # fallback: save to file
                    fig.savefig("AAPL_close.png")
                    logger.info("Saved plot to AAPL_close.png")
            else:
                logger.info("Skipping plot: no close data available for AAPL.")
        except Exception:
            logger.exception("Failed while plotting data.")
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting.")
    except Exception:
        logger.exception("Unexpected error in main execution.")