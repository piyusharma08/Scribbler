"""Price/volume history fetching via yfinance, with Streamlit caching."""
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(ticker: str, period: str = "6mo", interval: str = "1d"):
    """Fetch OHLCV history for a single ticker. Returns a DataFrame or None on failure."""
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        if df is None or df.empty or len(df) < 30:
            return None
        df = df.dropna()
        return df
    except Exception:
        return None


def fetch_many(tickers, period="6mo", interval="1d", max_workers=8, progress_cb=None):
    """Fetch history for many tickers concurrently. Returns dict(ticker -> df)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}
    total = len(tickers)
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(fetch_history, t, period, interval): t for t in tickers
        }
        for fut in as_completed(futures):
            t = futures[fut]
            results[t] = fut.result()
            done += 1
            if progress_cb:
                progress_cb(done, total, t)
    return results
