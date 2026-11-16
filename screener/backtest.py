"""Backtest the bullish technical signals against realized forward returns.

For each ticker, replays `technical.compute_signal_table` over its full
history (purely causal, no look-ahead) and, for every day a given signal
fired, records the realized forward return N days later. Comparing each
signal's hit-rate/avg-return against an unconditional baseline (every day,
signal or not) tells you whether a signal actually precedes above-average
moves for these stocks - rather than just looking plausible.

This is intentionally simple (no position sizing, slippage, or exits) - it
answers "is this signal informative at all", not "would this be a profitable
trading strategy."
"""
import numpy as np
import pandas as pd

from . import prices
from .technical import compute_signal_table, SIGNAL_COLUMNS

DEFAULT_HORIZONS = (5, 10, 20)


def _forward_returns(close: pd.Series, horizons):
    return {h: close.shift(-h) / close - 1 for h in horizons}


def _collect(returns_by_horizon, mask, horizons):
    return {h: returns_by_horizon[h][mask].dropna() for h in horizons}


def backtest_ticker(ticker: str, period="3y", lookback_days=5, horizons=DEFAULT_HORIZONS, min_history=250):
    """Return per-bucket raw forward-return samples for one ticker, or None
    if there isn't enough history to backtest meaningfully."""
    df = prices.fetch_history(ticker, period=period)
    if df is None or len(df) < min_history + max(horizons):
        return None

    table = compute_signal_table(df, lookback_days=lookback_days)
    close = df["Close"]
    fwd = _forward_returns(close, horizons)

    valid = pd.Series(np.arange(len(df)) >= min_history, index=df.index)

    buckets = {"bullish": _collect(fwd, table["bullish"] & valid, horizons)}
    for col in SIGNAL_COLUMNS:
        buckets[col] = _collect(fwd, table[col] & valid, horizons)
    buckets["baseline"] = _collect(fwd, valid, horizons)
    return buckets


def _merge(all_buckets, horizons):
    """Merge per-ticker bucket dicts into bucket -> horizon -> concatenated Series."""
    merged = {}
    for ticker_buckets in all_buckets:
        if ticker_buckets is None:
            continue
        for bucket, by_h in ticker_buckets.items():
            slot = merged.setdefault(bucket, {h: [] for h in horizons})
            for h in horizons:
                slot[h].append(by_h[h])
    return {
        bucket: {
            h: (pd.concat(series_list) if series_list else pd.Series(dtype=float))
            for h, series_list in by_h.items()
        }
        for bucket, by_h in merged.items()
    }


def _summarize(returns: pd.Series) -> dict:
    n = len(returns)
    if n == 0:
        return {"n": 0, "hit_rate": None, "avg_return": None, "median_return": None}
    return {
        "n": n,
        "hit_rate": round(float((returns > 0).mean()) * 100, 1),
        "avg_return": round(float(returns.mean()) * 100, 2),
        "median_return": round(float(returns.median()) * 100, 2),
    }


def run_backtest(
    symbol_ticker_pairs,
    period="3y",
    lookback_days=5,
    horizons=DEFAULT_HORIZONS,
    min_history=250,
    progress_cb=None,
):
    """
    symbol_ticker_pairs: iterable of (symbol, ticker) tuples.

    Returns {
      "per_bucket": {bucket_name: {horizon_days: {n, hit_rate, avg_return, median_return}}},
      "tickers_used": int,
      "tickers_skipped": [symbol, ...],   # insufficient history
    }

    Bucket "bullish" = the combined live screener signal. "baseline" = every
    valid day regardless of signal, for comparison. Remaining buckets are the
    individual signals in SIGNAL_COLUMNS, so you can see which ones actually
    carry predictive value vs. which just add noise to the combined score.
    """
    pairs = list(symbol_ticker_pairs)
    all_buckets = []
    skipped = []
    total = len(pairs)
    for i, (symbol, ticker) in enumerate(pairs, start=1):
        b = backtest_ticker(
            ticker, period=period, lookback_days=lookback_days, horizons=horizons, min_history=min_history
        )
        if b is None:
            skipped.append(symbol)
        else:
            all_buckets.append(b)
        if progress_cb:
            progress_cb(i, total, symbol)

    merged = _merge(all_buckets, horizons)
    per_bucket = {
        bucket: {h: _summarize(series) for h, series in by_h.items()} for bucket, by_h in merged.items()
    }
    return {
        "per_bucket": per_bucket,
        "tickers_used": len(all_buckets),
        "tickers_skipped": skipped,
    }


def results_to_dataframe(results: dict, horizons=DEFAULT_HORIZONS) -> pd.DataFrame:
    """Flatten run_backtest() output into a table: one row per bucket, one
    column pair (hit rate, avg return) per horizon - handy for display."""
    rows = []
    bucket_order = ["bullish"] + SIGNAL_COLUMNS + ["baseline"]
    for bucket in bucket_order:
        if bucket not in results["per_bucket"]:
            continue
        row = {"signal": bucket}
        for h in horizons:
            stats = results["per_bucket"][bucket][h]
            row[f"n ({h}d)"] = stats["n"]
            row[f"hit rate % ({h}d)"] = stats["hit_rate"]
            row[f"avg return % ({h}d)"] = stats["avg_return"]
        rows.append(row)
    return pd.DataFrame(rows)
