"""Bullish technical screen: indicators computed with plain pandas (robust, no
heavy TA dependency for the core signals) plus optional candlestick patterns
from pandas-ta as a bonus signal.

The signal logic is vectorized in `compute_signal_table` so the exact same
day-by-day rules used for live scoring (`analyze_technical`) can also be
replayed over full history for backtesting (`screener/backtest.py`) without
the two ever drifting apart.
"""
import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
    _HAS_PANDAS_TA = True
except Exception:
    _HAS_PANDAS_TA = False

# Signal name -> score contribution, re-tuned from a 5y/49-ticker Nifty-50
# backtest (screener/backtest.py) measuring each signal's forward-return edge
# over an unconditional baseline at 5/10/20-day horizons:
#   golden_cross / volume_spike: consistently positive edge at every horizon
#     (both hit-rate and avg return) -> weighted up.
#   macd_above_only: smaller but consistently positive edge -> weighted up slightly.
#   rsi_healthy_rising / macd_bullish_cross / breakout_20d: edge flips sign
#     across horizons (noise, not signal) -> weighted down.
#   uptrend_only: *negative* edge at every horizon (fires on already-extended
#     trends) -> weight dropped to 0; still shown as a label, just not scored.
WEIGHTS = {
    "golden_cross": 35,
    "uptrend_only": 0,
    "rsi_healthy_rising": 10,
    "macd_bullish_cross": 10,
    "macd_above_only": 15,
    "volume_spike": 25,
    "breakout_20d": 5,
}
SIGNAL_COLUMNS = list(WEIGHTS.keys())

# Calibrated against the same backtest: edge over baseline is broadly flat
# from threshold 25-45, with hit-rate/avg-return quality peaking ~35-40 while
# sample size (selectivity) drops as the threshold rises. 35 keeps near-peak
# edge (57.1% hit-rate / +1.33% avg return @20d vs. baseline 55.8%/1.08%)
# while retaining more usable results than 40.
BULLISH_THRESHOLD = 35

LABELS = {
    "golden_cross": "Golden cross (50d SMA crossed above 200d SMA)",
    "uptrend_only": "Uptrend (50d SMA above 200d SMA)",
    "macd_bullish_cross": "MACD bullish crossover",
    "macd_above_only": "MACD above signal line",
    "breakout_20d": "Breakout to 20d high",
}


def _sma(series, window):
    return series.rolling(window).mean()


def _rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def compute_signal_table(df: pd.DataFrame, lookback_days: int = 5, bullish_threshold: int = BULLISH_THRESHOLD) -> pd.DataFrame:
    """Vectorized, causal (no look-ahead) day-by-day signal table.

    Every column at row i depends only on data up to and including row i, so
    this can be replayed historically for backtesting as well as evaluated
    only on the last row for live scoring.

    Returns a DataFrame indexed like `df` with one boolean column per entry
    in SIGNAL_COLUMNS, plus raw `rsi`, `vol_ratio`, `score`, and `bullish`.
    """
    close = df["Close"]
    volume = df["Volume"]

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    above_now = sma50 > sma200
    was_below = sma50.shift(lookback_days) <= sma200.shift(lookback_days)
    golden_cross = above_now & was_below
    uptrend_only = above_now & ~golden_cross

    rsi = _rsi(close, 14)
    rsi_rising = rsi > rsi.shift(lookback_days)
    rsi_healthy_rising = (rsi >= 40) & (rsi <= 70) & rsi_rising

    macd_line, signal_line = _macd(close)
    macd_bullish_cross = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    macd_above_only = (macd_line > signal_line) & ~macd_bullish_cross

    avg_vol20 = volume.rolling(20).mean()
    vol_ratio = volume / avg_vol20.replace(0, np.nan)
    volume_spike = vol_ratio >= 1.5

    high20 = close.rolling(20).max()
    breakout_20d = close >= high20.shift(1)

    table = pd.DataFrame(
        {
            "golden_cross": golden_cross,
            "uptrend_only": uptrend_only,
            "rsi_healthy_rising": rsi_healthy_rising,
            "macd_bullish_cross": macd_bullish_cross,
            "macd_above_only": macd_above_only,
            "volume_spike": volume_spike,
            "breakout_20d": breakout_20d,
            "rsi": rsi,
            "vol_ratio": vol_ratio,
        },
        index=df.index,
    )
    for col in SIGNAL_COLUMNS:
        table[col] = table[col].fillna(False)

    score = sum(table[col].astype(int) * weight for col, weight in WEIGHTS.items())
    table["score"] = score.clip(upper=100)
    table["bullish"] = table["score"] >= bullish_threshold
    return table


def analyze_technical(df: pd.DataFrame, lookback_days: int = 5) -> dict:
    """Run the bullish rule set over an OHLCV dataframe, evaluated on the latest day.

    Returns {"score": 0-100, "signals": [str, ...], "bullish": bool}
    """
    table = compute_signal_table(df, lookback_days=lookback_days)
    last = table.iloc[-1]
    signals = []

    if last["golden_cross"]:
        signals.append(LABELS["golden_cross"])
    elif last["uptrend_only"]:
        signals.append(LABELS["uptrend_only"])

    if last["rsi_healthy_rising"]:
        signals.append(f"RSI healthy & rising ({last['rsi']:.0f})")

    if last["macd_bullish_cross"]:
        signals.append(LABELS["macd_bullish_cross"])
    elif last["macd_above_only"]:
        signals.append(LABELS["macd_above_only"])

    if last["volume_spike"]:
        signals.append(f"Volume spike ({last['vol_ratio']:.1f}x 20d avg)")

    if last["breakout_20d"]:
        signals.append(LABELS["breakout_20d"])

    score = float(last["score"])

    if _HAS_PANDAS_TA:
        try:
            patterns_df = pd.DataFrame(
                {"open": df["Open"], "high": df["High"], "low": df["Low"], "close": df["Close"]}
            )
            bullish_patterns = ["engulfing", "hammer", "morningstar"]
            for p in bullish_patterns:
                col = ta.cdl_pattern(
                    open_=patterns_df["open"],
                    high=patterns_df["high"],
                    low=patterns_df["low"],
                    close=patterns_df["close"],
                    name=p,
                )
                if col is not None:
                    series = col.iloc[:, 0] if hasattr(col, "iloc") and col.ndim == 2 else col
                    if series.iloc[-1] > 0:
                        signals.append(f"Bullish candlestick: {p}")
                        score += 10
        except Exception:
            pass

    score = min(score, 100)
    return {"score": score, "signals": signals, "bullish": score >= BULLISH_THRESHOLD}
