"""Bullish technical screen: indicators computed with plain pandas (robust, no
heavy TA dependency for the core signals) plus optional candlestick patterns
from pandas-ta as a bonus signal.
"""
import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
    _HAS_PANDAS_TA = True
except Exception:
    _HAS_PANDAS_TA = False


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


def analyze_technical(df: pd.DataFrame, lookback_days: int = 5) -> dict:
    """Run the bullish rule set over an OHLCV dataframe.

    Returns {"score": 0-100, "signals": [str, ...], "bullish": bool}
    """
    close = df["Close"]
    volume = df["Volume"]
    signals = []
    score = 0

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    if len(close) >= 200 and not sma50.isna().iloc[-1] and not sma200.isna().iloc[-1]:
        above_now = sma50.iloc[-1] > sma200.iloc[-1]
        was_below = (sma50.iloc[-lookback_days - 1] <= sma200.iloc[-lookback_days - 1])
        if above_now and was_below:
            signals.append("Golden cross (50d SMA crossed above 200d SMA)")
            score += 30
        elif above_now:
            signals.append("Uptrend (50d SMA above 200d SMA)")
            score += 10

    rsi = _rsi(close, 14)
    rsi_now = rsi.iloc[-1]
    rsi_rising = rsi.iloc[-1] > rsi.iloc[-min(lookback_days, len(rsi) - 1)]
    if 40 <= rsi_now <= 70 and rsi_rising:
        signals.append(f"RSI healthy & rising ({rsi_now:.0f})")
        score += 20

    macd_line, signal_line = _macd(close)
    crossed_up = (
        macd_line.iloc[-1] > signal_line.iloc[-1]
        and macd_line.iloc[-2] <= signal_line.iloc[-2]
    )
    if crossed_up:
        signals.append("MACD bullish crossover")
        score += 25
    elif macd_line.iloc[-1] > signal_line.iloc[-1]:
        signals.append("MACD above signal line")
        score += 10

    avg_vol20 = volume.rolling(20).mean()
    if not avg_vol20.isna().iloc[-1] and avg_vol20.iloc[-1] > 0:
        vol_ratio = volume.iloc[-1] / avg_vol20.iloc[-1]
        if vol_ratio >= 1.5:
            signals.append(f"Volume spike ({vol_ratio:.1f}x 20d avg)")
            score += 15

    high20 = close.rolling(20).max()
    if close.iloc[-1] >= high20.iloc[-2]:
        signals.append("Breakout to 20d high")
        score += 15

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
    return {"score": score, "signals": signals, "bullish": score >= 40}
