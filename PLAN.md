# Scribbler — Indian Stock Bullish-Pattern + News Screener

## Context
Application that scans a user-editable watchlist of Indian stocks and surfaces ones that are (a) showing bullish technical patterns and (b) currently in the news. Streamlit web dashboard, `yfinance` for price/volume data, free RSS feeds (Moneycontrol/Economic Times/Business Standard) for news — no paid API keys required. Default watchlist: Nifty 50 tickers, user-editable in the UI.

## Project layout
```
Scribbler/
  app.py                 # Streamlit entrypoint — UI, orchestration
  screener/
    __init__.py
    universe.py           # default Nifty 50 ticker list + watchlist load/save (CSV)
    prices.py             # yfinance fetch + caching
    technical.py          # indicator calcs + bullish pattern rules
    news.py               # RSS fetch, ticker-matching, simple sentiment scoring
    scoring.py            # combine technical + news into ranked results
  requirements.txt
  watchlist.csv           # default Nifty 50 symbols (editable by user via UI)
  README.md               # how to run it
```

## Core logic

**prices.py**
- `fetch_history(ticker, period="6mo", interval="1d")` using `yfinance.download`, with `st.cache_data(ttl=...)` caching (15 min).
- Tickers use `.NS` suffix (NSE) by default.

**technical.py** — bullish screen, using `pandas`/`pandas-ta` on OHLCV:
- Golden cross: 50-SMA crossed above 200-SMA recently (or price above both, trending up).
- RSI(14) between 40-70 and rising.
- MACD bullish crossover (MACD line crossed above signal line in last N days).
- Volume spike: last day volume > 1.5x the 20-day average volume.
- Price near recent high / breakout above N-day resistance (e.g., 20-day high).
- Candlestick bullish patterns via `pandas-ta` candle functions (engulfing, hammer, morning star) as bonus signal.
- `analyze_technical(df) -> dict(score, signals_list)`.

**news.py**
- Pull free RSS feeds (Moneycontrol markets, ET Markets, Business Standard markets) via `feedparser`.
- Match company name/symbol against headline text using a symbol→company-name alias map.
- Keyword-lexicon sentiment scoring (heuristic, not ML/NLP-grade — documented as such).
- Returns `dict(ticker -> dict(mentioned, headlines, sentiment_score))`.

**scoring.py**
- `final_score = 0.6 * technical_score + 0.4 * news_score`.
- Default filter: technical bullish AND mentioned in news (toggle to relax news requirement since RSS coverage is sparse).

**app.py (Streamlit UI)**
- Sidebar: editable watchlist (text area / CSV upload, default Nifty 50), lookback period, "require news mention" toggle, refresh button.
- Main: ranked, sortable results table — Ticker, Technical Score, Signals, News headlines, Sentiment, Final Score.
- Expander per row: candlestick chart (plotly) with signals annotated, matched headlines with links.
- ThreadPoolExecutor for concurrent price fetches.

## Dependencies
`streamlit`, `yfinance`, `pandas`, `pandas-ta`, `plotly`, `feedparser`

## Verification
1. `pip install -r requirements.txt` in a venv.
2. `streamlit run app.py` — UI loads with default Nifty 50 watchlist.
3. "Run Screener" fetches data without errors, shows ranked table with detected signals.
4. News matching returns headlines for frequently-covered large-caps (RELIANCE, TCS).
5. Toggle "require news mention" changes result set as expected.
6. Spot-check one flagged stock's chart against its claimed signal (e.g. golden cross).
