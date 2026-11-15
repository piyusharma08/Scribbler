# Scribbler — Indian Stock Screener (Bullish Patterns + In the News)

Streamlit app that scans a watchlist of Indian (NSE) stocks for bullish technical
signals and cross-references free market RSS feeds for news mentions.

## Setup
```
cd Scribbler
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run
```
streamlit run app.py
```
Then open the URL it prints (usually http://localhost:8501).

## How it works
- **Technical score**: golden cross (50d/200d SMA), RSI momentum, MACD crossover,
  volume spike vs 20d average, breakout above 20d high, and bullish candlestick
  patterns (via `pandas-ta` if available). See `screener/technical.py`.
- **News**: pulls headlines from Moneycontrol / Economic Times / Business Standard
  RSS feeds, matches them to watchlist symbols/company names, and scores sentiment
  with a small keyword lexicon (not an ML model — directional only). See
  `screener/news.py`.
- **Final score**: `0.6 * technical_score + 0.4 * news_score`, filtered to only
  show technically-bullish stocks (optionally also requiring a news mention).

## Watchlist
Default is Nifty 50 (`watchlist.csv`). Edit directly in the sidebar table, upload
your own CSV (`symbol,aliases` columns), or click "Save as default" to persist
your edits to `watchlist.csv`.

## Notes / limitations
- Price data via `yfinance` — free but can lag or occasionally fail for a given
  ticker; those tickers are silently skipped from results.
- News coverage via free RSS is sparse — if nothing matches, uncheck "Require
  news mention" in the sidebar to see technically-bullish stocks regardless of
  news coverage.
- Sentiment scoring is a simple positive/negative keyword count, intended to bias
  ranking, not as investment advice.
