# Scribbler — Indian Stock Screener (Bullish Patterns + In the News)

Streamlit app that scans a watchlist of Indian (NSE) stocks for bullish technical
signals and cross-references free market news for recent mentions and sentiment.

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
Then open the URL it prints (usually http://localhost:8501). The in-app **Docs**
tab explains the methodology (technical signals + weights, news matching,
sentiment, backtest) in detail, sourced live from the running code.

## Tabs
- **Screener**: pick a watchlist, run the scan, see ranked results with
  signals, confirmed/possible news, and per-stock stats.
- **Backtest**: checks whether each technical signal actually precedes
  above-average forward returns (not just whether it looks plausible).
- **Docs**: methodology reference — see this first if something's unclear.

## Watchlist
Default is Nifty 50. Use the **Index universe** dropdown + "Load Index" in the
sidebar to switch to Nifty 500 or "All" (NSE's Nifty Total Market, ~750
stocks — fetched live from NSE, no manual download needed; falls back to a
bundled snapshot if NSE is unreachable). Larger universes take noticeably
longer to screen. You can still upload your own CSV (`symbol,aliases`
columns), edit the table directly, or click "Save as default" to persist.

## Notes / limitations
- Price data via `yfinance` — free but can lag or occasionally fail for a given
  ticker; those tickers are silently skipped from results.
- News coverage via free RSS is sparse — if nothing matches, uncheck "Require
  news mention" in the sidebar to see technically-bullish stocks regardless of
  news coverage. Only headlines within the last 60 days are considered.
- News matching and sentiment both use ML (sentence embeddings / FinBERT) with
  automatic fallback to simpler methods if the models can't load — see the
  Docs tab for exactly how, including a known limitation (semantic matches
  are shown as "unconfirmed" and excluded from scoring; see Docs for why).
- Not investment advice.
