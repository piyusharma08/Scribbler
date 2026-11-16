# Scribbler — Indian Stock Bullish-Pattern + News Screener

## Context
Application that scans a watchlist of Indian (NSE) stocks and surfaces ones that are (a) showing bullish technical patterns and (b) currently in the news, with ML-based news matching and sentiment (both with automatic fallbacks), a backtest tool to validate the technical signals against real forward returns, and an in-app Docs tab. Streamlit web dashboard, `yfinance` for price/volume data, free RSS feeds for news, NSE archives for index-constituent watchlists — no paid API keys required.

## Project layout
```
Scribbler/
  app.py                 # Streamlit entrypoint — 3 tabs: Screener, Backtest, Docs
  screener/
    __init__.py
    universe.py           # NSE index watchlists (Nifty 50/500/Total Market) + aliases
    prices.py             # yfinance price history + per-stock info, caching
    technical.py          # vectorized bullish signal table + live scoring
    backtest.py           # replays signals over history vs. realized forward returns
    news.py               # RSS fetch+recency filter, ML news matching, ML sentiment
    scoring.py            # combine technical + news into ranked results
    data/                 # bundled fallback CSVs for each index tier
  requirements.txt
  watchlist.csv           # saved custom watchlist (optional, via "Save as default")
  README.md
```

## Core logic

**universe.py**
- `INDEX_OPTIONS`: Nifty 50 / Nifty 500 / "All" (Nifty Total Market, ~750 stocks) — fetched live from `archives.nseindia.com`, each giving `symbol` + `aliases` (derived from the real company name, since that's more useful for news matching than a bare ticker).
- `fetch_index_constituents(index_key)`: live fetch, cached 24h; falls back to a bundled snapshot CSV in `screener/data/` on any network failure.
- `ABBREVIATION_OVERRIDES`: known short-form aliases NSE's company-name column doesn't capture (e.g. "SBI", "HUL", "L&T"), merged in for Nifty-50 symbols.

**prices.py**
- `fetch_history(ticker, period, interval)` via `yfinance`, cached 15 min.
- `fetch_many(tickers, ...)`: `ThreadPoolExecutor`-based concurrent fetch with progress callback.
- `fetch_info(ticker)`: curated subset of `yfinance` `Ticker.info` (price, day change %, market cap, P/E, 52W range, avg volume, sector, dividend yield, beta) for the stock-detail view; missing fields degrade to `None`/"N/A", never an error.

**technical.py** — vectorized, causal (no look-ahead) signal table, reused identically by both live scoring and the backtest:
- `compute_signal_table(df)`: golden cross, uptrend-only, RSI healthy & rising, MACD bullish crossover, MACD-above-only, volume spike (≥1.5x 20d avg), 20-day breakout — each a boolean column plus a weighted `score` and `bullish` flag.
- `WEIGHTS` and `BULLISH_THRESHOLD` were **empirically retuned** via `backtest.py` against 5 years of Nifty-50 history (not guessed): signals with a consistent edge over baseline (golden_cross, volume_spike, macd_above_only) weighted up; signals with a flipped/no edge (rsi_healthy_rising, macd_bullish_cross, breakout_20d) weighted down; `uptrend_only` weighted to 0 (negative edge — fires on already-extended trends).
- `analyze_technical(df)`: live-scoring wrapper over the last row of the signal table, plus an optional bonus for bullish candlestick patterns if `pandas-ta` is installed (not included in default requirements; unavailable on some pip indices).

**backtest.py**
- Replays `compute_signal_table` over a ticker's full history; for every day a signal fired, records the realized forward return N days later (default horizons 5/10/20).
- Compares each signal's hit-rate/avg-return against an unconditional baseline (every valid day) to determine which signals actually carry predictive value — this is how the current `WEIGHTS` were chosen.

**news.py**
- `fetch_all_headlines()`: pulls configured RSS feeds, drops anything older than `NEWS_MAX_AGE_DAYS` (60) or with no parseable publish date — also catches feeds silently serving stale/cached content instead of live updates (found in testing). Each kept headline gets a human-readable `published_display` (e.g. "3 days ago").
- **Matching**: word-boundary regex (exact symbol/alias match) is the trusted "confirmed" signal, used for `mentioned` and sentiment scoring. A sentence-embedding model (`all-MiniLM-L6-v2`) runs alongside it to surface additional "possible" (unconfirmed) matches — kept separate and excluded from scoring, because testing found generic embeddings can't reliably distinguish similarly-named but unrelated companies (e.g. "Reliance Industries" vs "MAN Industries" scored similarly). Falls back to regex-only if the embedding model can't load.
- **Sentiment**: FinBERT (`ProsusAI/finbert`) scores *every* fetched headline individually (confirmed and possible alike get `sentiment_score`/`sentiment_label` attached directly on the headline dict, via `sentiment_label()`), normalized to a signed `[-1, 1]` score; falls back to a keyword lexicon (normalized to the same scale) if the model can't load. Each stock's aggregate `sentiment_score`/`sentiment_label` is averaged from its *confirmed* headlines only.
- Returns `dict(symbol -> dict(mentioned, headlines, possible_headlines, sentiment_score, sentiment_label, method, sentiment_method))` — headlines within `headlines`/`possible_headlines` each carry their own `sentiment_score`/`sentiment_label` too, so the UI can show FinBERT's actual per-article output, not just the aggregate.

**scoring.py**
- `final_score = WEIGHT_TECHNICAL (0.6) * technical_score + WEIGHT_NEWS (0.4) * news_score`.
- Default filter: technical bullish AND mentioned in confirmed news (toggle to relax the news requirement, since free RSS coverage is sparse).
- `combine()` passes through `sentiment_score`/`sentiment_label` alongside `news_score`, so the raw sentiment reading is visible in the UI, not just the derived 0-100 component.

**app.py (Streamlit UI)** — 3 tabs:
- **Screener**: sidebar watchlist editor (index-tier dropdown with live NSE fetch, CSV upload, manual edit, save-as-default), lookback period, "require news mention" toggle, min technical score. Results table (with a 🟢/🔴/⚪ Sentiment column) + per-stock expander with key-stats markdown table, candlestick chart, confirmed headlines (each tagged with its own sentiment label/score), and clearly-labeled unconfirmed semantic matches. A caption above the results states exactly which sidebar settings produced the currently-displayed run, and warns if the sidebar has since changed without clicking "Run Screener" again — toggling a widget alone doesn't rescan, only the button does, which otherwise reads as a stale/stuck-result bug.
- **Backtest**: runs `backtest.py` over the current watchlist, shows hit-rate/avg-return per signal vs. baseline.
- **Docs**: a numbered reference — (1) overview, (2) glossary (every acronym/term spelled out: RSI, MACD, SMA, EMA, OHLCV, FinBERT, cosine similarity, etc.), (3)/(4) step-by-step "how to use" for the Screener and Backtest tabs, (5) news matching & sentiment mechanics, (6) technical signals + live weights/threshold/formula, (7) a clearly-separated "methodology notes" section for the backtest-driven rationale behind the weights — kept apart from the glossary/how-to so the reference stays general rather than narrating one specific tuning session.

## Dependencies
`streamlit`, `yfinance`, `pandas`, `plotly`, `feedparser`, `requests`, `sentence-transformers`, `transformers` (pulls in `torch` — first run downloads the embedding model + FinBERT, ~500MB combined). Optional: `pandas-ta` for bonus candlestick signals (not installed by default).

## Verification
1. `pip install -r requirements.txt` in a venv.
2. `streamlit run app.py` — UI loads with default Nifty 50 watchlist.
3. "Run Screener" fetches data without errors, shows ranked table with detected signals, sentiment column, and key-stats.
4. Switch "Index universe" to Nifty 500 / All, confirm the watchlist repopulates and a scan still runs (longer for larger tiers).
5. News matching returns confirmed headlines for frequently-covered large-caps; unconfirmed semantic matches are visibly separate; each headline shows its own FinBERT sentiment label/score.
6. Toggle "Require news mention" without re-running; confirm the stale-settings warning appears, then confirm it clears after clicking "Run Screener" again.
7. Run Backtest; confirm the combined `bullish` signal beats `baseline` at the 20-day horizon.
8. Check the Docs tab reflects the live weights/thresholds (e.g. change a weight in `technical.py`, confirm Docs updates without touching doc text).
