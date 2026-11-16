import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from screener import universe, prices, technical, news, scoring, backtest

st.set_page_config(page_title="Scribbler", layout="wide")

st.title("Scribbler — Indian Stock Screener (Bullish Patterns + In the News)")
st.caption(
    "Scans your watchlist for bullish technical signals (golden cross, RSI, MACD, "
    "volume breakout, candlesticks) and cross-references recent market news for "
    "mentions and sentiment (ML-based, with fallbacks — see the Docs tab). Treat "
    "scores as directional, not investment advice."
)

# ---------- Sidebar ----------
st.sidebar.header("Watchlist")

if "watchlist_df" not in st.session_state:
    st.session_state.watchlist_df = universe.load_watchlist()

uploaded = st.sidebar.file_uploader("Upload watchlist CSV (columns: symbol, aliases)", type="csv")
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        if "symbol" not in df.columns:
            st.sidebar.error("CSV must have a 'symbol' column.")
        else:
            if "aliases" not in df.columns:
                df["aliases"] = df["symbol"]
            st.session_state.watchlist_df = df
            st.sidebar.success(f"Loaded {len(df)} symbols.")
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")

watchlist_df = st.sidebar.data_editor(
    st.session_state.watchlist_df,
    num_rows="dynamic",
    use_container_width=True,
    key="watchlist_editor",
)
st.session_state.watchlist_df = watchlist_df

index_choice = st.sidebar.selectbox("Index universe", list(universe.INDEX_OPTIONS.keys()))
if "All" in index_choice:
    st.sidebar.caption("⚠️ ~750 stocks — a full screener/backtest run can take several minutes.")
elif index_choice == "Nifty 500":
    st.sidebar.caption("⚠️ ~500 stocks — a full run will take noticeably longer than Nifty 50.")

col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("Load Index"):
        with st.spinner(f"Fetching {index_choice} constituents..."):
            st.session_state.watchlist_df = universe.fetch_index_constituents(index_choice)
        st.rerun()
with col_b:
    if st.button("Save as default"):
        universe.save_watchlist(st.session_state.watchlist_df)
        st.sidebar.success("Saved.")

st.sidebar.header("Screener Options")
period = st.sidebar.selectbox("Price history lookback", ["3mo", "6mo", "1y"], index=1)
require_news = st.sidebar.checkbox("Require news mention (strict)", value=True)
min_score = st.sidebar.slider("Minimum technical score", 0, 100, 35, step=5)

run = st.sidebar.button("Run Screener", type="primary")

st.sidebar.header("Backtest Options")
bt_period = st.sidebar.selectbox("Backtest history", ["1y", "3y", "5y"], index=1, key="bt_period")
bt_horizons_str = st.sidebar.text_input("Forward horizons (days, comma-separated)", "5,10,20")
run_bt = st.sidebar.button("Run Backtest")

# ---------- Main ----------
screener_tab, backtest_tab, docs_tab = st.tabs(["Screener", "Backtest", "Docs"])

with screener_tab:
    if run:
        wl = st.session_state.watchlist_df.dropna(subset=["symbol"])
        symbols = wl["symbol"].astype(str).str.strip().str.upper().tolist()
        aliases_map = {
            str(row["symbol"]).strip().upper(): str(row.get("aliases", row["symbol"])).split(";")
            for _, row in wl.iterrows()
        }
        tickers = [universe.ticker_for(s) for s in symbols]
        ticker_to_symbol = dict(zip(tickers, symbols))

        progress = st.progress(0.0, text="Fetching price history...")

        def on_progress(done, total, _ticker):
            progress.progress(done / total, text=f"Fetched {done}/{total} tickers...")

        price_data = prices.fetch_many(tickers, period=period, progress_cb=on_progress)
        progress.empty()

        with st.spinner("Checking news feeds..."):
            news_rows = [(s, aliases_map[s]) for s in symbols]
            news_results = news.match_news_for_watchlist(news_rows)

        first_result = next(iter(news_results.values()), {})
        st.session_state.last_news_method = first_result.get("method", "regex-only")
        st.session_state.last_sentiment_method = first_result.get("sentiment_method", "keyword-fallback")

        results = []
        for ticker in tickers:
            symbol = ticker_to_symbol[ticker]
            df = price_data.get(ticker)
            if df is None:
                continue
            tech_result = technical.analyze_technical(df)
            row = scoring.combine(symbol, tech_result, news_results.get(symbol), require_news)
            if row and row["technical_score"] >= min_score:
                results.append(row)

        ranked = scoring.rank_results(results)
        st.session_state.last_results = ranked
        st.session_state.last_price_data = price_data
        st.session_state.last_ticker_map = ticker_to_symbol
        st.session_state.last_run_settings = {
            "require_news": require_news,
            "min_score": min_score,
            "period": period,
            "universe_size": len(symbols),
        }

    if "last_results" in st.session_state:
        ranked = st.session_state.last_results

        settings = st.session_state.get("last_run_settings", {})
        is_stale = settings and (
            settings.get("require_news") != require_news
            or settings.get("min_score") != min_score
            or settings.get("period") != period
        )
        if settings:
            st.caption(
                f"Showing results from the last **Run Screener** click — "
                f"require_news={settings.get('require_news')}, min_score={settings.get('min_score')}, "
                f"period={settings.get('period')}, universe={settings.get('universe_size')} stocks."
                + (" **⚠️ Sidebar settings have changed since then — click Run Screener to refresh.**" if is_stale else "")
            )

        if st.session_state.get("last_news_method") == "regex-only":
            st.caption(
                "⚠️ Semantic (ML) news matching unavailable this run (model failed to load) "
                "— fell back to exact word/alias matching."
            )
        else:
            st.caption("News matching: semantic (ML) similarity + exact word/alias matching.")

        if st.session_state.get("last_sentiment_method") == "keyword-fallback":
            st.caption(
                "⚠️ FinBERT sentiment model unavailable this run — fell back to keyword lexicon."
            )
        else:
            st.caption("Sentiment: FinBERT (finance-domain ML model).")
        st.subheader(f"Results: {len(ranked)} stock(s) matched")

        if not ranked:
            st.info(
                "No stocks matched. Try unchecking 'Require news mention' (RSS coverage is "
                "sparse) or lowering the minimum technical score."
            )
        else:
            _sentiment_emoji = {"positive": "🟢 Positive", "negative": "🔴 Negative", "neutral": "⚪ Neutral"}
            table = pd.DataFrame(
                [
                    {
                        "Symbol": r["symbol"],
                        "Technical Score": r["technical_score"],
                        "In News": "Yes" if r["news_mentioned"] else "No",
                        "Sentiment": _sentiment_emoji.get(r["sentiment_label"], "—") if r["news_mentioned"] else "—",
                        "News Score": r["news_score"],
                        "Final Score": r["final_score"],
                        "Signals": ", ".join(r["signals"]),
                    }
                    for r in ranked
                ]
            )
            st.dataframe(table, use_container_width=True, hide_index=True)

            st.subheader("Details")
            for r in ranked:
                with st.expander(f"{r['symbol']} — final score {r['final_score']}"):
                    st.write("**Signals:**", ", ".join(r["signals"]) or "None")
                    ticker = universe.ticker_for(r["symbol"])

                    info = prices.fetch_info(ticker)

                    def _fmt(key, fmt="{}"):
                        v = info.get(key)
                        return fmt.format(v) if v is not None else "N/A"

                    stats_md = (
                        f"| Price | Day Chg % | Market Cap | P/E (TTM) | 52W Range | Avg Volume | Sector | Dividend Yield | Beta |\n"
                        f"|---|---|---|---|---|---|---|---|---|\n"
                        f"| {_fmt('Current Price', '₹{:.2f}')} "
                        f"| {_fmt('Day Change %', '{:+.2f}%')} "
                        f"| {_fmt('Market Cap', '₹{:,.0f}')} "
                        f"| {_fmt('P/E (TTM)', '{:.1f}')} "
                        f"| {_fmt('52W Low', '{:.0f}')}–{_fmt('52W High', '{:.0f}')} "
                        f"| {_fmt('Avg Volume', '{:,.0f}')} "
                        f"| {_fmt('Sector')} "
                        f"| {_fmt('Dividend Yield', '{:.2f}%')} "
                        f"| {_fmt('Beta', '{:.2f}')} |"
                    )
                    st.markdown(stats_md)

                    df = st.session_state.last_price_data.get(ticker)
                    if df is not None:
                        fig = go.Figure(
                            data=[
                                go.Candlestick(
                                    x=df.index,
                                    open=df["Open"],
                                    high=df["High"],
                                    low=df["Low"],
                                    close=df["Close"],
                                )
                            ]
                        )
                        fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig, use_container_width=True)

                    _sentiment_tag = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}
                    if r["headlines"]:
                        st.write("**Matched headlines:**")
                        for h in r["headlines"]:
                            tag = _sentiment_tag.get(h.get("sentiment_label"), "")
                            st.markdown(
                                f"- {tag} [{h['title']}]({h['link']}) "
                                f"_{h.get('published_display', h.get('published', ''))}_ "
                                f"— {h.get('sentiment_label', 'n/a')} ({h.get('sentiment_score', 0):+.2f})"
                            )
                    else:
                        st.write("No matched headlines in current RSS feeds.")

                    if r.get("possible_headlines"):
                        st.caption(
                            "Possible mentions (ML semantic match, unconfirmed — not used in "
                            "scoring; may be unrelated companies with similar names):"
                        )
                        for h in r["possible_headlines"]:
                            tag = _sentiment_tag.get(h.get("sentiment_label"), "")
                            st.markdown(f"- {tag} _{h['title']}_ — {h.get('sentiment_label', 'n/a')}")
    else:
        st.info("Configure your watchlist in the sidebar and click **Run Screener** to begin.")

with backtest_tab:
    st.caption(
        "Replays each technical signal over historical data and compares realized "
        "forward returns against an unconditional baseline (every day, signal or "
        "not). A signal with a higher hit-rate / avg return than baseline is "
        "actually informative; one that matches baseline is just noise in the "
        "combined score. No position sizing or exits modeled — this measures "
        "signal quality, not trading performance."
    )

    if run_bt:
        try:
            horizons = tuple(int(h.strip()) for h in bt_horizons_str.split(",") if h.strip())
        except ValueError:
            st.error("Forward horizons must be comma-separated integers, e.g. 5,10,20")
            horizons = None

        if horizons:
            wl = st.session_state.watchlist_df.dropna(subset=["symbol"])
            symbols = wl["symbol"].astype(str).str.strip().str.upper().tolist()
            pairs = [(s, universe.ticker_for(s)) for s in symbols]

            progress = st.progress(0.0, text="Backtesting...")

            def on_bt_progress(done, total, symbol):
                progress.progress(done / total, text=f"Backtested {done}/{total}: {symbol}")

            bt_results = backtest.run_backtest(
                pairs, period=bt_period, horizons=horizons, progress_cb=on_bt_progress
            )
            progress.empty()
            st.session_state.last_backtest = bt_results
            st.session_state.last_backtest_horizons = horizons

    if "last_backtest" in st.session_state:
        bt_results = st.session_state.last_backtest
        horizons = st.session_state.last_backtest_horizons
        st.write(
            f"Backtested **{bt_results['tickers_used']}** ticker(s) over `{bt_period}`."
            + (f" Skipped (insufficient history): {', '.join(bt_results['tickers_skipped'])}."
               if bt_results["tickers_skipped"] else "")
        )
        bt_df = backtest.results_to_dataframe(bt_results, horizons=horizons)
        st.dataframe(bt_df, use_container_width=True, hide_index=True)
        st.caption(
            "`bullish` = the combined live screener signal. `baseline` = every "
            "valid day regardless of signal. Compare each individual signal's "
            "hit rate / avg return against `baseline` to see which actually carry "
            "predictive value for this watchlist."
        )
    else:
        st.info("Click **Run Backtest** in the sidebar to evaluate signal quality on your watchlist.")

with docs_tab:
    st.header("1. Overview")
    st.markdown(
        """
Scribbler scans a watchlist of Indian (NSE) stocks for **bullish technical signals** and
cross-references recent market news to show which of those stocks are currently in the news,
with a sentiment reading.

**Run it**: `streamlit run app.py` from the project directory (see `README.md` for environment
setup / dependency install).
        """
    )

    st.header("2. Glossary")
    st.markdown("Terms used throughout the app, in plain language.")
    glossary = [
        ("NSE", "National Stock Exchange of India — the exchange this app's tickers/prices come from (symbols suffixed `.NS`)."),
        ("OHLCV", "Open, High, Low, Close, Volume — the standard daily price/volume data a stock chart is built from."),
        ("SMA", "Simple Moving Average — the average closing price over the last N days. Smooths out day-to-day noise to show trend direction."),
        ("EMA", "Exponential Moving Average — like an SMA, but weights recent days more heavily. Used inside MACD."),
        ("RSI", "Relative Strength Index — a 0-100 momentum indicator based on recent gains vs. losses. Below ~30 is usually called 'oversold', above ~70 'overbought'."),
        ("MACD", "Moving Average Convergence Divergence — the difference between a fast and a slow EMA (the 'MACD line'), compared against its own smoothed average (the 'signal line'). Used to spot momentum turns."),
        ("Golden cross", "When a shorter-term moving average (50-day) crosses above a longer-term one (200-day) — a classic trend-change signal."),
        ("Bullish", "Expected/showing upward price movement. (Opposite: bearish.)"),
        ("Technical score / signal", "A score (0-100) and the list of price/volume-based rules that fired for a stock — see Section 4."),
        ("Backtest", "Replaying a rule over historical data to check whether it actually worked in the past, rather than just assuming it does — see Section 6."),
        ("Forward return", "The % price change from a given day to N days later — what a backtest measures to judge a signal."),
        ("Hit rate", "In a backtest, the % of occurrences where the forward return was positive."),
        ("P/E (TTM)", "Price-to-Earnings ratio, trailing twelve months — price per share divided by earnings per share over the last year. A common (rough) valuation gauge."),
        ("Beta", "A stock's volatility relative to the overall market (1.0 = moves with the market; >1 = more volatile; <1 = less)."),
        ("ML", "Machine Learning — in this app, a sentence-embedding model (for news matching) and FinBERT (for sentiment), both with automatic non-ML fallbacks if unavailable."),
        ("Sentence embedding", "A numeric vector representation of a piece of text; two texts about similar things get similar vectors. Used here to find headlines that might be about a stock without an exact name match — see Section 5."),
        ("Cosine similarity", "A 0-1-ish score measuring how similar two embeddings are. Higher = more semantically similar text."),
        ("FinBERT", "A BERT-family language model fine-tuned specifically to classify financial text as positive/negative/neutral. Used here for headline sentiment."),
        ("Sentiment", "Whether a piece of text reads as positive, negative, or neutral for the stock/market it discusses."),
        ("Confirmed vs. possible (news)", "'Confirmed' = exact name/symbol match (high confidence). 'Possible' = ML similarity match only (lower confidence, unconfirmed) — see Section 5."),
        ("RSS feed", "A standard format news sites publish their latest articles in, which this app polls for headlines."),
        ("Watchlist", "The list of stocks (symbol + aliases) a scan runs against — an NSE index or your own custom list."),
    ]
    st.dataframe(pd.DataFrame(glossary, columns=["Term", "Definition"]), use_container_width=True, hide_index=True)

    st.header("3. How to use the Screener tab")
    st.markdown(
        """
1. **Choose a watchlist** (sidebar): pick an index from the **Index universe** dropdown (Nifty 50 /
   Nifty 500 / All) and click **Load Index** — this fetches the constituent list live from NSE. You
   can also upload your own CSV (`symbol,aliases` columns) or edit the table directly. Click
   **Save as default** to persist your current watchlist for next time.
2. **Set options** (sidebar): price history lookback window, whether a confirmed news mention is
   *required* to appear in results, and the minimum technical score to include.
3. Click **Run Screener**. Larger watchlists (Nifty 500 / All) take noticeably longer — a progress
   bar tracks price fetching.
4. **Read the results table**: one row per stock that passed the bullish + score filters, with its
   technical score, whether it's in the news, sentiment, and a combined final score (highest first).
5. **Expand a stock's row** under "Details" for: its key stats (price, market cap, P/E, etc.), a
   candlestick chart, confirmed headlines with sentiment, and any unconfirmed ML-matched headlines.
6. If you change a sidebar setting, you must click **Run Screener** again to refresh — a caption
   above the results tells you which settings the currently-displayed results were run with, and
   warns if the sidebar no longer matches them.
        """
    )

    st.header("4. How to use the Backtest tab")
    st.markdown(
        """
1. Pick a **watchlist** the same way as the Screener (shared sidebar setting).
2. Set **Backtest history** (how many years of price data to replay) and **forward horizons**
   (comma-separated days, e.g. `5,10,20`) in the sidebar.
3. Click **Run Backtest**.
4. Read the results table: one row per signal, with hit rate % and avg return % at each horizon,
   plus a `baseline` row (every day, signal or not) to compare against. A signal is only useful if
   it beats `baseline` — matching it means that signal isn't adding real information.
        """
    )

    st.header("5. How news matching & sentiment work")
    st.markdown(
        f"""
**Matching a headline to a stock** (`screener/news.py`) uses two methods together:
- **Confirmed (regex)** — exact word-boundary match of the stock's symbol or known company-name
  aliases against headline text. This is the only source used for the "mentioned" flag and for
  sentiment scoring.
- **Possible (semantic/ML, unconfirmed)** — sentence-embedding similarity (`{news.EMBEDDING_MODEL_NAME}`)
  surfaces additional headlines that might be about the stock (e.g. paraphrases without an exact
  name match), shown separately and excluded from scoring.

**Sentiment** is scored per headline with **{news.SENTIMENT_MODEL_NAME}** (FinBERT), shown as a
label + score (🟢/🔴/⚪, signed -1 to +1) next to each confirmed headline, and aggregated into each
stock's overall sentiment. Only confirmed headlines feed this — not the unconfirmed "possible"
ones. If FinBERT can't load, scoring falls back to a keyword lexicon on the same scale, and the
UI shows a warning when this fallback is active.

**Recency**: headlines older than **{news.NEWS_MAX_AGE_DAYS} days**, or with no parseable publish
date, are dropped before matching even runs.
        """
    )

    st.header("6. Technical signals reference")
    st.markdown(
        f"""
Each day, every stock is checked against a fixed set of rules computed from its OHLCV history
(`screener/technical.py`). Each rule that fires contributes its weight to a 0-100 technical score;
a stock is flagged bullish once its score reaches **{technical.BULLISH_THRESHOLD}**.
        """
    )
    signal_explanations = {
        "golden_cross": "The 50-day SMA crossed above the 200-day SMA in the last few days.",
        "uptrend_only": "50-day SMA is above the 200-day SMA, but didn't just cross (an already-established uptrend).",
        "rsi_healthy_rising": "14-day RSI is between 40-70 (not overbought/oversold) and rising.",
        "macd_bullish_cross": "MACD line crossed above its signal line in the last day.",
        "macd_above_only": "MACD line is above its signal line, but didn't just cross.",
        "volume_spike": "Today's volume is at least 1.5x the 20-day average volume.",
        "breakout_20d": "Price closed at or above the prior 20-day high.",
    }
    doc_rows = [
        {"Signal": sig, "Weight": technical.WEIGHTS[sig], "Definition": signal_explanations.get(sig, "")}
        for sig in technical.SIGNAL_COLUMNS
    ]
    st.dataframe(pd.DataFrame(doc_rows), use_container_width=True, hide_index=True)
    st.caption(
        "Bonus, not included above: a few bullish candlestick patterns (engulfing, hammer, morning "
        "star) add +10 each if the optional `pandas-ta` dependency is installed."
    )
    st.markdown(
        f"""
```
final_score = {scoring.WEIGHT_TECHNICAL} * technical_score + {scoring.WEIGHT_NEWS} * news_score
```
A stock only appears in results if it's technically bullish; a confirmed news mention is only
required if the "Require news mention" sidebar toggle is on.
        """
    )

    st.header("7. Methodology notes: why these weights?")
    st.markdown(
        f"""
The weights and threshold in Section 6 were **not guessed** — they came from running the Backtest
tab's methodology (Section 4) against 5 years of Nifty-50 price history, replaying each signal and
comparing its hit-rate/avg-return to the unconditional `baseline`:

- `golden_cross` and `volume_spike` beat baseline consistently at every horizon tested → weighted up.
- `macd_above_only` had a smaller but consistent edge → weighted up slightly.
- `rsi_healthy_rising`, `macd_bullish_cross`, `breakout_20d` had edges that flipped sign across
  horizons (i.e. inconsistent — noise, not a reliable signal) → weighted down.
- `uptrend_only` had a *negative* edge at every horizon tested — it tends to fire on trends that are
  already late/extended → weight set to **0**. It's still detected and shown as a label for
  transparency, but contributes nothing to the score.

Similarly, the news-matching split (Section 5) exists because testing found generic sentence
embeddings cannot reliably tell similarly-named but unrelated companies apart (e.g. "Reliance
Industries" vs. "MAN Industries" scored similarly) — so ML-only matches are kept separate and
excluded from scoring rather than risk a wrong match silently affecting a stock's ranking. The
embedding similarity floor is `{news.SEMANTIC_FLOOR}`, with each headline assignable to only its
single best-matching symbol (not an independent per-symbol threshold) to reduce cross-stock false
positives.

These are point-in-time findings from the watchlist/period tested, not universal truths — re-run
the Backtest tab periodically, especially after changing the watchlist or market conditions shift,
to confirm the weights still hold up.
        """
    )
