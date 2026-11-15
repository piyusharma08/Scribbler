import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from screener import universe, prices, technical, news, scoring

st.set_page_config(page_title="Scribbler", layout="wide")

st.title("Scribbler — Indian Stock Screener (Bullish Patterns + In the News)")
st.caption(
    "Scans your watchlist for bullish technical signals (golden cross, RSI, MACD, "
    "volume breakout, candlesticks) and cross-references free market RSS feeds for "
    "news mentions. Sentiment is a simple keyword heuristic, not ML-grade — treat "
    "scores as directional, not precise."
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

col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("Reset to Nifty 50"):
        st.session_state.watchlist_df = universe.default_watchlist_df()
        st.rerun()
with col_b:
    if st.button("Save as default"):
        universe.save_watchlist(st.session_state.watchlist_df)
        st.sidebar.success("Saved.")

st.sidebar.header("Options")
period = st.sidebar.selectbox("Price history lookback", ["3mo", "6mo", "1y"], index=1)
require_news = st.sidebar.checkbox("Require news mention (strict)", value=True)
min_score = st.sidebar.slider("Minimum technical score", 0, 100, 40, step=5)

run = st.sidebar.button("Run Screener", type="primary")

# ---------- Main ----------
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

if "last_results" in st.session_state:
    ranked = st.session_state.last_results
    st.subheader(f"Results: {len(ranked)} stock(s) matched")

    if not ranked:
        st.info(
            "No stocks matched. Try unchecking 'Require news mention' (RSS coverage is "
            "sparse) or lowering the minimum technical score."
        )
    else:
        table = pd.DataFrame(
            [
                {
                    "Symbol": r["symbol"],
                    "Technical Score": r["technical_score"],
                    "In News": "Yes" if r["news_mentioned"] else "No",
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

                if r["headlines"]:
                    st.write("**Matched headlines:**")
                    for h in r["headlines"]:
                        st.markdown(f"- [{h['title']}]({h['link']}) _{h['published']}_")
                else:
                    st.write("No matched headlines in current RSS feeds.")
else:
    st.info("Configure your watchlist in the sidebar and click **Run Screener** to begin.")
