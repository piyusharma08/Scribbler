"""Watchlist sources: NSE index constituent lists (Nifty 50 / 500 / Total
Market) plus ticker <-> company-name aliases used for news matching.
"""
import io
import os
import re

import pandas as pd
import requests
import streamlit as st

WATCHLIST_PATH = os.path.join(os.path.dirname(__file__), "..", "watchlist.csv")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Display label -> NSE archives list name (https://archives.nseindia.com/content/indices/<name>.csv)
INDEX_OPTIONS = {
    "Nifty 50": "ind_nifty50list",
    "Nifty 500": "ind_nifty500list",
    "All (Nifty Total Market, ~750 stocks)": "ind_niftytotalmarket_list",
}

# Bundled snapshot used if the live NSE fetch fails (no internet, site change,
# etc.) - downloaded once during development, not meant to be kept perfectly
# fresh; index constituents change infrequently.
_FALLBACK_FILES = {
    "ind_nifty50list": "nifty50_fallback.csv",
    "ind_nifty500list": "nifty500_fallback.csv",
    "ind_niftytotalmarket_list": "niftytotalmarket_fallback.csv",
}

# Known short-form abbreviations NSE's "Company Name" column doesn't capture
# (e.g. "State Bank of India Ltd." -> no "SBI") - merged in as extra aliases
# for these Nifty-50 symbols so news matching doesn't regress for them.
ABBREVIATION_OVERRIDES = {
    "RELIANCE": ["Reliance"],
    "TCS": ["TCS"],
    "HDFCBANK": ["HDFC Bank"],
    "ICICIBANK": ["ICICI Bank"],
    "HINDUNILVR": ["HUL"],
    "ITC": ["ITC"],
    "SBIN": ["SBI"],
    "BHARTIARTL": ["Airtel"],
    "LT": ["L&T"],
    "KOTAKBANK": ["Kotak Bank"],
    "AXISBANK": ["Axis Bank"],
    "M&M": ["Mahindra & Mahindra", "M&M"],
    "HCLTECH": ["HCLTech"],
    "DRREDDY": ["Dr Reddy's"],
    "ONGC": ["ONGC"],
    "BPCL": ["BPCL"],
    "SBILIFE": ["SBI Life"],
    "UPL": ["UPL"],
    "LTIM": ["LTIMindtree"],
    "TATACONSUM": ["Tata Consumer"],
}

_SUFFIX_RE = re.compile(r"\s+(ltd\.?|limited)\s*$", re.IGNORECASE)


def _strip_corporate_suffix(name: str) -> str:
    return _SUFFIX_RE.sub("", name).strip()


def _build_watchlist_from_nse_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        symbol = str(row["Symbol"]).strip().upper()
        company = str(row["Company Name"]).strip()
        stripped = _strip_corporate_suffix(company)
        aliases = list(dict.fromkeys([stripped, company] + ABBREVIATION_OVERRIDES.get(symbol, [])))
        rows.append({"symbol": symbol, "aliases": ";".join(a for a in aliases if a)})
    return pd.DataFrame(rows)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_index_constituents(index_key: str) -> pd.DataFrame:
    """Fetch an NSE index constituent list and build a watchlist DataFrame
    (symbol, aliases) from it. Falls back to a bundled snapshot on any
    network/parse failure."""
    list_name = INDEX_OPTIONS[index_key]
    try:
        resp = requests.get(
            f"https://archives.nseindia.com/content/indices/{list_name}.csv",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception:
        fallback_path = os.path.join(DATA_DIR, _FALLBACK_FILES[list_name])
        df = pd.read_csv(fallback_path)
    return _build_watchlist_from_nse_df(df)


def default_watchlist_df():
    return fetch_index_constituents("Nifty 50")


def load_watchlist():
    """Load watchlist from CSV if present, else fall back to default Nifty 50."""
    if os.path.exists(WATCHLIST_PATH):
        try:
            df = pd.read_csv(WATCHLIST_PATH)
            if "symbol" in df.columns:
                if "aliases" not in df.columns:
                    df["aliases"] = df["symbol"]
                return df
        except Exception:
            pass
    return default_watchlist_df()


def save_watchlist(df):
    df.to_csv(WATCHLIST_PATH, index=False)


def ticker_for(symbol: str) -> str:
    """NSE ticker as used by yfinance."""
    symbol = symbol.strip().upper()
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol
    return f"{symbol}.NS"
