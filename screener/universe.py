"""Default watchlist (Nifty 50) and ticker <-> company-name aliases for news matching."""
import os
import pandas as pd

WATCHLIST_PATH = os.path.join(os.path.dirname(__file__), "..", "watchlist.csv")

# symbol (NSE, without .NS suffix) -> company name/aliases used for news matching
NIFTY50 = {
    "RELIANCE": ["Reliance Industries", "Reliance"],
    "TCS": ["Tata Consultancy Services", "TCS"],
    "HDFCBANK": ["HDFC Bank"],
    "ICICIBANK": ["ICICI Bank"],
    "INFY": ["Infosys"],
    "HINDUNILVR": ["Hindustan Unilever", "HUL"],
    "ITC": ["ITC Ltd", "ITC"],
    "SBIN": ["State Bank of India", "SBI"],
    "BHARTIARTL": ["Bharti Airtel", "Airtel"],
    "LT": ["Larsen & Toubro", "L&T"],
    "KOTAKBANK": ["Kotak Mahindra Bank", "Kotak Bank"],
    "AXISBANK": ["Axis Bank"],
    "BAJFINANCE": ["Bajaj Finance"],
    "ASIANPAINT": ["Asian Paints"],
    "MARUTI": ["Maruti Suzuki"],
    "SUNPHARMA": ["Sun Pharma", "Sun Pharmaceutical"],
    "TITAN": ["Titan Company", "Titan"],
    "ULTRACEMCO": ["UltraTech Cement"],
    "WIPRO": ["Wipro"],
    "ADANIENT": ["Adani Enterprises"],
    "ADANIPORTS": ["Adani Ports"],
    "NTPC": ["NTPC"],
    "POWERGRID": ["Power Grid"],
    "M&M": ["Mahindra & Mahindra", "M&M"],
    "TATAMOTORS": ["Tata Motors"],
    "TATASTEEL": ["Tata Steel"],
    "JSWSTEEL": ["JSW Steel"],
    "HCLTECH": ["HCL Technologies", "HCLTech"],
    "TECHM": ["Tech Mahindra"],
    "NESTLEIND": ["Nestle India"],
    "BRITANNIA": ["Britannia Industries", "Britannia"],
    "DRREDDY": ["Dr Reddy's", "Dr. Reddy's Laboratories"],
    "CIPLA": ["Cipla"],
    "DIVISLAB": ["Divi's Laboratories"],
    "GRASIM": ["Grasim Industries"],
    "HINDALCO": ["Hindalco"],
    "COALINDIA": ["Coal India"],
    "ONGC": ["ONGC", "Oil and Natural Gas Corporation"],
    "BPCL": ["Bharat Petroleum", "BPCL"],
    "INDUSINDBK": ["IndusInd Bank"],
    "BAJAJFINSV": ["Bajaj Finserv"],
    "BAJAJ-AUTO": ["Bajaj Auto"],
    "HEROMOTOCO": ["Hero MotoCorp"],
    "EICHERMOT": ["Eicher Motors"],
    "SBILIFE": ["SBI Life Insurance", "SBI Life"],
    "HDFCLIFE": ["HDFC Life"],
    "APOLLOHOSP": ["Apollo Hospitals"],
    "UPL": ["UPL Ltd", "UPL"],
    "SHRIRAMFIN": ["Shriram Finance"],
    "LTIM": ["LTIMindtree"],
    "TATACONSUM": ["Tata Consumer Products", "Tata Consumer"],
}


def default_watchlist_df():
    return pd.DataFrame(
        {
            "symbol": list(NIFTY50.keys()),
            "aliases": [";".join(v) for v in NIFTY50.values()],
        }
    )


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
