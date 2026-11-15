"""Free RSS news fetching + simple keyword-based sentiment scoring.

Sentiment here is a lightweight heuristic lexicon, not an ML/NLP model -
good enough to bias ranking, not to be trusted as ground truth.
"""
import re
from datetime import datetime, timedelta, timezone

import feedparser
import streamlit as st

RSS_FEEDS = [
    "https://www.moneycontrol.com/rss/marketreports.xml",
    "https://www.moneycontrol.com/rss/business.xml",
    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "https://www.business-standard.com/rss/markets-106.rss",
]

POSITIVE_WORDS = [
    "surge", "surges", "rally", "rallies", "jumps", "jump", "soars", "soar",
    "beats estimates", "beat estimates", "upgrade", "upgraded", "outperform",
    "order win", "wins order", "bags order", "record profit", "strong growth",
    "all-time high", "buyback", "bullish", "gains", "gain", "rises", "rise",
    "profit jumps", "revenue growth", "expansion", "stake buy", "block deal buy",
]

NEGATIVE_WORDS = [
    "falls", "fall", "plunges", "plunge", "crashes", "crash", "downgrade",
    "downgraded", "underperform", "probe", "investigation", "misses estimates",
    "miss estimates", "loss", "losses", "decline", "declines", "slump", "slumps",
    "weak", "sell-off", "selloff", "fraud", "default", "resigns", "scam",
]

_WORD_RE = re.compile(r"[a-z0-9&.'-]+")


@st.cache_data(ttl=900, show_spinner=False)
def fetch_all_headlines():
    """Fetch and parse all configured RSS feeds. Returns list of dicts."""
    items = []
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = getattr(entry, "title", "") or ""
                link = getattr(entry, "link", "") or ""
                published = getattr(entry, "published", "") or getattr(entry, "updated", "")
                summary = getattr(entry, "summary", "") or ""
                items.append(
                    {
                        "title": title,
                        "link": link,
                        "published": published,
                        "summary": summary,
                        "text": f"{title} {summary}",
                    }
                )
        except Exception:
            continue
    return items


def _score_sentiment(text: str) -> int:
    t = text.lower()
    score = 0
    for w in POSITIVE_WORDS:
        if w in t:
            score += 1
    for w in NEGATIVE_WORDS:
        if w in t:
            score -= 1
    return score


def match_news_for_watchlist(watchlist_rows):
    """watchlist_rows: iterable of (symbol, aliases_list).

    Returns dict(symbol -> {"mentioned": bool, "headlines": [...], "sentiment_score": int})
    """
    headlines = fetch_all_headlines()
    result = {}
    for symbol, aliases in watchlist_rows:
        names = [symbol] + list(aliases)
        matched = []
        for item in headlines:
            text_lower = item["text"].lower()
            if any(name.lower() in text_lower for name in names if name):
                matched.append(item)
        sentiment_total = sum(_score_sentiment(m["text"]) for m in matched)
        result[symbol] = {
            "mentioned": len(matched) > 0,
            "headlines": matched[:5],
            "sentiment_score": sentiment_total,
        }
    return result


def news_component_score(sentiment_score: int, mentioned: bool) -> int:
    """Map raw sentiment tally to a 0-100 component score for ranking."""
    if not mentioned:
        return 0
    # clamp sentiment_score (-5..+5 typical) into 30..100 range when mentioned
    base = 50
    scaled = max(-20, min(20, sentiment_score * 10))
    return max(0, min(100, base + scaled))
