"""Free RSS news fetching, ML-based (semantic) stock/headline matching, and
ML-based (FinBERT) sentiment scoring, with a keyword-lexicon fallback for
both if the relevant model can't be loaded.
"""
import re
from datetime import datetime, timedelta, timezone

import feedparser
import numpy as np
import streamlit as st

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    _HAS_SENTENCE_TRANSFORMERS = False

try:
    from transformers import pipeline as hf_pipeline
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
SENTIMENT_MODEL_NAME = "ProsusAI/finbert"

# Calibrated empirically (see PLAN.md): a flat per-symbol similarity cutoff
# can't cleanly separate real matches from cross-stock contamination - e.g.
# a "TCS" query scored 0.336 against a *Reliance* headline, higher than a
# genuine no-literal-overlap Reliance paraphrase (0.329). Using argmax
# disambiguation (each headline can only be claimed by its single
# best-matching symbol) resolves that, so SEMANTIC_FLOOR only needs to reject
# headlines where even the best-matching symbol is irrelevant (unrelated
# headlines scored <0.15 in calibration).
SEMANTIC_FLOOR = 0.30

# "Recent" per the product requirement ("within 1-2 months"). Also has the
# side effect of filtering out feeds serving stale/cached content instead of
# live updates (found in testing: two of the four RSS_FEEDS below were
# serving a snapshot over a year old) - recency filtering catches that too.
NEWS_MAX_AGE_DAYS = 60

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


def _parse_entry_datetime(entry):
    """Extract a timezone-aware UTC datetime from a feedparser entry, or
    None if no parseable date is present."""
    struct = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if struct is None:
        return None
    try:
        return datetime(*struct[:6], tzinfo=timezone.utc)
    except Exception:
        return None


def _humanize_age(published_dt):
    if published_dt is None:
        return "date unknown"
    delta = datetime.now(timezone.utc) - published_dt
    days = delta.days
    if days <= 0:
        return "today"
    if days == 1:
        return "1 day ago"
    if days < 30:
        return f"{days} days ago"
    return published_dt.strftime("%b %d, %Y")


@st.cache_data(ttl=900, show_spinner=False)
def fetch_all_headlines():
    """Fetch and parse all configured RSS feeds, keeping only items within
    NEWS_MAX_AGE_DAYS. Items with no parseable publish date are dropped too
    - recency can't be verified without one. Returns list of dicts."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=NEWS_MAX_AGE_DAYS)
    items = []
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                published_dt = _parse_entry_datetime(entry)
                if published_dt is None or published_dt < cutoff:
                    continue
                title = getattr(entry, "title", "") or ""
                link = getattr(entry, "link", "") or ""
                summary = getattr(entry, "summary", "") or ""
                items.append(
                    {
                        "title": title,
                        "link": link,
                        "published": getattr(entry, "published", "") or getattr(entry, "updated", ""),
                        "published_dt": published_dt,
                        "published_display": _humanize_age(published_dt),
                        "summary": summary,
                        "text": f"{title} {summary}",
                    }
                )
        except Exception:
            continue
    return items


def _score_sentiment_keyword(text: str) -> float:
    """Keyword-lexicon fallback, normalized to the same [-1, 1] scale FinBERT
    produces so news_component_score works regardless of which method ran."""
    t = text.lower()
    tally = 0
    for w in POSITIVE_WORDS:
        if w in t:
            tally += 1
    for w in NEGATIVE_WORDS:
        if w in t:
            tally -= 1
    return max(-3, min(3, tally)) / 3.0


@st.cache_resource(show_spinner=False)
def _load_sentiment_model():
    """Load the FinBERT sentiment pipeline once per process. Returns None on
    any failure so callers fall back to the keyword lexicon."""
    if not _HAS_TRANSFORMERS:
        return None
    try:
        return hf_pipeline("sentiment-analysis", model=SENTIMENT_MODEL_NAME)
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def _finbert_scores(_model, texts):
    """Cached batch FinBERT inference. Returns a list of signed scores in
    [-1, 1] - +confidence for positive, -confidence for negative, 0 for
    neutral - aligned index-for-index with `texts`."""
    texts = list(texts)
    if not texts:
        return []
    results = _model(texts, truncation=True)
    scores = []
    for r in results:
        label = r["label"].lower()
        if label == "positive":
            scores.append(r["score"])
        elif label == "negative":
            scores.append(-r["score"])
        else:
            scores.append(0.0)
    return scores


def _build_name_pattern(names):
    """Word-boundary, case-insensitive regex matching any of `names` as a
    whole token (or phrase) - not as a substring inside another word. Names
    are re.escape'd so symbols like 'M&M' or 'BAJAJ-AUTO' are matched
    literally rather than as regex syntax.
    """
    escaped = [re.escape(n.strip()) for n in names if n and n.strip()]
    if not escaped:
        return None
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


def _regex_match_indices(symbol, aliases, headlines):
    pattern = _build_name_pattern([symbol] + list(aliases))
    if pattern is None:
        return set()
    return {i for i, item in enumerate(headlines) if pattern.search(item["text"])}


@st.cache_resource(show_spinner=False)
def _load_embedding_model():
    """Load the sentence-embedding model once per process. Returns None on
    any failure (no internet on first run, import error, etc.) so callers
    can fall back to regex-only matching instead of crashing."""
    if not _HAS_SENTENCE_TRANSFORMERS:
        return None
    try:
        return SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def _embed_texts(_model, texts):
    """Cached batch embedding. `_model` is prefixed with underscore so
    Streamlit doesn't try (and fail) to hash the model object itself."""
    return _model.encode(list(texts), convert_to_numpy=True)


def _semantic_assignments(watchlist_rows, headlines, model):
    """For each headline, assign it to its single best-matching symbol
    (argmax across the whole watchlist, not an independent per-symbol
    threshold) - see SEMANTIC_FLOOR docstring for why this disambiguation
    matters. Returns dict(symbol -> list of (headline_index, score)).
    """
    if model is None or not headlines:
        return {}

    rows = list(watchlist_rows)
    queries = [" / ".join(dict.fromkeys([symbol] + list(aliases))) for symbol, aliases in rows]
    headline_texts = [h["text"] for h in headlines]

    query_emb = _embed_texts(model, tuple(queries))
    headline_emb = _embed_texts(model, tuple(headline_texts))
    sims = st_util.cos_sim(query_emb, headline_emb).numpy()  # shape (n_symbols, n_headlines)

    best_symbol_idx = np.argmax(sims, axis=0)
    best_scores = sims[best_symbol_idx, np.arange(sims.shape[1])]

    assignments = {}
    for headline_idx, (symbol_idx, score) in enumerate(zip(best_symbol_idx, best_scores)):
        if score < SEMANTIC_FLOOR:
            continue
        symbol = rows[symbol_idx][0]
        assignments.setdefault(symbol, []).append((headline_idx, float(score)))
    return assignments


def match_news_for_watchlist(watchlist_rows):
    """watchlist_rows: iterable of (symbol, aliases_list).

    Regex (exact word/alias) matching is the trusted signal for "mentioned"
    and sentiment scoring - verified against live RSS feeds to produce
    correct matches. Semantic (sentence-embedding) similarity is run
    alongside it to surface *possible* additional mentions the regex missed
    (e.g. paraphrases), but is NOT trusted to confirm a match on its own.

    Why: live testing found generic sentence embeddings cannot reliably
    distinguish similarly-named but unrelated companies - e.g. "MAN
    Industries" and "Jagatjit Industries" scored as high or higher against
    the query "Reliance Industries" as genuine Reliance headlines did (both
    ~0.4-0.47), because generic corporate vocabulary ("Industries", "Q2",
    "profit") dominates the embedding for a general-purpose model with no
    financial-entity training. No similarity threshold separates true
    matches from this class of false positive, so semantic-only hits are
    kept clearly separate and excluded from scoring until a better-suited
    model (e.g. finance-domain fine-tuned embeddings) is available.

    Returns dict(symbol -> {
        "mentioned": bool,            # true only on a regex (confirmed) match
        "headlines": [...],           # confirmed matches, used for sentiment/scoring
        "possible_headlines": [...],  # semantic-only, unconfirmed - informational only
        "sentiment_score": float,     # avg signed sentiment in [-1, 1] over confirmed headlines
        "method": "semantic+regex" | "regex-only",
        "sentiment_method": "finbert" | "keyword-fallback",
    })
    """
    rows = list(watchlist_rows)
    headlines = fetch_all_headlines()

    model = _load_embedding_model()
    semantic = _semantic_assignments(rows, headlines, model)
    method = "semantic+regex" if model is not None else "regex-only"

    sentiment_model = _load_sentiment_model()
    headline_texts = tuple(h["text"] for h in headlines)
    if sentiment_model is not None:
        sentiment_scores = _finbert_scores(sentiment_model, headline_texts)
        sentiment_method = "finbert"
    else:
        sentiment_scores = [_score_sentiment_keyword(t) for t in headline_texts]
        sentiment_method = "keyword-fallback"

    # Attach each headline's own sentiment so the UI can show the model's
    # actual per-article output, not just the aggregated 0-100 news score.
    for item, score in zip(headlines, sentiment_scores):
        item["sentiment_score"] = score
        item["sentiment_label"] = sentiment_label(score)

    result = {}
    for symbol, aliases in rows:
        regex_idx = sorted(_regex_match_indices(symbol, aliases, headlines))
        semantic_hits = semantic.get(symbol, [])
        semantic_only_idx = sorted(
            (i for i, _ in semantic_hits if i not in regex_idx),
            key=lambda i: -dict(semantic_hits)[i],
        )

        confirmed = [headlines[i] for i in regex_idx][:5]
        possible = [headlines[i] for i in semantic_only_idx][:5]

        confirmed_idx = regex_idx[:5]
        sentiment_avg = (
            sum(sentiment_scores[i] for i in confirmed_idx) / len(confirmed_idx)
            if confirmed_idx
            else 0.0
        )
        result[symbol] = {
            "mentioned": len(confirmed) > 0,
            "headlines": confirmed,
            "possible_headlines": possible,
            "sentiment_score": sentiment_avg,
            "sentiment_label": sentiment_label(sentiment_avg) if confirmed else "neutral",
            "method": method,
            "sentiment_method": sentiment_method,
        }
    return result


def sentiment_label(score: float, neutral_band: float = 0.1) -> str:
    """Map a signed [-1, 1] sentiment score to a human label."""
    if score > neutral_band:
        return "positive"
    if score < -neutral_band:
        return "negative"
    return "neutral"


def news_component_score(sentiment_score: float, mentioned: bool) -> int:
    """Map a signed sentiment score in [-1, 1] to a 0-100 component score."""
    if not mentioned:
        return 0
    base = 50
    scaled = max(-1.0, min(1.0, sentiment_score)) * 40
    return max(0, min(100, round(base + scaled)))
