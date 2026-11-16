"""Combine technical + news scores into a ranked result set."""
from .news import news_component_score

WEIGHT_TECHNICAL = 0.6
WEIGHT_NEWS = 0.4


def combine(symbol, tech_result, news_result, require_news: bool, weight_tech=WEIGHT_TECHNICAL, weight_news=WEIGHT_NEWS):
    """Return a result row dict, or None if it should be filtered out."""
    if tech_result is None:
        return None
    if not tech_result["bullish"]:
        return None

    news_mentioned = news_result["mentioned"] if news_result else False
    if require_news and not news_mentioned:
        return None

    news_score = news_component_score(
        news_result["sentiment_score"] if news_result else 0, news_mentioned
    )
    final_score = weight_tech * tech_result["score"] + weight_news * news_score

    return {
        "symbol": symbol,
        "technical_score": tech_result["score"],
        "signals": tech_result["signals"],
        "news_mentioned": news_mentioned,
        "news_score": news_score,
        "sentiment_score": news_result["sentiment_score"] if news_result else 0.0,
        "sentiment_label": news_result.get("sentiment_label", "neutral") if news_result else "neutral",
        "headlines": news_result["headlines"] if news_result else [],
        "possible_headlines": news_result.get("possible_headlines", []) if news_result else [],
        "final_score": round(final_score, 1),
    }


def rank_results(rows):
    return sorted([r for r in rows if r is not None], key=lambda r: r["final_score"], reverse=True)
