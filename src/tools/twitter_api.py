"""Twitter/X data fetcher using Apify's Tweet Scraper actor.

Fetches KOL tweets and caches them locally as JSON.
Requires APIFY_API_TOKEN environment variable.
"""

import datetime
import json
import os
import pathlib

TWEETS_CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "tweets"
CACHE_MAX_AGE_HOURS = 24


def _get_cache_path(handle: str) -> pathlib.Path:
    return TWEETS_CACHE_DIR / f"{handle.lower()}.json"


def load_cached_tweets(handle: str) -> list[dict] | None:
    """Load tweets from local JSON cache. Returns None if no cache or stale."""
    cache_path = _get_cache_path(handle)
    if not cache_path.exists():
        return None

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        cached_at = datetime.datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
        age = datetime.datetime.now() - cached_at
        if age.total_seconds() > CACHE_MAX_AGE_HOURS * 3600:
            return None  # stale
        return data.get("tweets", [])
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def _save_cache(handle: str, tweets: list[dict]) -> None:
    TWEETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _get_cache_path(handle)
    payload = {
        "handle": handle,
        "cached_at": datetime.datetime.now().isoformat(),
        "tweet_count": len(tweets),
        "tweets": tweets,
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_kol_tweets(handle: str, max_tweets: int = 200) -> list[dict]:
    """Fetch tweets from Apify's Tweet Scraper actor.

    Requires APIFY_API_TOKEN env var.
    Returns list of dicts with keys: text, date, likes, retweets, replies, url.
    """
    api_token = os.environ.get("APIFY_API_TOKEN")
    if not api_token:
        print("Warning: APIFY_API_TOKEN not set. Cannot fetch tweets from Apify.")
        return []

    try:
        from apify_client import ApifyClient
    except ImportError:
        print("Warning: apify-client not installed. Run: poetry add apify-client")
        return []

    client = ApifyClient(token=api_token)

    run_input = {
        "searchTerms": [f"from:{handle}"],
        "maxTweets": max_tweets,
        "sort": "Latest",
    }

    try:
        run = client.actor("apidojo/tweet-scraper").call(run_input=run_input)
        raw_tweets = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    except Exception as e:
        print(f"Warning: Apify tweet fetch failed: {e}")
        return []

    # Normalize to a clean format
    tweets = []
    for raw in raw_tweets:
        tweet = {
            "text": raw.get("text") or raw.get("full_text") or raw.get("tweet_text", ""),
            "date": raw.get("created_at") or raw.get("createdAt") or raw.get("date", ""),
            "likes": raw.get("favorite_count") or raw.get("likeCount") or raw.get("likes", 0),
            "retweets": raw.get("retweet_count") or raw.get("retweetCount") or raw.get("retweets", 0),
            "replies": raw.get("reply_count") or raw.get("replyCount") or raw.get("replies", 0),
            "url": raw.get("url") or raw.get("tweetUrl") or "",
        }
        if tweet["text"]:
            tweets.append(tweet)

    # Cache the results
    if tweets:
        _save_cache(handle, tweets)

    return tweets


def get_kol_tweets(handle: str, max_tweets: int = 200) -> list[dict]:
    """Get KOL tweets: try cache first, then Apify if stale/missing."""
    cached = load_cached_tweets(handle)
    if cached:
        return cached

    tweets = fetch_kol_tweets(handle, max_tweets)
    if tweets:
        return tweets

    # Last resort: load stale cache if it exists (better than nothing)
    cache_path = _get_cache_path(handle)
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return data.get("tweets", [])
        except (json.JSONDecodeError, KeyError):
            pass

    return []
