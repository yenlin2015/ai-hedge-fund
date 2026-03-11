"""王短鳥 (Wang Duan Niao) — Meme/Narrative Crypto Trading Agent.

Based on the Twitter KOL @wanghebbf. This agent catches early meme coin
narratives from Crypto Twitter, making high-conviction fast trades based
on social sentiment and narrative momentum.

Unlike traditional agents that analyze financial statements, this agent
analyzes crypto narratives, social sentiment, and price momentum.
"""

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing_extensions import Literal
import json

from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_prices, prices_to_df
from src.tools.twitter_api import get_kol_tweets
from src.utils.llm import call_llm
from src.utils.progress import progress

# ── Twitter handle for this KOL ─────────────────────────────────────────────
KOL_HANDLE = "wanghebbf"

# ── Ticker aliases for tweet matching ────────────────────────────────────────
CRYPTO_ALIASES = {
    "BTC": ["btc", "$btc", "bitcoin", "比特幣", "比特币"],
    "ETH": ["eth", "$eth", "ethereum", "以太坊", "以太幣"],
    "SOL": ["sol", "$sol", "solana"],
    "DOGE": ["doge", "$doge", "dogecoin", "狗狗幣"],
    "PEPE": ["pepe", "$pepe"],
    "WIF": ["wif", "$wif", "dogwifhat"],
    "BONK": ["bonk", "$bonk"],
    "ARB": ["arb", "$arb", "arbitrum"],
    "OP": ["op", "$op", "optimism"],
    "SUI": ["sui", "$sui"],
    "SEI": ["sei", "$sei"],
    "APT": ["apt", "$apt", "aptos"],
    "AVAX": ["avax", "$avax", "avalanche"],
    "LINK": ["link", "$link", "chainlink"],
    "UNI": ["uni", "$uni", "uniswap"],
    "NEAR": ["near", "$near"],
    "INJ": ["inj", "$inj", "injective"],
    "FET": ["fet", "$fet", "fetch"],
    "RENDER": ["render", "$render", "rndr", "$rndr"],
    "TIA": ["tia", "$tia", "celestia"],
    "TON": ["ton", "$ton", "toncoin"],
    "XRP": ["xrp", "$xrp", "ripple"],
    "ADA": ["ada", "$ada", "cardano"],
    "DOT": ["dot", "$dot", "polkadot"],
    "AAVE": ["aave", "$aave"],
    "MKR": ["mkr", "$mkr", "maker"],
    "FIL": ["fil", "$fil", "filecoin"],
    "ATOM": ["atom", "$atom", "cosmos"],
    "LDO": ["ldo", "$ldo", "lido"],
}


class WangDuanNiaoSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


# ── Helper functions ─────────────────────────────────────────────────────────


def format_tweets_for_prompt(tweets: list[dict], limit: int = 50) -> str:
    """Format tweets into a compact string for the LLM prompt."""
    if not tweets:
        return "(No tweets available)"

    # Sort by date descending, take most recent
    sorted_tweets = sorted(tweets, key=lambda t: t.get("date", ""), reverse=True)[:limit]

    lines = []
    for t in sorted_tweets:
        date = t.get("date", "")[:10]  # YYYY-MM-DD
        text = t.get("text", "").replace("\n", " ").strip()
        likes = t.get("likes", 0)
        rts = t.get("retweets", 0)
        lines.append(f"[{date}] {text} (♥{likes} ↻{rts})")

    return "\n".join(lines)


def filter_tweets_by_ticker(tweets: list[dict], ticker: str) -> str:
    """Find tweets mentioning a specific ticker and return formatted string."""
    ticker_upper = ticker.upper()
    aliases = CRYPTO_ALIASES.get(ticker_upper, [ticker.lower(), f"${ticker.lower()}"])

    matching = []
    for t in tweets:
        text_lower = t.get("text", "").lower()
        if any(alias in text_lower for alias in aliases):
            matching.append(t)

    if not matching:
        return f"(No tweets found mentioning {ticker})"

    return format_tweets_for_prompt(matching, limit=20)


def summarize_price_action(prices_list) -> str:
    """Create a brief price action summary from Price objects."""
    if not prices_list or len(prices_list) < 2:
        return "(No price data available)"

    df = prices_to_df(prices_list)
    if df.empty or len(df) < 2:
        return "(Insufficient price data)"

    current = df["close"].iloc[-1]
    lines = [f"Current price: ${current:,.2f}"]

    # Calculate returns for different periods
    for label, days in [("24h", 1), ("7d", 7), ("30d", 30)]:
        if len(df) > days:
            prev = df["close"].iloc[-(days + 1)]
            pct = ((current - prev) / prev) * 100
            lines.append(f"{label}: {pct:+.1f}%")

    # Simple MA context
    if len(df) >= 20:
        ma20 = df["close"].rolling(20).mean().iloc[-1]
        position = "above" if current > ma20 else "below"
        lines.append(f"Price is {position} 20-day MA (${ma20:,.2f})")

    if len(df) >= 50:
        ma50 = df["close"].rolling(50).mean().iloc[-1]
        position = "above" if current > ma50 else "below"
        lines.append(f"Price is {position} 50-day MA (${ma50:,.2f})")

    # Volume trend
    if len(df) >= 7:
        recent_vol = df["volume"].iloc[-7:].mean()
        older_vol = df["volume"].iloc[-14:-7].mean() if len(df) >= 14 else recent_vol
        if older_vol > 0:
            vol_change = ((recent_vol - older_vol) / older_vol) * 100
            lines.append(f"7d avg volume vs prior 7d: {vol_change:+.1f}%")

    return " | ".join(lines)


# ── Main agent function ─────────────────────────────────────────────────────


def wang_duan_niao_agent(state: AgentState, agent_id: str = "wang_duan_niao_agent"):
    """Analyzes crypto using 王短鳥's meme/narrative trading style.

    1. Loads KOL tweets (cached or fresh from Apify)
    2. Gets price data for context
    3. Feeds tweets + price action to LLM with KOL persona
    4. Returns bullish/bearish/neutral signal
    """
    data = state["data"]
    tickers = data["tickers"]
    end_date = data["end_date"]
    start_date = data["start_date"]

    # Load KOL tweets once for all tickers
    progress.update_status(agent_id, None, "Loading KOL tweets from @wanghebbf")
    tweets = get_kol_tweets(KOL_HANDLE, max_tweets=200)
    recent_tweets = format_tweets_for_prompt(tweets, limit=50)

    analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Getting price data")
        prices = get_prices(ticker, start_date, end_date)
        price_context = summarize_price_action(prices)

        progress.update_status(agent_id, ticker, "Filtering relevant tweets")
        ticker_tweets = filter_tweets_by_ticker(tweets, ticker)

        progress.update_status(agent_id, ticker, "Generating 王短鳥 analysis")
        output = generate_wang_duan_niao_output(
            ticker=ticker,
            recent_tweets=recent_tweets,
            ticker_tweets=ticker_tweets,
            price_context=price_context,
            state=state,
            agent_id=agent_id,
        )

        analysis[ticker] = {
            "signal": output.signal,
            "confidence": output.confidence,
            "reasoning": output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=output.reasoning)

    # Standard agent return
    message = HumanMessage(content=json.dumps(analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(analysis, agent_id)

    state["data"]["analyst_signals"][agent_id] = analysis

    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


# ── LLM output generation ───────────────────────────────────────────────────


def generate_wang_duan_niao_output(
    ticker: str,
    recent_tweets: str,
    ticker_tweets: str,
    price_context: str,
    state: AgentState,
    agent_id: str = "wang_duan_niao_agent",
) -> WangDuanNiaoSignal:
    """Generate trading signal using 王短鳥's meme/narrative style."""

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are 王短鳥 (Wang Duan Niao), a legendary Crypto Twitter KOL known for catching meme coin narratives early and making high-conviction, fast-moving trades.

Your trading style:
- Hunt for early narrative shifts BEFORE they go mainstream on CT
- Read Crypto Twitter sentiment as a leading indicator — when CT is talking, you're already positioned
- High risk/reward mindset — you go big when conviction is high, you cut fast when wrong
- Fast entries, fast exits — you don't marry positions, you trade momentum
- Meme coins, narrative plays, and social momentum are your bread and butter
- In crypto, narrative IS fundamentals — adoption follows attention
- You track what wallets are doing, what CT is buzzing about, and what narratives are forming

Decision framework:
- BULLISH: Strong narrative forming, CT sentiment shifting positive, price showing momentum, early in the move
- BEARISH: Narrative dying/overplayed, CT losing interest, smart money exiting, momentum fading
- NEUTRAL: No clear narrative catalyst, mixed signals, better to wait for clarity

Confidence guide:
- 80-100: Clear narrative catalyst, strong CT buzz, price confirming — you're going big
- 60-79: Narrative emerging but not confirmed, watching closely
- 40-59: Mixed signals, could go either way
- 20-39: Weak setup, probably sitting this one out
- 0-19: No edge, no narrative, skip

Below are your recent tweets for context on your current market views and sentiment.
Use them to inform your analysis, staying true to your voice and trading style.

Keep reasoning concise (under 200 chars). Be direct — talk like a CT trader, not a banker.""",
            ),
            (
                "human",
                """Analyze {ticker} in your style.

Your recent tweets (for context on your current market views):
{recent_tweets}

Your tweets mentioning {ticker}:
{ticker_tweets}

Current price action for {ticker}:
{price_context}

Return exactly:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": float (0-100),
  "reasoning": "your call in your voice"
}}""",
            ),
        ]
    )

    prompt = template.invoke({
        "ticker": ticker,
        "recent_tweets": recent_tweets,
        "ticker_tweets": ticker_tweets,
        "price_context": price_context,
    })

    def create_default_signal():
        return WangDuanNiaoSignal(
            signal="neutral",
            confidence=30.0,
            reasoning="No tweets or data to read the narrative. Sitting out.",
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=WangDuanNiaoSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_signal,
    )
