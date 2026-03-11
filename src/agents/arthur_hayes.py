"""Arthur Hayes — Macro-Driven Crypto Trading Agent.

Based on Arthur Hayes, co-founder of BitMEX and author of the Crypto Trader
Digest blog. This agent analyzes crypto through the lens of global macro:
central bank liquidity, monetary policy, DXY, yield curves, and credit cycles.

His thesis: crypto is a liquidity sponge — when central banks print, crypto
pumps; when they tighten, crypto dumps. Everything else is noise.
"""

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing_extensions import Literal
import json

from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_prices, prices_to_df
from src.utils.llm import call_llm
from src.utils.progress import progress


class ArthurHayesSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


# ── Helper functions ─────────────────────────────────────────────────────────


def summarize_price_action(prices_list) -> str:
    """Create a brief price action summary from Price objects."""
    if not prices_list or len(prices_list) < 2:
        return "(No price data available)"

    df = prices_to_df(prices_list)
    if df.empty or len(df) < 2:
        return "(Insufficient price data)"

    current = df["close"].iloc[-1]
    lines = [f"Current price: ${current:,.2f}"]

    for label, days in [("24h", 1), ("7d", 7), ("30d", 30), ("90d", 90)]:
        if len(df) > days:
            prev = df["close"].iloc[-(days + 1)]
            pct = ((current - prev) / prev) * 100
            lines.append(f"{label}: {pct:+.1f}%")

    if len(df) >= 20:
        ma20 = df["close"].rolling(20).mean().iloc[-1]
        position = "above" if current > ma20 else "below"
        lines.append(f"Price is {position} 20-day MA (${ma20:,.2f})")

    if len(df) >= 50:
        ma50 = df["close"].rolling(50).mean().iloc[-1]
        position = "above" if current > ma50 else "below"
        lines.append(f"Price is {position} 50-day MA (${ma50:,.2f})")

    if len(df) >= 7:
        recent_vol = df["volume"].iloc[-7:].mean()
        older_vol = df["volume"].iloc[-14:-7].mean() if len(df) >= 14 else recent_vol
        if older_vol > 0:
            vol_change = ((recent_vol - older_vol) / older_vol) * 100
            lines.append(f"7d avg volume vs prior 7d: {vol_change:+.1f}%")

    # Volatility context (Hayes loves vol)
    if len(df) >= 30:
        returns = df["close"].pct_change().dropna()
        vol_30d = returns.iloc[-30:].std() * (365**0.5) * 100
        lines.append(f"30d annualized volatility: {vol_30d:.1f}%")

    return " | ".join(lines)


# ── Main agent function ─────────────────────────────────────────────────────


def arthur_hayes_agent(state: AgentState, agent_id: str = "arthur_hayes_agent"):
    """Analyzes crypto using Arthur Hayes's macro-liquidity framework.

    1. Gets price data and market context
    2. Feeds data to LLM with Hayes's macro persona
    3. Returns bullish/bearish/neutral signal
    """
    data = state["data"]
    tickers = data["tickers"]
    end_date = data["end_date"]
    start_date = data["start_date"]

    analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Getting price data")
        prices = get_prices(ticker, start_date, end_date)
        price_context = summarize_price_action(prices)

        progress.update_status(agent_id, ticker, "Generating Arthur Hayes analysis")
        output = generate_arthur_hayes_output(
            ticker=ticker,
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

    message = HumanMessage(content=json.dumps(analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(analysis, agent_id)

    state["data"]["analyst_signals"][agent_id] = analysis

    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


# ── LLM output generation ───────────────────────────────────────────────────


def generate_arthur_hayes_output(
    ticker: str,
    price_context: str,
    state: AgentState,
    agent_id: str = "arthur_hayes_agent",
) -> ArthurHayesSignal:
    """Generate trading signal using Arthur Hayes's macro-liquidity framework."""

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are Arthur Hayes, co-founder of BitMEX and author of the Crypto Trader Digest blog. You are one of the most influential voices in crypto, known for your sharp macro analysis and colorful writing style.

Your trading philosophy:
- Crypto is a LIQUIDITY SPONGE — it absorbs excess fiat liquidity from central banks. When the money printer goes brrr, crypto pumps. When liquidity drains, crypto dumps. This is the ONLY thing that matters long-term.
- The Fed, BOJ, PBOC, and ECB are your north stars. Watch their balance sheets, not the news cycle.
- DXY (US Dollar Index) is inversely correlated to crypto. Dollar weakness = crypto strength.
- Real yields matter. Negative real yields push capital into risk assets like crypto. Positive real yields pull capital out.
- The yield curve tells you where we are in the credit cycle. Steepening after inversion = recession coming = eventual money printing = bullish crypto (with pain first).
- Bitcoin is digital gold and the apex predator of crypto. ETH is a decentralized computer. Everything else is a bet on narratives.
- You think in terms of cycles: accumulation -> markup -> distribution -> markdown. Position accordingly.
- You are NOT a permabull. You will go to cash or short when macro conditions demand it. Capital preservation matters.
- You size positions aggressively when conviction is high and the macro setup is clean.

Your writing/thinking style:
- You use vivid metaphors and cultural references (anime, history, economics)
- You connect seemingly unrelated macro events to crypto price action
- You think about second and third-order effects of policy decisions
- You are blunt, opinionated, and unapologetic about your views
- You back up your thesis with data and logical reasoning

Decision framework:
- BULLISH: Central bank liquidity expanding (or about to), DXY weakening, real yields falling, credit conditions loosening, BTC holding key support levels
- BEARISH: Liquidity draining (QT, rate hikes), DXY strengthening, real yields rising, credit stress emerging, key support levels breaking
- NEUTRAL: Mixed macro signals, transitional period between regimes, no clear edge

Confidence guide:
- 80-100: Macro setup is crystal clear — liquidity direction confirmed, positioning aggressively
- 60-79: Macro thesis is forming but needs confirmation — building positions
- 40-59: Conflicting signals across macro indicators — small positions or hedged
- 20-39: Uncertain macro regime — mostly cash, watching
- 0-19: No edge — sitting on hands

Keep reasoning concise (under 250 chars). Write like Hayes — sharp, macro-focused, with conviction.""",
            ),
            (
                "human",
                """Analyze {ticker} through your macro-liquidity lens.

Current price action for {ticker}:
{price_context}

Consider the current global macro environment:
- Where are we in the liquidity cycle?
- What are central banks doing with their balance sheets?
- What does the DXY and yield curve tell us?
- How does {ticker} fit into the macro picture?

Return exactly:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": float (0-100),
  "reasoning": "your macro thesis in your voice"
}}""",
            ),
        ]
    )

    prompt = template.invoke({
        "ticker": ticker,
        "price_context": price_context,
    })

    def create_default_signal():
        return ArthurHayesSignal(
            signal="neutral",
            confidence=25.0,
            reasoning="Insufficient data to read the macro tea leaves. Sitting in cash until the picture clears.",
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=ArthurHayesSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_signal,
    )
