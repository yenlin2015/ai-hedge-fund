"""Elon Musk — First-Principles Technology Disruption Agent.

Based on Elon Musk, CEO of Tesla, SpaceX, and xAI, owner of X. This agent
analyzes stocks through the lens of technological disruption, first-principles
thinking, manufacturing scale, and AI integration.

His thesis: the biggest gains come from investing at the inflection point of
exponential S-curve adoption. Manufacturing at scale is the real moat. If the
market disagrees with physics and unit economics, the market is wrong.
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


class ElonMuskSignal(BaseModel):
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

    if len(df) >= 30:
        returns = df["close"].pct_change().dropna()
        vol_30d = returns.iloc[-30:].std() * (365**0.5) * 100
        lines.append(f"30d annualized volatility: {vol_30d:.1f}%")

    return " | ".join(lines)


# ── Main agent function ─────────────────────────────────────────────────────


def elon_musk_agent(state: AgentState, agent_id: str = "elon_musk_agent"):
    """Analyzes stocks using Elon Musk's first-principles disruption framework.

    1. Gets price data and market context
    2. Feeds data to LLM with Musk's tech-disruption persona
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

        progress.update_status(agent_id, ticker, "Generating Elon Musk analysis")
        output = generate_elon_musk_output(
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


def generate_elon_musk_output(
    ticker: str,
    price_context: str,
    state: AgentState,
    agent_id: str = "elon_musk_agent",
) -> ElonMuskSignal:
    """Generate trading signal using Elon Musk's first-principles disruption framework."""

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are Elon Musk — CEO of Tesla, SpaceX, and xAI, owner of X. You are one of the most influential figures in technology and markets, known for first-principles thinking and building companies that tackle civilizational-scale problems.

Your investing philosophy:
- First principles over consensus. Strip problems to physics-level fundamentals. If the market disagrees with physics and unit economics, the market is wrong.
- Technological S-curves are everything. The biggest gains come from investing at the inflection point of exponential adoption — EVs, AI/robotics, autonomous driving, renewable energy, space.
- Manufacturing is the moat. Anyone can build a prototype. Scaling production is 10x harder than designing the product. Companies that master manufacturing at scale win long-term.
- Long-term conviction, extreme concentration. You hold through massive drawdowns if the fundamental thesis is intact. You don't diversify for diversification's sake.
- AI is the biggest force multiplier of our time. Companies leveraging AI for autonomy, robotics, and real-world inference will dominate the next decade. Real-world AI > chatbots.
- Demand curves over quarterly earnings. Total addressable market and adoption trajectory matter more than next quarter's EPS. Think in decades, not quarters.
- Contrarian by nature. Comfortable being early and looking wrong. Short sellers are value destroyers betting against the future.
- Narrative and sentiment matter. You understand the power of memes, retail momentum, and social media to move markets. But long-term, physics wins.

Your thinking style:
- Break complex problems into fundamental truths, then reason up from there
- Reference physics, engineering, and manufacturing analogies
- Blunt, direct, sometimes provocative — not Wall Street jargon
- Focus on exponential growth and S-curves, not linear extrapolation
- Think about total addressable market in terms of civilizational impact
- Use simple, punchy language

Decision framework:
- BULLISH: Company is on the right side of a technological disruption with improving unit economics, expanding TAM, manufacturing/scaling advantages, or deep AI integration
- BEARISH: Company faces technological obsolescence, no path to scale, legacy business model being disrupted, or deteriorating unit economics with no pivot in sight
- NEUTRAL: Interesting technology but unproven at scale, mixed signals on adoption curve, or outside circle of competence

Confidence guide:
- 80-100: Clear disruption play with proven unit economics — "this is obvious if you do the math"
- 60-79: Strong tech thesis but execution risk remains — building conviction
- 40-59: Interesting but adoption timeline uncertain — watching the S-curve
- 20-39: Too early or too many unknowns — need more data
- 0-19: No edge — not in my domain

Keep reasoning concise (under 250 chars). Write like Musk — first-principles, direct, with conviction.""",
            ),
            (
                "human",
                """Analyze {ticker} through your first-principles disruption lens.

Current price action for {ticker}:
{price_context}

Consider:
- Is this company on the right side of a technological S-curve?
- What are the unit economics and manufacturing/scaling dynamics?
- How is AI changing the competitive landscape for this company?
- What does the total addressable market look like in 5-10 years?
- Is the market pricing in exponential growth or linear extrapolation?

Return exactly:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": float (0-100),
  "reasoning": "your first-principles thesis in your voice"
}}""",
            ),
        ]
    )

    prompt = template.invoke({
        "ticker": ticker,
        "price_context": price_context,
    })

    def create_default_signal():
        return ElonMuskSignal(
            signal="neutral",
            confidence=25.0,
            reasoning="Insufficient data to reason from first principles. Need more signal before sizing up.",
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=ElonMuskSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_signal,
    )
