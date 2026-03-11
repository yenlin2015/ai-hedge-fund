# AI Hedge Fund

An AI-powered hedge fund simulator using multiple LangGraph agents to make trading decisions. Supports both **equities** and **crypto** assets. Educational and research purposes only - NOT for real trading.

## Project Structure

- `src/` - Core Python application (agents, backtesting, CLI, data, graph, tools, utils)
- `src/agents/` - Analyst agents (equity + crypto)
- `src/tools/` - Data APIs (`api.py` routes to equity or crypto adapters automatically)
- `src/tools/crypto_api.py` - Crypto price/metrics via OKX + CoinGecko (no API keys needed)
- `src/tools/twitter_api.py` - KOL tweet fetcher via Apify (for sentiment agents)
- `app/backend/` - FastAPI backend API
- `app/frontend/` - React/Vite frontend
- `tests/` - Test suite (pytest)

## Setup

```bash
poetry install                          # Install Python dependencies
cd app/frontend && npm install          # Install frontend dependencies
```

## Common Commands

```bash
# Run the hedge fund CLI (equities)
poetry run python src/main.py --ticker AAPL,MSFT,NVDA

# Run the hedge fund CLI (crypto)
poetry run python src/main.py --ticker BTC,ETH,SOL --analysts arthur_hayes,wang_duan_niao

# Run backtester
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA

# Run backend API server
poetry run uvicorn app.backend.main:app --reload

# Run frontend dev server
cd app/frontend && npm run dev
```

## Testing

```bash
poetry run pytest                       # Run all tests
poetry run pytest tests/<file> -x       # Run a specific test file
```

## Linting & Formatting

```bash
poetry run flake8 <file_or_dir>         # Lint
poetry run black <file_or_dir>          # Format (line-length: 420)
poetry run isort <file_or_dir>          # Sort imports (profile: black)
```

## Key Architecture

- **LangGraph workflow**: `start_node -> [selected analyst agents in parallel] -> risk_manager -> portfolio_manager -> END`
- **Agent state**: Defined in `src/graph/state.py` (AgentState TypedDict)
- **Analyst registry**: `src/utils/analysts.py` is the single source of truth for all available analysts
- **LLM configuration**: `src/llm/models.py` with model definitions in `src/llm/api_models.json` and `src/llm/ollama_models.json`
- **Unified data layer**: `src/tools/api.py` auto-detects crypto tickers via `is_crypto_ticker()` and routes to the crypto adapter (`src/tools/crypto_api.py`), so all agents work with both equities and crypto without modification

## Crypto Support

Crypto tickers are automatically detected and routed to OKX (OHLCV data) + CoinGecko (market cap). No API keys needed for crypto data.

**Supported tickers**: BTC, ETH, SOL, DOGE, XRP, ADA, AVAX, LINK, DOT, MATIC, AAVE, UNI, NEAR, MKR, LDO, OP, ARB, SUI, SEI, APT, TIA, INJ, FET, RENDER, TON, ATOM, FIL, PEPE, WIF, BONK

**Crypto-focused agents**:
- `wang_duan_niao` - Meme/narrative hunter, uses Crypto Twitter sentiment via Apify
- `arthur_hayes` - Macro-liquidity cycle analysis (central bank policy, DXY, yield curves)
- `elon_musk` - First-principles tech disruption analysis (S-curves, AI, manufacturing scale)

## Environment Variables

Copy `.env.example` to `.env` and configure:
- **LLM provider keys** - For the AI models you plan to use
- **Financial Datasets API key** - For equity market data
- **APIFY_API_TOKEN** (optional) - For fetching KOL tweets, used by `wang_duan_niao` agent. Get from https://console.apify.com/account/integrations
