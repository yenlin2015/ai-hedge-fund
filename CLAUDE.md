# AI Hedge Fund

An AI-powered hedge fund simulator using multiple LangGraph agents to make trading decisions. Educational and research purposes only - NOT for real trading.

## Project Structure

- `src/` - Core Python application (agents, backtesting, CLI, data, graph, tools, utils)
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
# Run the hedge fund CLI
poetry run python src/main.py --ticker AAPL,MSFT,NVDA

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

## Environment Variables

Copy `.env.example` to `.env` and configure API keys for the LLM providers and Financial Datasets API you plan to use.
