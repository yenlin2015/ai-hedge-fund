"""Crypto data adapter using OKX public API (prices) and CoinGecko (market data).

Provides the same interfaces as the equities API so all existing agents
work with crypto tickers without modification.
"""

import datetime
import requests
import time

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    FinancialMetrics,
    InsiderTrade,
    LineItem,
    Price,
)

_cache = get_cache()

# ── Ticker mapping ──────────────────────────────────────────────────────────

CRYPTO_TICKER_MAP = {
    "BTC": {"coingecko": "bitcoin", "okx": "BTC-USDT"},
    "ETH": {"coingecko": "ethereum", "okx": "ETH-USDT"},
    "SOL": {"coingecko": "solana", "okx": "SOL-USDT"},
    "DOGE": {"coingecko": "dogecoin", "okx": "DOGE-USDT"},
    "XRP": {"coingecko": "ripple", "okx": "XRP-USDT"},
    "ADA": {"coingecko": "cardano", "okx": "ADA-USDT"},
    "AVAX": {"coingecko": "avalanche-2", "okx": "AVAX-USDT"},
    "LINK": {"coingecko": "chainlink", "okx": "LINK-USDT"},
    "DOT": {"coingecko": "polkadot", "okx": "DOT-USDT"},
    "MATIC": {"coingecko": "matic-network", "okx": "MATIC-USDT"},
    "PEPE": {"coingecko": "pepe", "okx": "PEPE-USDT"},
    "UNI": {"coingecko": "uniswap", "okx": "UNI-USDT"},
    "NEAR": {"coingecko": "near", "okx": "NEAR-USDT"},
    "APT": {"coingecko": "aptos", "okx": "APT-USDT"},
    "OP": {"coingecko": "optimism", "okx": "OP-USDT"},
    "ARB": {"coingecko": "arbitrum", "okx": "ARB-USDT"},
    "SUI": {"coingecko": "sui", "okx": "SUI-USDT"},
    "SEI": {"coingecko": "sei-network", "okx": "SEI-USDT"},
    "TIA": {"coingecko": "celestia", "okx": "TIA-USDT"},
    "INJ": {"coingecko": "injective-protocol", "okx": "INJ-USDT"},
    "FET": {"coingecko": "fetch-ai", "okx": "FET-USDT"},
    "RENDER": {"coingecko": "render-token", "okx": "RENDER-USDT"},
    "WIF": {"coingecko": "dogwifcoin", "okx": "WIF-USDT"},
    "BONK": {"coingecko": "bonk", "okx": "BONK-USDT"},
    "AAVE": {"coingecko": "aave", "okx": "AAVE-USDT"},
    "MKR": {"coingecko": "maker", "okx": "MKR-USDT"},
    "LDO": {"coingecko": "lido-dao", "okx": "LDO-USDT"},
    "TON": {"coingecko": "the-open-network", "okx": "TON-USDT"},
    "ATOM": {"coingecko": "cosmos", "okx": "ATOM-USDT"},
    "FIL": {"coingecko": "filecoin", "okx": "FIL-USDT"},
}


def is_crypto_ticker(ticker: str) -> bool:
    return ticker.upper() in CRYPTO_TICKER_MAP


# ── Helpers ─────────────────────────────────────────────────────────────────

def _date_to_ms(date_str: str) -> int:
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp() * 1000)


def _ms_to_date(ms: int | str) -> str:
    return datetime.datetime.utcfromtimestamp(int(ms) / 1000).strftime("%Y-%m-%d")


def _okx_request(url: str, max_retries: int = 3) -> dict | None:
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 429 and attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code != 200:
                return None
            data = resp.json()
            if data.get("code") != "0":
                return None
            return data
        except (requests.RequestException, ValueError):
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            return None
    return None


def _coingecko_request(url: str, max_retries: int = 3) -> dict | None:
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, timeout=15, headers={"Accept": "application/json"})
            if resp.status_code == 429 and attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code != 200:
                return None
            return resp.json()
        except (requests.RequestException, ValueError):
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            return None
    return None


# ── Price data (OKX) ────────────────────────────────────────────────────────

def get_crypto_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch daily OHLCV candles from OKX public API."""
    ticker = ticker.upper()
    cache_key = f"crypto_{ticker}_{start_date}_{end_date}"

    if cached := _cache.get_prices(cache_key):
        return [Price(**p) for p in cached]

    mapping = CRYPTO_TICKER_MAP.get(ticker)
    if not mapping:
        return []

    inst_id = mapping["okx"]
    start_ms = _date_to_ms(start_date)
    end_ms = _date_to_ms(end_date) + 86400_000  # include end_date

    all_candles = []
    current_after = end_ms

    # OKX returns newest-first; paginate backwards with `after` param
    while True:
        url = (
            f"https://www.okx.com/api/v5/market/history-candles"
            f"?instId={inst_id}&bar=1D&limit=100&after={current_after}"
        )
        data = _okx_request(url)
        if not data or not data.get("data"):
            break

        candles = data["data"]
        all_candles.extend(candles)

        # OKX data is newest-first, so the oldest candle is last
        oldest_ts = int(candles[-1][0])
        if oldest_ts <= start_ms or len(candles) < 100:
            break

        current_after = oldest_ts

    if not all_candles:
        return []

    # Convert to Price objects, filter to date range
    prices = []
    for c in all_candles:
        ts = int(c[0])
        if ts < start_ms or ts > end_ms:
            continue
        date_str = _ms_to_date(ts)
        prices.append(Price(
            open=float(c[1]),
            high=float(c[2]),
            low=float(c[3]),
            close=float(c[4]),
            volume=int(float(c[5])),
            time=date_str,
        ))

    # Sort oldest-first (to match equities API convention)
    prices.sort(key=lambda p: p.time)

    # Deduplicate by date
    seen = set()
    unique_prices = []
    for p in prices:
        if p.time not in seen:
            seen.add(p.time)
            unique_prices.append(p)
    prices = unique_prices

    if prices:
        _cache.set_prices(cache_key, [p.model_dump() for p in prices])

    return prices


# ── Market cap (CoinGecko) ──────────────────────────────────────────────────

def get_crypto_market_cap(ticker: str, end_date: str) -> float | None:
    """Fetch current market cap from CoinGecko."""
    ticker = ticker.upper()
    mapping = CRYPTO_TICKER_MAP.get(ticker)
    if not mapping:
        return None

    coin_id = mapping["coingecko"]
    url = (
        f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        f"?localization=false&tickers=false&community_data=false&developer_data=false"
    )
    data = _coingecko_request(url)
    if not data:
        return None

    try:
        return data["market_data"]["market_cap"]["usd"]
    except (KeyError, TypeError):
        return None


# ── Financial metrics (synthetic for crypto) ────────────────────────────────

def get_crypto_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """Return a single FinancialMetrics with market_cap filled; all traditional fields None."""
    ticker = ticker.upper()
    market_cap = get_crypto_market_cap(ticker, end_date)

    return [
        FinancialMetrics(
            ticker=ticker,
            report_period=end_date,
            period=period,
            currency="USD",
            market_cap=market_cap,
            enterprise_value=None,
            price_to_earnings_ratio=None,
            price_to_book_ratio=None,
            price_to_sales_ratio=None,
            enterprise_value_to_ebitda_ratio=None,
            enterprise_value_to_revenue_ratio=None,
            free_cash_flow_yield=None,
            peg_ratio=None,
            gross_margin=None,
            operating_margin=None,
            net_margin=None,
            return_on_equity=None,
            return_on_assets=None,
            return_on_invested_capital=None,
            asset_turnover=None,
            inventory_turnover=None,
            receivables_turnover=None,
            days_sales_outstanding=None,
            operating_cycle=None,
            working_capital_turnover=None,
            current_ratio=None,
            quick_ratio=None,
            cash_ratio=None,
            operating_cash_flow_ratio=None,
            debt_to_equity=None,
            debt_to_assets=None,
            interest_coverage=None,
            revenue_growth=None,
            earnings_growth=None,
            book_value_growth=None,
            earnings_per_share_growth=None,
            free_cash_flow_growth=None,
            operating_income_growth=None,
            ebitda_growth=None,
            payout_ratio=None,
            earnings_per_share=None,
            book_value_per_share=None,
            free_cash_flow_per_share=None,
        )
    ]


# ── Line items, insider trades, news (not applicable for crypto) ────────────

def get_crypto_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    """Crypto has no financial statements. Return empty list."""
    return []


def get_crypto_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """No insider trades for crypto. Return empty list."""
    return []


def get_crypto_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[CompanyNews]:
    """Crypto news — returns empty for now. Can plug in CryptoCompare/LunarCrush later."""
    return []
