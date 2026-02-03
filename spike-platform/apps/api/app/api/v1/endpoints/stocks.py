"""
Stock endpoints - Stock information, quotes, and search.
"""

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.api.v1.deps.auth import get_current_user, get_optional_user, ClerkUser

router = APIRouter()


class StockQuote(BaseModel):
    """Real-time stock quote."""

    symbol: str
    name: str
    exchange: str
    price: float
    change: float
    change_percent: float
    volume: int
    high: float
    low: float
    open: float
    prev_close: float
    timestamp: str


class StockInfo(BaseModel):
    """Detailed stock information."""

    symbol: str
    name: str
    exchange: str
    sector: str
    industry: str
    market_cap: float
    pe_ratio: float | None
    pb_ratio: float | None
    dividend_yield: float | None
    eps: float | None
    high_52w: float
    low_52w: float
    avg_volume: int
    description: str | None


class StockSearchResult(BaseModel):
    """Stock search result."""

    symbol: str
    name: str
    exchange: str
    sector: str


@router.get("/search", response_model=list[StockSearchResult])
async def search_stocks(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=50),
    user: ClerkUser | None = Depends(get_optional_user),
) -> list[StockSearchResult]:
    """
    Search stocks by symbol or name.
    Returns matching stocks from NSE/BSE.
    """
    # TODO: Implement actual search against database/Meilisearch
    # Mock response for now
    mock_results = [
        StockSearchResult(
            symbol="RELIANCE",
            name="Reliance Industries Ltd",
            exchange="NSE",
            sector="Oil & Gas",
        ),
        StockSearchResult(
            symbol="TCS",
            name="Tata Consultancy Services Ltd",
            exchange="NSE",
            sector="Information Technology",
        ),
        StockSearchResult(
            symbol="HDFCBANK",
            name="HDFC Bank Ltd",
            exchange="NSE",
            sector="Financial Services",
        ),
    ]

    # Filter by query
    q_lower = q.lower()
    return [
        r
        for r in mock_results
        if q_lower in r.symbol.lower() or q_lower in r.name.lower()
    ][:limit]


@router.get("/{symbol}/quote", response_model=StockQuote)
async def get_stock_quote(
    symbol: str,
    user: ClerkUser | None = Depends(get_optional_user),
) -> StockQuote:
    """
    Get real-time quote for a stock.
    Includes current price, change, volume, and OHLC.
    """
    # TODO: Implement real market data fetch
    return StockQuote(
        symbol=symbol.upper(),
        name=f"{symbol.upper()} Stock",
        exchange="NSE",
        price=2456.75,
        change=28.50,
        change_percent=1.17,
        volume=5234567,
        high=2478.90,
        low=2432.10,
        open=2445.00,
        prev_close=2428.25,
        timestamp="2024-01-28T15:30:00+05:30",
    )


@router.get("/{symbol}/info", response_model=StockInfo)
async def get_stock_info(
    symbol: str,
    user: ClerkUser = Depends(get_current_user),
) -> StockInfo:
    """
    Get detailed information about a stock.
    Requires authentication.
    """
    # TODO: Implement real data fetch
    return StockInfo(
        symbol=symbol.upper(),
        name=f"{symbol.upper()} Ltd",
        exchange="NSE",
        sector="Technology",
        industry="IT Services",
        market_cap=15000000000000,
        pe_ratio=28.5,
        pb_ratio=8.2,
        dividend_yield=1.2,
        eps=85.50,
        high_52w=2890.00,
        low_52w=1980.00,
        avg_volume=4500000,
        description="Leading company in its sector.",
    )


@router.get("/{symbol}/history")
async def get_stock_history(
    symbol: str,
    period: str = Query("1M", regex="^(1D|1W|1M|3M|6M|1Y|5Y|MAX)$"),
    interval: str = Query("1d", regex="^(1m|5m|15m|1h|1d|1wk|1mo)$"),
    user: ClerkUser = Depends(get_current_user),
) -> dict:
    """
    Get historical OHLCV data for a stock.
    Requires authentication.
    """
    # TODO: Implement real historical data fetch
    return {
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "data": [
            {
                "timestamp": "2024-01-28",
                "open": 2445.00,
                "high": 2478.90,
                "low": 2432.10,
                "close": 2456.75,
                "volume": 5234567,
            },
            {
                "timestamp": "2024-01-27",
                "open": 2420.00,
                "high": 2450.00,
                "low": 2415.00,
                "close": 2428.25,
                "volume": 4890234,
            },
        ],
    }


@router.get("/trending")
async def get_trending_stocks(
    limit: int = Query(10, ge=1, le=50),
    user: ClerkUser | None = Depends(get_optional_user),
) -> list[dict]:
    """
    Get trending stocks based on volume and price movement.
    """
    return [
        {
            "symbol": "TATAELXSI",
            "name": "Tata Elxsi",
            "price": 7245.50,
            "change_percent": 8.09,
            "volume": "2.3L",
            "trend_score": 95,
        },
        {
            "symbol": "IRCTC",
            "name": "IRCTC",
            "price": 892.75,
            "change_percent": 6.47,
            "volume": "5.1L",
            "trend_score": 88,
        },
    ]
