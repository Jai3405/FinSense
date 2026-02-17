"""
Stock endpoints - Stock information, quotes, and search.
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.api.v1.deps.auth import get_current_user, get_optional_user, ClerkUser
from app.services.market_data import get_market_data_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Period mapping: frontend format -> yfinance format
PERIOD_MAP = {
    "1D": "1d",
    "1W": "5d",
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "1Y": "1y",
    "5Y": "5y",
    "MAX": "max",
}


def ensure_ns_suffix(symbol: str) -> str:
    """Add .NS suffix for NSE stocks if not already present."""
    if not symbol.endswith((".NS", ".BO")):
        return f"{symbol}.NS"
    return symbol


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
    try:
        market_data = get_market_data_service()
        results = await market_data.search_stocks(q)
        return [
            StockSearchResult(
                symbol=r.symbol,
                name=r.name,
                exchange=r.exchange,
                sector=r.type,  # SearchResult.type maps to sector field
            )
            for r in results[:limit]
        ]
    except Exception as e:
        logger.error(f"Stock search failed for query '{q}': {e}")
        raise HTTPException(
            status_code=503,
            detail="Market data service is temporarily unavailable. Please try again.",
        )


@router.get("/{symbol}/quote", response_model=StockQuote)
async def get_stock_quote(
    symbol: str,
    user: ClerkUser | None = Depends(get_optional_user),
) -> StockQuote:
    """
    Get real-time quote for a stock.
    Includes current price, change, volume, and OHLC.
    """
    try:
        market_data = get_market_data_service()
        yf_symbol = ensure_ns_suffix(symbol)
        quote = await market_data.get_stock_quote(yf_symbol)
        return StockQuote(
            symbol=symbol.upper(),
            name=quote.name,
            exchange="NSE",
            price=quote.price,
            change=quote.change,
            change_percent=quote.change_percent,
            volume=quote.volume,
            high=quote.high,
            low=quote.low,
            open=quote.open,
            prev_close=quote.prev_close,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        logger.error(f"Failed to fetch quote for {symbol}: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Unable to fetch quote for {symbol}. Market data service may be unavailable.",
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
    try:
        market_data = get_market_data_service()
        yf_symbol = ensure_ns_suffix(symbol)
        quote = await market_data.get_stock_quote(yf_symbol)

        # Fetch extra info directly from yfinance ticker
        import yfinance as yf

        ticker = yf.Ticker(yf_symbol)
        info = ticker.info

        return StockInfo(
            symbol=symbol.upper(),
            name=quote.name,
            exchange="NSE",
            sector=info.get("sector", ""),
            industry=info.get("industry", ""),
            market_cap=quote.market_cap or 0,
            pe_ratio=quote.pe_ratio,
            pb_ratio=info.get("priceToBook"),
            dividend_yield=info.get("dividendYield"),
            eps=info.get("trailingEps"),
            high_52w=quote.week_52_high or 0,
            low_52w=quote.week_52_low or 0,
            avg_volume=info.get("averageVolume", 0) or 0,
            description=info.get("longBusinessSummary"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch info for {symbol}: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Unable to fetch info for {symbol}. Market data service may be unavailable.",
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
    try:
        market_data = get_market_data_service()
        yf_symbol = ensure_ns_suffix(symbol)
        yf_period = PERIOD_MAP.get(period, "1mo")

        bars = await market_data.get_historical(yf_symbol, yf_period, interval)
        return {
            "symbol": symbol.upper(),
            "period": period,
            "interval": interval,
            "data": [
                {
                    "timestamp": bar.timestamp.isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
                for bar in bars
            ],
        }
    except Exception as e:
        logger.error(f"Failed to fetch history for {symbol}: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Unable to fetch history for {symbol}. Market data service may be unavailable.",
        )


@router.get("/trending")
async def get_trending_stocks(
    limit: int = Query(10, ge=1, le=50),
    user: ClerkUser | None = Depends(get_optional_user),
) -> list[dict]:
    """
    Get trending stocks based on volume and price movement.
    """
    try:
        market_data = get_market_data_service()
        quotes = await market_data.get_trending(limit)
        return [
            {
                "symbol": q.symbol.replace(".NS", "").replace(".BO", ""),
                "name": q.name,
                "price": q.price,
                "change_percent": q.change_percent,
                "volume": q.volume,
                "trend_score": round(abs(q.change_percent) * 10, 1),
            }
            for q in quotes
        ]
    except Exception as e:
        logger.error(f"Failed to fetch trending stocks: {e}")
        raise HTTPException(
            status_code=503,
            detail="Unable to fetch trending stocks. Market data service may be unavailable.",
        )
