"""
Market data endpoints - Indices, sectors, and market-wide analysis.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.v1.deps.auth import get_optional_user, ClerkUser
from app.services.market_data import get_market_data_service

logger = logging.getLogger(__name__)

router = APIRouter()


class MarketIndex(BaseModel):
    """Market index data."""

    symbol: str
    name: str
    value: float
    change: float
    change_percent: float
    high: float
    low: float
    volume: int | None = None


class SectorPerformance(BaseModel):
    """Sector performance data."""

    name: str
    change_percent: float
    top_gainer: str
    top_loser: str


class MarketRegimeResponse(BaseModel):
    """Current market regime analysis."""

    regime: str
    confidence: float
    trend: str
    duration_days: int
    signals: list[str]
    recommendation: str


class MarketBreadth(BaseModel):
    """Market breadth indicators."""

    advances: int
    declines: int
    unchanged: int
    advance_decline_ratio: float
    new_highs: int
    new_lows: int


@router.get("/indices", response_model=list[MarketIndex])
async def get_market_indices(
    user: ClerkUser | None = Depends(get_optional_user),
) -> list[MarketIndex]:
    """
    Get live data for major Indian market indices.
    """
    try:
        market_data = get_market_data_service()
        indices = await market_data.get_market_indices()
        return [
            MarketIndex(
                symbol=idx.symbol,
                name=idx.name,
                value=idx.value,
                change=idx.change,
                change_percent=idx.change_percent,
                high=0,  # yfinance indices don't always provide intraday high/low
                low=0,
            )
            for idx in indices
        ]
    except Exception as e:
        logger.error(f"Failed to fetch market indices: {e}")
        raise HTTPException(
            status_code=503,
            detail="Unable to fetch market indices. Market data service may be unavailable.",
        )


@router.get("/sectors", response_model=list[SectorPerformance])
async def get_sector_performance(
    user: ClerkUser | None = Depends(get_optional_user),
) -> list[SectorPerformance]:
    """
    Get performance of all sectors.
    """
    try:
        market_data = get_market_data_service()
        sectors = await market_data.get_sector_performance()
        return [
            SectorPerformance(
                name=s.name,
                change_percent=s.change_percent,
                top_gainer=s.top_gainer,
                top_loser=s.top_loser,
            )
            for s in sectors
        ]
    except Exception as e:
        logger.error(f"Failed to fetch sector performance: {e}")
        raise HTTPException(
            status_code=503,
            detail="Unable to fetch sector performance. Market data service may be unavailable.",
        )


@router.get("/regime", response_model=MarketRegimeResponse)
async def get_market_regime(
    user: ClerkUser | None = Depends(get_optional_user),
) -> MarketRegimeResponse:
    """
    Get current market regime analysis.

    Regimes:
    - bull_strong: Strong uptrend with high confidence
    - bull_weak: Uptrend with weakening momentum
    - bear_strong: Strong downtrend
    - bear_weak: Downtrend with potential reversal
    - sideways: Range-bound market
    - volatile: High volatility, unclear direction
    - recovery: Transitioning from bear to bull
    """
    # TODO: Phase 2 - Replace with ML-based regime detection
    return MarketRegimeResponse(
        regime="bull_strong",
        confidence=78,
        trend="uptrend",
        duration_days=45,
        signals=[
            "NIFTY above 200-DMA",
            "Positive FII flows for 15 consecutive days",
            "Advance-decline ratio above 1.5",
            "VIX below 15",
            "Midcaps outperforming largecaps",
        ],
        recommendation="Favor high-beta stocks and momentum strategies. "
        "Consider reducing hedges.",
    )


@router.get("/breadth", response_model=MarketBreadth)
async def get_market_breadth(
    user: ClerkUser | None = Depends(get_optional_user),
) -> MarketBreadth:
    """
    Get market breadth indicators.
    """
    # TODO: Phase 2 - Replace with real breadth data
    advances = 1245
    declines = 892
    return MarketBreadth(
        advances=advances,
        declines=declines,
        unchanged=156,
        advance_decline_ratio=round(advances / declines, 2),
        new_highs=89,
        new_lows=23,
    )


@router.get("/fii-dii")
async def get_fii_dii_data(
    user: ClerkUser | None = Depends(get_optional_user),
) -> dict:
    """
    Get FII/DII investment data.
    """
    # TODO: Phase 2 - Replace with real scraped FII/DII data
    return {
        "date": "2024-01-28",
        "fii": {
            "buy": 12500.45,
            "sell": 10200.30,
            "net": 2300.15,
            "net_mtd": 15400.00,
            "net_ytd": 45000.00,
        },
        "dii": {
            "buy": 8500.20,
            "sell": 9200.45,
            "net": -700.25,
            "net_mtd": 5200.00,
            "net_ytd": 22000.00,
        },
    }


@router.get("/status")
async def get_market_status(
    user: ClerkUser | None = Depends(get_optional_user),
) -> dict:
    """
    Get current market status (open/closed) and trading hours.
    """
    try:
        market_data = get_market_data_service()
        return await market_data.get_market_status()
    except Exception as e:
        logger.error(f"Failed to fetch market status: {e}")
        raise HTTPException(
            status_code=503,
            detail="Unable to fetch market status. Market data service may be unavailable.",
        )
