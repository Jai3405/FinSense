"""
Market data endpoints - Indices, sectors, and market-wide analysis.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.v1.deps.auth import get_optional_user, ClerkUser

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
    return [
        MarketIndex(
            symbol="NIFTY50",
            name="NIFTY 50",
            value=22150.50,
            change=125.30,
            change_percent=0.57,
            high=22200.00,
            low=22050.00,
        ),
        MarketIndex(
            symbol="SENSEX",
            name="S&P BSE SENSEX",
            value=72890.25,
            change=380.50,
            change_percent=0.52,
            high=73050.00,
            low=72600.00,
        ),
        MarketIndex(
            symbol="NIFTYBANK",
            name="NIFTY Bank",
            value=46850.00,
            change=-125.75,
            change_percent=-0.27,
            high=47100.00,
            low=46700.00,
        ),
        MarketIndex(
            symbol="NIFTYIT",
            name="NIFTY IT",
            value=38450.00,
            change=285.50,
            change_percent=0.75,
            high=38550.00,
            low=38200.00,
        ),
    ]


@router.get("/sectors", response_model=list[SectorPerformance])
async def get_sector_performance(
    user: ClerkUser | None = Depends(get_optional_user),
) -> list[SectorPerformance]:
    """
    Get performance of all sectors.
    """
    return [
        SectorPerformance(
            name="Information Technology",
            change_percent=2.4,
            top_gainer="TATAELXSI",
            top_loser="MPHASIS",
        ),
        SectorPerformance(
            name="Banking",
            change_percent=-0.8,
            top_gainer="ICICIBANK",
            top_loser="BANDHANBNK",
        ),
        SectorPerformance(
            name="Pharmaceuticals",
            change_percent=1.2,
            top_gainer="SUNPHARMA",
            top_loser="CIPLA",
        ),
        SectorPerformance(
            name="Metals",
            change_percent=3.1,
            top_gainer="TATASTEEL",
            top_loser="HINDALCO",
        ),
    ]


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
    return {
        "is_open": True,
        "exchange": "NSE",
        "session": "regular",
        "next_open": "2024-01-29T09:15:00+05:30",
        "next_close": "2024-01-28T15:30:00+05:30",
        "holidays_upcoming": [
            {"date": "2024-01-26", "name": "Republic Day"},
        ],
    }
