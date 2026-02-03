"""
FinScore endpoints - SPIKE's proprietary stock rating system.
Universal 0-10 score combining 9 dimensions of analysis.
"""

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.api.v1.deps.auth import get_current_user, ClerkUser

router = APIRouter()


class FinScoreComponents(BaseModel):
    """9 dimensions of FinScore analysis."""

    quality: float  # Fundamentals (ROE, ROCE, debt ratios)
    momentum: float  # Price momentum (RSI, MACD, trend)
    value: float  # Valuation (PE, PB, EV/EBITDA vs sector)
    sentiment: float  # News & social sentiment
    risk: float  # Volatility, beta, max drawdown
    flow: float  # FII/DII flows, delivery %
    regime_fit: float  # How well stock fits current market regime
    sector: float  # Sector dynamics and rotation
    technical: float  # Technical signals and patterns


class FinScoreResponse(BaseModel):
    """Complete FinScore response for a stock."""

    symbol: str
    name: str
    overall_score: float  # 0-10
    signal: str  # "strong_buy", "buy", "hold", "sell", "strong_sell"
    confidence: float  # 0-100
    components: FinScoreComponents
    regime: str  # Current market regime
    insights: list[str]
    updated_at: str


class FinScoreHistoryPoint(BaseModel):
    """Historical FinScore data point."""

    date: str
    score: float
    signal: str


class TopFinScoreStock(BaseModel):
    """Stock in top FinScore rankings."""

    rank: int
    symbol: str
    name: str
    sector: str
    score: float
    signal: str
    change_1d: float
    change_1w: float


@router.get("/{symbol}", response_model=FinScoreResponse)
async def get_finscore(
    symbol: str,
    user: ClerkUser = Depends(get_current_user),
) -> FinScoreResponse:
    """
    Get complete FinScore analysis for a stock.

    FinScore is SPIKE's proprietary 0-10 rating that synthesizes:
    - Quality & Fundamentals
    - Price Momentum
    - Value & Valuation
    - Sentiment Analysis
    - Risk Assessment
    - Institutional Flows
    - Regime Fit
    - Sector Dynamics
    - Technical Signals
    """
    # TODO: Implement actual FinScore calculation
    return FinScoreResponse(
        symbol=symbol.upper(),
        name=f"{symbol.upper()} Ltd",
        overall_score=8.4,
        signal="strong_buy",
        confidence=85,
        components=FinScoreComponents(
            quality=8.5,
            momentum=8.2,
            value=7.8,
            sentiment=8.0,
            risk=8.8,
            flow=9.0,
            regime_fit=8.5,
            sector=8.2,
            technical=8.0,
        ),
        regime="bull_strong",
        insights=[
            "Strong fundamentals with consistent ROE above 20%",
            "Positive momentum with price above all major MAs",
            "FII buying continues with 2.5% stake increase this month",
            "Well positioned for current bullish market regime",
            "Technical breakout above resistance with volume confirmation",
        ],
        updated_at="2024-01-28T15:30:00+05:30",
    )


@router.get("/{symbol}/history", response_model=list[FinScoreHistoryPoint])
async def get_finscore_history(
    symbol: str,
    period: str = Query("3M", regex="^(1M|3M|6M|1Y)$"),
    user: ClerkUser = Depends(get_current_user),
) -> list[FinScoreHistoryPoint]:
    """
    Get historical FinScore data for trend analysis.
    """
    # TODO: Implement actual historical data
    return [
        FinScoreHistoryPoint(date="2024-01-28", score=8.4, signal="strong_buy"),
        FinScoreHistoryPoint(date="2024-01-21", score=8.2, signal="buy"),
        FinScoreHistoryPoint(date="2024-01-14", score=7.8, signal="buy"),
        FinScoreHistoryPoint(date="2024-01-07", score=7.5, signal="buy"),
    ]


@router.get("/rankings/top", response_model=list[TopFinScoreStock])
async def get_top_finscore_stocks(
    sector: str | None = Query(None, description="Filter by sector"),
    market_cap: str | None = Query(None, regex="^(small|mid|large)$"),
    limit: int = Query(20, ge=1, le=100),
    user: ClerkUser = Depends(get_current_user),
) -> list[TopFinScoreStock]:
    """
    Get stocks with highest FinScore ratings.
    Useful for discovering investment opportunities.
    """
    # TODO: Implement actual ranking query
    return [
        TopFinScoreStock(
            rank=1,
            symbol="RELIANCE",
            name="Reliance Industries",
            sector="Oil & Gas",
            score=8.7,
            signal="strong_buy",
            change_1d=1.17,
            change_1w=3.45,
        ),
        TopFinScoreStock(
            rank=2,
            symbol="TCS",
            name="Tata Consultancy Services",
            sector="IT",
            score=8.5,
            signal="strong_buy",
            change_1d=0.85,
            change_1w=2.12,
        ),
        TopFinScoreStock(
            rank=3,
            symbol="HDFCBANK",
            name="HDFC Bank",
            sector="Banking",
            score=8.4,
            signal="strong_buy",
            change_1d=0.95,
            change_1w=1.89,
        ),
    ]


@router.get("/compare")
async def compare_finscores(
    symbols: str = Query(..., description="Comma-separated symbols (max 5)"),
    user: ClerkUser = Depends(get_current_user),
) -> dict:
    """
    Compare FinScores of multiple stocks.
    Useful for relative analysis and stock selection.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")][:5]

    # TODO: Implement actual comparison
    return {
        "stocks": [
            {
                "symbol": sym,
                "score": 8.0 - (i * 0.3),
                "signal": "buy",
                "components": {
                    "quality": 8.5 - (i * 0.2),
                    "momentum": 8.0 - (i * 0.1),
                    "value": 7.8,
                },
            }
            for i, sym in enumerate(symbol_list)
        ],
        "recommendation": f"Based on FinScore, {symbol_list[0]} is the top pick.",
    }


@router.post("/alert")
async def create_finscore_alert(
    symbol: str,
    threshold: float = Query(..., ge=0, le=10),
    direction: str = Query(..., regex="^(above|below)$"),
    user: ClerkUser = Depends(get_current_user),
) -> dict:
    """
    Create an alert when FinScore crosses a threshold.
    """
    # TODO: Implement alert creation
    return {
        "status": "created",
        "alert": {
            "symbol": symbol.upper(),
            "threshold": threshold,
            "direction": direction,
            "current_score": 8.4,
        },
    }
