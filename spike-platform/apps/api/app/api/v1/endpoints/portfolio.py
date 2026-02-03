"""
Portfolio endpoints - Portfolio management, analysis, and optimization.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.v1.deps.auth import get_current_user, require_plan, ClerkUser

router = APIRouter()


class Holding(BaseModel):
    """Portfolio holding."""

    symbol: str
    name: str
    quantity: int
    avg_price: float
    current_price: float
    invested: float
    current_value: float
    returns: float
    returns_percent: float
    allocation: float
    finscore: float


class PortfolioSummary(BaseModel):
    """Portfolio summary."""

    total_value: float
    invested: float
    returns: float
    returns_percent: float
    today_change: float
    today_change_percent: float
    holding_count: int


class RiskMetrics(BaseModel):
    """Portfolio risk metrics."""

    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    beta: float
    var_95: float
    concentration_risk: float


class AddHoldingRequest(BaseModel):
    """Request to add a holding."""

    symbol: str
    quantity: int
    avg_price: float


@router.get("/summary", response_model=PortfolioSummary)
async def get_portfolio_summary(
    user: ClerkUser = Depends(get_current_user),
) -> PortfolioSummary:
    """Get portfolio summary for current user."""
    return PortfolioSummary(
        total_value=1245678.00,
        invested=1000000.00,
        returns=245678.00,
        returns_percent=24.57,
        today_change=29456.00,
        today_change_percent=2.42,
        holding_count=12,
    )


@router.get("/holdings", response_model=list[Holding])
async def get_holdings(
    user: ClerkUser = Depends(get_current_user),
) -> list[Holding]:
    """Get all holdings in the portfolio."""
    return [
        Holding(
            symbol="RELIANCE",
            name="Reliance Industries",
            quantity=50,
            avg_price=2200.00,
            current_price=2456.75,
            invested=110000.00,
            current_value=122837.50,
            returns=12837.50,
            returns_percent=11.67,
            allocation=9.86,
            finscore=8.4,
        ),
        Holding(
            symbol="TCS",
            name="Tata Consultancy Services",
            quantity=30,
            avg_price=3500.00,
            current_price=3845.20,
            invested=105000.00,
            current_value=115356.00,
            returns=10356.00,
            returns_percent=9.86,
            allocation=9.26,
            finscore=7.9,
        ),
    ]


@router.post("/holdings")
async def add_holding(
    request: AddHoldingRequest,
    user: ClerkUser = Depends(get_current_user),
) -> dict:
    """Add a new holding to the portfolio."""
    return {
        "status": "added",
        "holding": {
            "symbol": request.symbol.upper(),
            "quantity": request.quantity,
            "avg_price": request.avg_price,
        },
    }


@router.delete("/holdings/{symbol}")
async def remove_holding(
    symbol: str,
    user: ClerkUser = Depends(get_current_user),
) -> dict:
    """Remove a holding from the portfolio."""
    return {"status": "removed", "symbol": symbol.upper()}


@router.get("/risk", response_model=RiskMetrics)
async def get_risk_metrics(
    user: ClerkUser = Depends(get_current_user),
) -> RiskMetrics:
    """Get portfolio risk metrics."""
    return RiskMetrics(
        volatility=18.5,
        sharpe_ratio=1.45,
        max_drawdown=-12.3,
        beta=1.05,
        var_95=-25000.00,
        concentration_risk=35.0,
    )


@router.get("/allocation")
async def get_allocation(
    user: ClerkUser = Depends(get_current_user),
) -> dict:
    """Get portfolio allocation breakdown."""
    return {
        "by_sector": [
            {"sector": "IT", "allocation": 35.0, "value": 435987.00},
            {"sector": "Banking", "allocation": 25.0, "value": 311419.50},
            {"sector": "Oil & Gas", "allocation": 15.0, "value": 186851.70},
        ],
        "by_market_cap": [
            {"category": "Large Cap", "allocation": 70.0},
            {"category": "Mid Cap", "allocation": 25.0},
            {"category": "Small Cap", "allocation": 5.0},
        ],
    }


@router.get("/optimize")
async def get_optimization_suggestions(
    user: ClerkUser = Depends(require_plan(["pro", "pro_plus", "premium"])),
) -> dict:
    """
    Get AI-powered portfolio optimization suggestions.
    Pro feature.
    """
    return {
        "current_sharpe": 1.45,
        "optimized_sharpe": 1.72,
        "suggestions": [
            {
                "action": "reduce",
                "symbol": "TATASTEEL",
                "reason": "Overweight in metals sector (15% vs recommended 8%)",
                "target_allocation": 5.0,
            },
            {
                "action": "increase",
                "symbol": "HDFCBANK",
                "reason": "Underweight in banking sector, high FinScore",
                "target_allocation": 12.0,
            },
            {
                "action": "add",
                "symbol": "SUNPHARMA",
                "reason": "Diversification into defensive sector, FinScore 8.2",
                "target_allocation": 5.0,
            },
        ],
        "expected_improvement": {
            "volatility_reduction": -2.5,
            "sharpe_improvement": 0.27,
            "regime_fit_improvement": 15.0,
        },
    }


@router.post("/rebalance")
async def execute_rebalance(
    user: ClerkUser = Depends(require_plan(["premium"])),
) -> dict:
    """
    Execute portfolio rebalancing based on optimization.
    Premium feature - requires broker integration.
    """
    return {
        "status": "pending",
        "message": "Rebalancing requires broker authorization. "
        "Please connect your broker account.",
    }
