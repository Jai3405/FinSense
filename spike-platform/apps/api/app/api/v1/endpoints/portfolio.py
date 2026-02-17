"""
Portfolio endpoints - Portfolio management, analysis, and optimization.
"""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.deps.auth import get_current_user, require_plan, ClerkUser
from app.db.session import get_db
from app.models.portfolio import Portfolio as PortfolioModel, Holding as HoldingModel
from app.models.watchlist import Watchlist as WatchlistModel
from app.models.user import User
from app.services.market_data import get_market_data_service

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_ns_suffix(symbol: str) -> str:
    """Add .NS suffix for yfinance if not already present."""
    if not symbol.endswith((".NS", ".BO")):
        return f"{symbol}.NS"
    return symbol


def strip_ns_suffix(symbol: str) -> str:
    """Remove .NS/.BO suffix for display."""
    for suffix in (".NS", ".BO"):
        if symbol.endswith(suffix):
            return symbol[: -len(suffix)]
    return symbol


async def ensure_user_setup(user: ClerkUser, db: AsyncSession):
    """Ensure user, default portfolio, and default watchlist exist in DB."""
    result = await db.execute(select(User).where(User.id == user.id))
    db_user = result.scalar_one_or_none()

    if not db_user:
        db_user = User(
            id=user.id,
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
        )
        db.add(db_user)
        await db.flush()

    # Ensure default portfolio
    result = await db.execute(
        select(PortfolioModel).where(PortfolioModel.user_id == user.id)
    )
    portfolio = result.scalar_one_or_none()
    if not portfolio:
        portfolio = PortfolioModel(user_id=user.id, name="My Portfolio")
        db.add(portfolio)
        await db.flush()

    # Ensure default watchlist
    result = await db.execute(
        select(WatchlistModel).where(WatchlistModel.user_id == user.id)
    )
    watchlist = result.scalar_one_or_none()
    if not watchlist:
        watchlist = WatchlistModel(user_id=user.id, name="My Watchlist")
        db.add(watchlist)
        await db.flush()

    return db_user, portfolio, watchlist


# ---------------------------------------------------------------------------
# Pydantic response models (unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/summary", response_model=PortfolioSummary)
async def get_portfolio_summary(
    user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PortfolioSummary:
    """Get portfolio summary for current user."""
    _, portfolio, _ = await ensure_user_setup(user, db)

    # Fetch holdings
    result = await db.execute(
        select(HoldingModel).where(HoldingModel.portfolio_id == portfolio.id)
    )
    holdings = result.scalars().all()

    if not holdings:
        return PortfolioSummary(
            total_value=0.0,
            invested=0.0,
            returns=0.0,
            returns_percent=0.0,
            today_change=0.0,
            today_change_percent=0.0,
            holding_count=0,
        )

    market_data = get_market_data_service()

    # Fetch live quotes for all holdings concurrently
    async def _safe_quote(h: HoldingModel):
        try:
            return h, await market_data.get_stock_quote(ensure_ns_suffix(h.symbol))
        except Exception as exc:
            logger.warning("Could not fetch quote for %s: %s", h.symbol, exc)
            return h, None

    results = await asyncio.gather(*[_safe_quote(h) for h in holdings])

    total_value = 0.0
    total_invested = 0.0
    today_change = 0.0

    for holding, quote in results:
        invested = holding.quantity * holding.avg_price
        total_invested += invested
        if quote:
            current_value = holding.quantity * quote.price
            total_value += current_value
            today_change += holding.quantity * quote.change
        else:
            # Fall back to avg_price if quote unavailable
            total_value += invested

    returns = total_value - total_invested
    returns_percent = (returns / total_invested * 100) if total_invested else 0.0
    today_change_percent = (today_change / (total_value - today_change) * 100) if (total_value - today_change) else 0.0

    return PortfolioSummary(
        total_value=round(total_value, 2),
        invested=round(total_invested, 2),
        returns=round(returns, 2),
        returns_percent=round(returns_percent, 2),
        today_change=round(today_change, 2),
        today_change_percent=round(today_change_percent, 2),
        holding_count=len(holdings),
    )


@router.get("/holdings", response_model=list[Holding])
async def get_holdings(
    user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[Holding]:
    """Get all holdings in the portfolio."""
    _, portfolio, _ = await ensure_user_setup(user, db)

    result = await db.execute(
        select(HoldingModel).where(HoldingModel.portfolio_id == portfolio.id)
    )
    holdings = result.scalars().all()

    if not holdings:
        return []

    market_data = get_market_data_service()

    # Fetch live quotes concurrently
    async def _safe_quote(h: HoldingModel):
        try:
            return h, await market_data.get_stock_quote(ensure_ns_suffix(h.symbol))
        except Exception as exc:
            logger.warning("Could not fetch quote for %s: %s", h.symbol, exc)
            return h, None

    results = await asyncio.gather(*[_safe_quote(h) for h in holdings])

    # First pass: compute total value for allocation percentages
    total_value = 0.0
    enriched = []
    for holding, quote in results:
        price = quote.price if quote else holding.avg_price
        current_value = holding.quantity * price
        total_value += current_value
        enriched.append((holding, quote, current_value))

    # Second pass: build response objects
    response: list[Holding] = []
    for holding, quote, current_value in enriched:
        invested = holding.quantity * holding.avg_price
        price = quote.price if quote else holding.avg_price
        name = quote.name if quote else holding.symbol
        returns = current_value - invested
        returns_percent = (returns / invested * 100) if invested else 0.0
        allocation = (current_value / total_value * 100) if total_value else 0.0

        response.append(
            Holding(
                symbol=strip_ns_suffix(holding.symbol),
                name=strip_ns_suffix(name),
                quantity=holding.quantity,
                avg_price=round(holding.avg_price, 2),
                current_price=round(price, 2),
                invested=round(invested, 2),
                current_value=round(current_value, 2),
                returns=round(returns, 2),
                returns_percent=round(returns_percent, 2),
                allocation=round(allocation, 2),
                finscore=0.0,  # Phase 2
            )
        )

    return response


@router.post("/holdings")
async def add_holding(
    request: AddHoldingRequest,
    user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Add a new holding to the portfolio."""
    _, portfolio, _ = await ensure_user_setup(user, db)

    symbol = request.symbol.upper()

    # Check if holding already exists for this symbol
    result = await db.execute(
        select(HoldingModel).where(
            HoldingModel.portfolio_id == portfolio.id,
            HoldingModel.symbol == symbol,
        )
    )
    existing = result.scalar_one_or_none()

    if existing:
        # Update: compute new weighted average price and add quantities
        total_qty = existing.quantity + request.quantity
        new_avg = (
            (existing.quantity * existing.avg_price + request.quantity * request.avg_price)
            / total_qty
        )
        existing.quantity = total_qty
        existing.avg_price = round(new_avg, 2)
    else:
        new_holding = HoldingModel(
            portfolio_id=portfolio.id,
            symbol=symbol,
            quantity=request.quantity,
            avg_price=request.avg_price,
        )
        db.add(new_holding)

    await db.flush()

    return {
        "status": "added",
        "holding": {
            "symbol": symbol,
            "quantity": request.quantity,
            "avg_price": request.avg_price,
        },
    }


@router.delete("/holdings/{symbol}")
async def remove_holding(
    symbol: str,
    user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Remove a holding from the portfolio."""
    _, portfolio, _ = await ensure_user_setup(user, db)

    symbol_upper = symbol.upper()

    result = await db.execute(
        select(HoldingModel).where(
            HoldingModel.portfolio_id == portfolio.id,
            HoldingModel.symbol == symbol_upper,
        )
    )
    holding = result.scalar_one_or_none()

    if not holding:
        raise HTTPException(status_code=404, detail=f"Holding '{symbol_upper}' not found")

    await db.delete(holding)
    await db.flush()

    return {"status": "removed", "symbol": symbol_upper}


# ---------------------------------------------------------------------------
# Mock endpoints (kept for Phase 2)
# ---------------------------------------------------------------------------

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
