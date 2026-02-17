"""
Watchlist endpoints - User watchlist management.
"""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.deps.auth import get_current_user, ClerkUser
from app.db.session import get_db
from app.models.user import User
from app.models.portfolio import Portfolio as PortfolioModel
from app.models.watchlist import (
    Watchlist as WatchlistModel,
    WatchlistStock as WatchlistStockModel,
)
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

class WatchlistStock(BaseModel):
    """Stock in watchlist."""

    symbol: str
    name: str
    price: float
    change: float
    change_percent: float
    finscore: float
    added_at: str


class AddToWatchlistRequest(BaseModel):
    """Request to add stock to watchlist."""

    symbol: str
    alert_price: float | None = None
    notes: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/", response_model=list[WatchlistStock])
async def get_watchlist(
    user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[WatchlistStock]:
    """Get user's watchlist."""
    _, _, watchlist = await ensure_user_setup(user, db)

    result = await db.execute(
        select(WatchlistStockModel).where(
            WatchlistStockModel.watchlist_id == watchlist.id
        )
    )
    stocks = result.scalars().all()

    if not stocks:
        return []

    market_data = get_market_data_service()

    # Fetch live quotes concurrently
    async def _safe_quote(s: WatchlistStockModel):
        try:
            return s, await market_data.get_stock_quote(ensure_ns_suffix(s.symbol))
        except Exception as exc:
            logger.warning("Could not fetch quote for %s: %s", s.symbol, exc)
            return s, None

    results = await asyncio.gather(*[_safe_quote(s) for s in stocks])

    response: list[WatchlistStock] = []
    for stock, quote in results:
        if quote:
            price = quote.price
            change = quote.change
            change_percent = quote.change_percent
            name = quote.name
        else:
            price = 0.0
            change = 0.0
            change_percent = 0.0
            name = stock.symbol

        response.append(
            WatchlistStock(
                symbol=strip_ns_suffix(stock.symbol),
                name=strip_ns_suffix(name),
                price=round(price, 2),
                change=round(change, 2),
                change_percent=round(change_percent, 2),
                finscore=0.0,  # Phase 2
                added_at=stock.added_at.isoformat() if stock.added_at else "",
            )
        )

    return response


@router.post("/")
async def add_to_watchlist(
    request: AddToWatchlistRequest,
    user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Add stock to watchlist."""
    _, _, watchlist = await ensure_user_setup(user, db)

    symbol = request.symbol.upper()

    # Check if already in watchlist
    result = await db.execute(
        select(WatchlistStockModel).where(
            WatchlistStockModel.watchlist_id == watchlist.id,
            WatchlistStockModel.symbol == symbol,
        )
    )
    existing = result.scalar_one_or_none()

    if existing:
        raise HTTPException(
            status_code=409, detail=f"'{symbol}' is already in your watchlist"
        )

    new_stock = WatchlistStockModel(
        watchlist_id=watchlist.id,
        symbol=symbol,
        alert_price=request.alert_price,
        notes=request.notes,
    )
    db.add(new_stock)
    await db.flush()

    return {
        "status": "added",
        "symbol": symbol,
    }


@router.delete("/{symbol}")
async def remove_from_watchlist(
    symbol: str,
    user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Remove stock from watchlist."""
    _, _, watchlist = await ensure_user_setup(user, db)

    symbol_upper = symbol.upper()

    result = await db.execute(
        select(WatchlistStockModel).where(
            WatchlistStockModel.watchlist_id == watchlist.id,
            WatchlistStockModel.symbol == symbol_upper,
        )
    )
    stock = result.scalar_one_or_none()

    if not stock:
        raise HTTPException(
            status_code=404, detail=f"'{symbol_upper}' not found in your watchlist"
        )

    await db.delete(stock)
    await db.flush()

    return {"status": "removed", "symbol": symbol_upper}


@router.post("/{symbol}/alert")
async def set_price_alert(
    symbol: str,
    target_price: float,
    direction: str,  # "above" or "below"
    user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Set price alert for a watchlist stock."""
    _, _, watchlist = await ensure_user_setup(user, db)

    symbol_upper = symbol.upper()

    if direction not in ("above", "below"):
        raise HTTPException(
            status_code=400, detail="direction must be 'above' or 'below'"
        )

    result = await db.execute(
        select(WatchlistStockModel).where(
            WatchlistStockModel.watchlist_id == watchlist.id,
            WatchlistStockModel.symbol == symbol_upper,
        )
    )
    stock = result.scalar_one_or_none()

    if not stock:
        raise HTTPException(
            status_code=404, detail=f"'{symbol_upper}' not found in your watchlist"
        )

    stock.alert_price = target_price
    stock.alert_direction = direction
    await db.flush()

    return {
        "status": "alert_set",
        "symbol": symbol_upper,
        "target_price": target_price,
        "direction": direction,
    }
