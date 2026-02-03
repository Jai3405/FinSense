"""
Watchlist endpoints - User watchlist management.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.v1.deps.auth import get_current_user, ClerkUser

router = APIRouter()


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


@router.get("/", response_model=list[WatchlistStock])
async def get_watchlist(
    user: ClerkUser = Depends(get_current_user),
) -> list[WatchlistStock]:
    """Get user's watchlist."""
    return [
        WatchlistStock(
            symbol="RELIANCE",
            name="Reliance Industries",
            price=2456.75,
            change=28.50,
            change_percent=1.17,
            finscore=8.4,
            added_at="2024-01-15T10:30:00+05:30",
        ),
        WatchlistStock(
            symbol="TCS",
            name="Tata Consultancy Services",
            price=3845.20,
            change=-12.30,
            change_percent=-0.32,
            finscore=7.9,
            added_at="2024-01-20T14:45:00+05:30",
        ),
    ]


@router.post("/")
async def add_to_watchlist(
    request: AddToWatchlistRequest,
    user: ClerkUser = Depends(get_current_user),
) -> dict:
    """Add stock to watchlist."""
    return {
        "status": "added",
        "symbol": request.symbol.upper(),
    }


@router.delete("/{symbol}")
async def remove_from_watchlist(
    symbol: str,
    user: ClerkUser = Depends(get_current_user),
) -> dict:
    """Remove stock from watchlist."""
    return {"status": "removed", "symbol": symbol.upper()}


@router.post("/{symbol}/alert")
async def set_price_alert(
    symbol: str,
    target_price: float,
    direction: str,  # "above" or "below"
    user: ClerkUser = Depends(get_current_user),
) -> dict:
    """Set price alert for a watchlist stock."""
    return {
        "status": "alert_set",
        "symbol": symbol.upper(),
        "target_price": target_price,
        "direction": direction,
    }
