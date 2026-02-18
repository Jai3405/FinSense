"""
API v1 Router - Aggregates all endpoint routers.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth,
    finscore,
    market,
    portfolio,
    stocks,
    themes,
    watchlist,
    ai,
    websocket,
)

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(stocks.router, prefix="/stocks", tags=["Stocks"])
api_router.include_router(finscore.router, prefix="/finscore", tags=["FinScore"])
api_router.include_router(market.router, prefix="/market", tags=["Market Data"])
api_router.include_router(portfolio.router, prefix="/portfolio", tags=["Portfolio"])
api_router.include_router(watchlist.router, prefix="/watchlist", tags=["Watchlist"])
api_router.include_router(themes.router, prefix="/themes", tags=["Smart Themes"])
api_router.include_router(ai.router, prefix="/ai", tags=["AI Services"])
api_router.include_router(websocket.router, tags=["WebSocket"])
