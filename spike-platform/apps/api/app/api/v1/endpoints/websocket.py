"""WebSocket endpoint for live price streaming."""
import asyncio
import json
import logging
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.market_data import get_market_data_service

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections and subscriptions."""

    def __init__(self):
        self.active_connections: dict[WebSocket, set[str]] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[websocket] = set()

    def disconnect(self, websocket: WebSocket):
        self.active_connections.pop(websocket, None)

    def subscribe(self, websocket: WebSocket, symbols: list[str]):
        if websocket in self.active_connections:
            self.active_connections[websocket].update(symbols)

    def unsubscribe(self, websocket: WebSocket, symbols: list[str]):
        if websocket in self.active_connections:
            self.active_connections[websocket].difference_update(symbols)

    def get_all_symbols(self) -> set[str]:
        all_symbols = set()
        for symbols in self.active_connections.values():
            all_symbols.update(symbols)
        return all_symbols

    async def broadcast_quote(self, symbol: str, data: dict):
        disconnected = []
        for ws, symbols in self.active_connections.items():
            if symbol in symbols:
                try:
                    await ws.send_json(
                        {"type": "quote", "symbol": symbol, "data": data}
                    )
                except Exception:
                    disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws)


manager = ConnectionManager()


@router.websocket("/ws/prices")
async def price_stream(websocket: WebSocket):
    """
    WebSocket endpoint for live price updates.

    Client sends: {"action": "subscribe", "symbols": ["RELIANCE.NS", "TCS.NS"]}
    Client sends: {"action": "unsubscribe", "symbols": ["RELIANCE.NS"]}
    Server sends: {"type": "quote", "symbol": "RELIANCE.NS", "data": {...}}
    """
    await manager.connect(websocket)
    market_data = get_market_data_service()

    # Background task to push price updates
    async def push_prices():
        while True:
            try:
                symbols = manager.active_connections.get(websocket, set())
                for symbol in list(symbols):
                    try:
                        quote = await market_data.get_stock_quote(symbol)
                        await websocket.send_json(
                            {
                                "type": "quote",
                                "symbol": symbol,
                                "data": {
                                    "price": quote.price,
                                    "change": quote.change,
                                    "change_percent": quote.change_percent,
                                    "volume": quote.volume,
                                    "high": quote.high,
                                    "low": quote.low,
                                    "timestamp": datetime.now().isoformat(),
                                },
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Error fetching quote for {symbol}: {e}")
                await asyncio.sleep(5)  # Update every 5 seconds
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Price push error: {e}")
                await asyncio.sleep(5)

    # Start background price pusher
    price_task = asyncio.create_task(push_prices())

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            symbols = data.get("symbols", [])

            if action == "subscribe":
                manager.subscribe(websocket, symbols)
                await websocket.send_json(
                    {"type": "subscribed", "symbols": symbols}
                )
            elif action == "unsubscribe":
                manager.unsubscribe(websocket, symbols)
                await websocket.send_json(
                    {"type": "unsubscribed", "symbols": symbols}
                )
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        price_task.cancel()
        manager.disconnect(websocket)
