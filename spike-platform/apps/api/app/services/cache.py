"""Redis cache service for market data."""

import json
import logging
from typing import Any

import redis.asyncio as aioredis

from app.core.config import settings

logger = logging.getLogger(__name__)


class CacheService:
    """Redis cache with TTL-based expiry for market data."""

    def __init__(self, redis_client: aioredis.Redis):
        self._redis = redis_client

    async def get(self, key: str) -> Any | None:
        """Get a cached value."""
        try:
            data = await self._redis.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.warning(f"Cache get error for {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set a cached value with optional TTL in seconds."""
        try:
            ttl = ttl or settings.REDIS_CACHE_TTL
            await self._redis.set(key, json.dumps(value, default=str), ex=ttl)
        except Exception as e:
            logger.warning(f"Cache set error for {key}: {e}")

    async def delete(self, key: str) -> None:
        """Delete a cached value."""
        try:
            await self._redis.delete(key)
        except Exception as e:
            logger.warning(f"Cache delete error for {key}: {e}")

    async def get_stock_quote(self, symbol: str) -> dict | None:
        """Get cached stock quote (5s TTL)."""
        return await self.get(f"quote:{symbol}")

    async def set_stock_quote(self, symbol: str, data: dict) -> None:
        """Cache stock quote with 5s TTL."""
        await self.set(f"quote:{symbol}", data, ttl=5)

    async def get_market_indices(self) -> list | None:
        """Get cached market indices (10s TTL)."""
        return await self.get("market:indices")

    async def set_market_indices(self, data: list) -> None:
        """Cache market indices with 10s TTL."""
        await self.set("market:indices", data, ttl=10)

    async def get_sector_performance(self) -> list | None:
        """Get cached sector performance (30s TTL)."""
        return await self.get("market:sectors")

    async def set_sector_performance(self, data: list) -> None:
        """Cache sector performance with 30s TTL."""
        await self.set("market:sectors", data, ttl=30)

    async def get_trending(self) -> list | None:
        """Get cached trending stocks (30s TTL)."""
        return await self.get("market:trending")

    async def set_trending(self, data: list) -> None:
        """Cache trending stocks with 30s TTL."""
        await self.set("market:trending", data, ttl=30)

    async def get_historical(self, symbol: str, period: str, interval: str) -> list | None:
        """Get cached historical data (5m TTL)."""
        return await self.get(f"history:{symbol}:{period}:{interval}")

    async def set_historical(
        self, symbol: str, period: str, interval: str, data: list
    ) -> None:
        """Cache historical data with 5m TTL."""
        await self.set(f"history:{symbol}:{period}:{interval}", data, ttl=300)


# Global cache instance
_cache_service: CacheService | None = None


def get_cache_service() -> CacheService | None:
    """Get cache service (may be None if Redis not connected)."""
    return _cache_service


def set_cache_service(cache: CacheService) -> None:
    """Set the global cache service."""
    global _cache_service
    _cache_service = cache
