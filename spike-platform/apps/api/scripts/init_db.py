"""Initialize database tables."""
import asyncio
import sys
import os

# Add the api app to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.session import engine, Base
# Import all models so they register with Base.metadata
from app.models.user import User
from app.models.portfolio import Portfolio, Holding, Transaction
from app.models.watchlist import Watchlist, WatchlistStock

async def init():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    print("Tables created successfully")

if __name__ == "__main__":
    asyncio.run(init())
