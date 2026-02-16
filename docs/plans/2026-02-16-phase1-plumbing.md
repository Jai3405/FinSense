# Phase 1: Plumbing — Real Data Foundation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace all mock data with real Indian market data flowing from Angel One SmartAPI through FastAPI into the Next.js frontend.

**Architecture:** Docker Compose runs PostgreSQL+TimescaleDB and Redis locally. Angel One SmartAPI provides real-time and historical NSE data. FastAPI endpoints query real data instead of returning hardcoded values. Frontend uses React Query hooks to fetch from the API.

**Tech Stack:** FastAPI, SQLAlchemy (async), PostgreSQL + TimescaleDB, Redis, Angel One SmartAPI, yfinance (fallback), React Query, Next.js 15

---

### Task 1: Docker Compose for PostgreSQL + Redis

**Files:**
- Create: `spike-platform/docker-compose.yml`
- Create: `spike-platform/apps/api/.env`

**Step 1: Create docker-compose.yml**

```yaml
version: "3.9"
services:
  postgres:
    image: timescale/timescaledb:latest-pg16
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: spike
      POSTGRES_PASSWORD: spike_dev_2026
      POSTGRES_DB: spike
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redisdata:/data

volumes:
  pgdata:
  redisdata:
```

**Step 2: Create apps/api/.env**

```env
DATABASE_URL=postgresql+asyncpg://spike:spike_dev_2026@localhost:5432/spike
REDIS_URL=redis://localhost:6379/0
ENVIRONMENT=development
DEBUG=true
API_SECRET_KEY=dev-secret-key-change-in-prod
```

**Step 3: Start services**

Run: `cd spike-platform && docker compose up -d`
Expected: Both postgres and redis containers running

**Step 4: Verify connections**

Run: `docker compose ps`
Expected: Both services healthy on ports 5432 and 6379

**Step 5: Commit**

```bash
git add docker-compose.yml apps/api/.env
git commit -m "infra: add Docker Compose for PostgreSQL + Redis"
```

---

### Task 2: Database Migrations — Create Tables

**Files:**
- Modify: `spike-platform/apps/api/app/main.py` (add table creation on startup)
- Existing models: `apps/api/app/models/user.py`, `portfolio.py`, `watchlist.py`

**Step 1: Create migration script to initialize DB**

Create: `spike-platform/apps/api/scripts/init_db.py`

```python
"""Initialize database tables."""
import asyncio
from app.db.session import engine
from app.models.user import Base as UserBase
from app.models.portfolio import Base as PortfolioBase
from app.models.watchlist import Base as WatchlistBase

async def init():
    async with engine.begin() as conn:
        await conn.run_sync(UserBase.metadata.create_all)
        await conn.run_sync(PortfolioBase.metadata.create_all)
        await conn.run_sync(WatchlistBase.metadata.create_all)
    print("Tables created successfully")

if __name__ == "__main__":
    asyncio.run(init())
```

**Step 2: Run migration**

Run: `cd spike-platform/apps/api && python -m scripts.init_db`
Expected: "Tables created successfully"

**Step 3: Verify tables exist**

Run: `docker exec -it spike-platform-postgres-1 psql -U spike -c "\dt"`
Expected: users, portfolios, holdings, transactions, watchlists, watchlist_stocks tables

**Step 4: Commit**

```bash
git add scripts/init_db.py
git commit -m "infra: add database initialization script"
```

---

### Task 3: Angel One SmartAPI Integration — Data Service

**Files:**
- Create: `spike-platform/apps/api/app/services/market_data.py`
- Create: `spike-platform/apps/api/app/services/angel_one.py`
- Modify: `spike-platform/apps/api/pyproject.toml` (add smartapi-python dependency)

**Step 1: Add Angel One SDK dependency**

Add to pyproject.toml dependencies:
```
smartapi-python>=1.3.0
```

Run: `cd spike-platform/apps/api && pip install smartapi-python`

**Step 2: Create Angel One service**

Create: `spike-platform/apps/api/app/services/angel_one.py`

This service handles:
- Login with API key + TOTP
- Fetch stock quotes (LTP, OHLC)
- Fetch historical candle data
- WebSocket connection for live prices
- Token refresh

Key methods:
```python
class AngelOneService:
    async def get_quote(self, symbol: str, exchange: str = "NSE") -> dict
    async def get_candle_data(self, symbol: str, interval: str, from_date: str, to_date: str) -> list
    async def search_scrip(self, query: str) -> list
    def get_token_map(self) -> dict  # symbol → token mapping
```

**Step 3: Create market data service with yfinance fallback**

Create: `spike-platform/apps/api/app/services/market_data.py`

```python
class MarketDataService:
    """Unified market data service. Angel One primary, yfinance fallback."""

    async def get_stock_quote(self, symbol: str) -> StockQuote
    async def get_historical(self, symbol: str, period: str, interval: str) -> list[OHLCV]
    async def get_market_indices(self) -> list[IndexData]
    async def get_sector_performance(self) -> list[SectorData]
    async def search_stocks(self, query: str) -> list[SearchResult]
```

**Step 4: Add environment variables for Angel One**

Add to `apps/api/.env`:
```env
ANGEL_ONE_API_KEY=your_api_key
ANGEL_ONE_CLIENT_ID=your_client_id
ANGEL_ONE_PASSWORD=your_password
ANGEL_ONE_TOTP_SECRET=your_totp_secret
```

**Step 5: Commit**

```bash
git add apps/api/app/services/ apps/api/pyproject.toml apps/api/.env
git commit -m "feat: add Angel One SmartAPI + market data service"
```

---

### Task 4: Replace Mock Endpoints — Stocks & Market

**Files:**
- Modify: `spike-platform/apps/api/app/api/v1/endpoints/stocks.py`
- Modify: `spike-platform/apps/api/app/api/v1/endpoints/market.py`

**Step 1: Wire stocks endpoints to MarketDataService**

Replace hardcoded mock data in stocks.py:
- `GET /stocks/search` → `market_data.search_stocks(query)`
- `GET /stocks/{symbol}/quote` → `market_data.get_stock_quote(symbol)`
- `GET /stocks/{symbol}/history` → `market_data.get_historical(symbol, period, interval)`
- `GET /stocks/trending` → `market_data.get_trending()` (top volume + top gainers)

**Step 2: Wire market endpoints to MarketDataService**

Replace hardcoded mock data in market.py:
- `GET /market/indices` → `market_data.get_market_indices()`
- `GET /market/sectors` → `market_data.get_sector_performance()`
- `GET /market/status` → Check Indian market hours (9:15-15:30 IST, Mon-Fri)
- `GET /market/breadth` → `market_data.get_market_breadth()`

**Step 3: Test endpoints**

Run: `cd spike-platform/apps/api && uvicorn app.main:app --reload`
Test: `curl http://localhost:8000/api/v1/stocks/RELIANCE.NS/quote`
Expected: Real RELIANCE stock price from Angel One / yfinance

**Step 4: Commit**

```bash
git add apps/api/app/api/v1/endpoints/stocks.py apps/api/app/api/v1/endpoints/market.py
git commit -m "feat: wire stock and market endpoints to real data"
```

---

### Task 5: Replace Mock Endpoints — Portfolio & Watchlist

**Files:**
- Modify: `spike-platform/apps/api/app/api/v1/endpoints/portfolio.py`
- Modify: `spike-platform/apps/api/app/api/v1/endpoints/watchlist.py`

**Step 1: Wire portfolio endpoints to database**

Replace mocks with actual SQLAlchemy queries:
- `GET /portfolio/summary` → Query holdings, calculate total value with live prices
- `GET /portfolio/holdings` → Query holdings table joined with live quotes
- `POST /portfolio/holdings` → INSERT into holdings table
- `DELETE /portfolio/holdings/{symbol}` → DELETE from holdings table
- `GET /portfolio/allocation` → GROUP BY sector from holdings + stock info

**Step 2: Wire watchlist endpoints to database**

- `GET /watchlist/` → Query watchlist_stocks for user
- `POST /watchlist/` → INSERT into watchlist_stocks
- `DELETE /watchlist/{symbol}` → DELETE from watchlist_stocks

**Step 3: Test with curl**

```bash
# Add a holding
curl -X POST http://localhost:8000/api/v1/portfolio/holdings \
  -H "Content-Type: application/json" \
  -d '{"symbol": "RELIANCE.NS", "quantity": 10, "avg_price": 2450}'

# Get portfolio summary
curl http://localhost:8000/api/v1/portfolio/summary
```

**Step 4: Commit**

```bash
git add apps/api/app/api/v1/endpoints/portfolio.py apps/api/app/api/v1/endpoints/watchlist.py
git commit -m "feat: wire portfolio and watchlist to database + live prices"
```

---

### Task 6: Redis Cache Layer

**Files:**
- Create: `spike-platform/apps/api/app/services/cache.py`
- Modify: `spike-platform/apps/api/app/main.py` (initialize Redis on startup)

**Step 1: Create Redis cache service**

```python
class CacheService:
    """Redis cache with TTL for market data."""

    async def get(self, key: str) -> dict | None
    async def set(self, key: str, value: dict, ttl: int = 300)
    async def get_stock_quote(self, symbol: str) -> dict | None  # 5s TTL
    async def get_market_indices(self) -> list | None  # 10s TTL
    async def get_finscore(self, symbol: str) -> dict | None  # 1h TTL
```

**Step 2: Add cache layer to market data endpoints**

Pattern: Check cache → if miss, fetch from Angel One → store in cache → return

**Step 3: Initialize Redis in FastAPI lifespan**

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.redis = await aioredis.from_url(settings.REDIS_URL)
    app.state.cache = CacheService(app.state.redis)
    yield
    # Shutdown
    await app.state.redis.close()
```

**Step 4: Commit**

```bash
git add apps/api/app/services/cache.py apps/api/app/main.py
git commit -m "feat: add Redis cache layer for market data"
```

---

### Task 7: Frontend API Client

**Files:**
- Create: `spike-platform/apps/web/src/lib/api/client.ts`
- Create: `spike-platform/apps/web/src/lib/api/types.ts`
- Create: `spike-platform/apps/web/src/lib/api/hooks/use-stocks.ts`
- Create: `spike-platform/apps/web/src/lib/api/hooks/use-market.ts`
- Create: `spike-platform/apps/web/src/lib/api/hooks/use-portfolio.ts`
- Create: `spike-platform/apps/web/src/lib/api/hooks/use-finscore.ts`

**Step 1: Create typed API client**

```typescript
// client.ts
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function apiClient<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const token = await getToken(); // from Clerk
  const res = await fetch(`${API_URL}/api/v1${endpoint}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options?.headers,
    },
  });
  if (!res.ok) throw new ApiError(res.status, await res.json());
  return res.json();
}
```

**Step 2: Create type definitions**

```typescript
// types.ts
export interface StockQuote {
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
  high: number;
  low: number;
  open: number;
  prev_close: number;
}

export interface MarketIndex { ... }
export interface PortfolioSummary { ... }
export interface Holding { ... }
// etc.
```

**Step 3: Create React Query hooks**

```typescript
// hooks/use-stocks.ts
export function useStockQuote(symbol: string) {
  return useQuery({
    queryKey: ["stock", "quote", symbol],
    queryFn: () => apiClient<StockQuote>(`/stocks/${symbol}/quote`),
    refetchInterval: 5000, // refresh every 5s during market hours
  });
}

export function useStockSearch(query: string) {
  return useQuery({
    queryKey: ["stock", "search", query],
    queryFn: () => apiClient<SearchResult[]>(`/stocks/search?q=${query}`),
    enabled: query.length >= 2,
  });
}
```

```typescript
// hooks/use-market.ts
export function useMarketIndices() {
  return useQuery({
    queryKey: ["market", "indices"],
    queryFn: () => apiClient<MarketIndex[]>("/market/indices"),
    refetchInterval: 10000,
  });
}

export function useMarketStatus() { ... }
export function useSectorPerformance() { ... }
```

```typescript
// hooks/use-portfolio.ts
export function usePortfolioSummary() {
  return useQuery({
    queryKey: ["portfolio", "summary"],
    queryFn: () => apiClient<PortfolioSummary>("/portfolio/summary"),
  });
}

export function useHoldings() { ... }
export function useAddHolding() { return useMutation(...) }
export function useRemoveHolding() { return useMutation(...) }
```

**Step 4: Commit**

```bash
git add apps/web/src/lib/api/
git commit -m "feat: add typed API client and React Query hooks"
```

---

### Task 8: Wire Dashboard Pages to Real API

**Files:**
- Modify: `spike-platform/apps/web/src/components/dashboard/market-overview.tsx`
- Modify: `spike-platform/apps/web/src/components/dashboard/portfolio-summary.tsx`
- Modify: `spike-platform/apps/web/src/components/dashboard/watchlist-widget.tsx`
- Modify: `spike-platform/apps/web/src/components/dashboard/top-movers.tsx`
- Modify: `spike-platform/apps/web/src/components/layout/sidebar.tsx` (portfolio widget)

**Step 1: Replace hardcoded market data in market-overview.tsx**

```typescript
// Before: const indices = [{ name: "NIFTY 50", value: "22,150", ... }]
// After:
const { data: indices, isLoading } = useMarketIndices();
const { data: sectors } = useSectorPerformance();
const { data: status } = useMarketStatus();
```

Add loading skeletons for when data is fetching.

**Step 2: Replace hardcoded portfolio data in portfolio-summary.tsx**

```typescript
// Before: const data = { totalValue: 1245678, ... }
// After:
const { data: summary, isLoading } = usePortfolioSummary();
```

**Step 3: Replace hardcoded watchlist in watchlist-widget.tsx**

```typescript
const { data: watchlist } = useWatchlist();
```

**Step 4: Replace hardcoded top movers in top-movers.tsx**

```typescript
const { data: trending } = useTrendingStocks();
```

**Step 5: Update sidebar portfolio widget with real data**

```typescript
const { data: portfolio } = usePortfolioSummary();
// Use portfolio.total_value and portfolio.returns_percent
```

**Step 6: Test the full dashboard**

Run: `cd spike-platform && ./start-dev.sh`
Expected: Dashboard showing real NIFTY prices, real stock data, empty portfolio (user hasn't added holdings yet)

**Step 7: Commit**

```bash
git add apps/web/src/components/dashboard/ apps/web/src/components/layout/sidebar.tsx
git commit -m "feat: wire dashboard to real API data"
```

---

### Task 9: Wire Remaining Pages to Real API

**Files:**
- Modify: `apps/web/src/app/dashboard/screener/page.tsx`
- Modify: `apps/web/src/app/dashboard/charts/page.tsx`
- Modify: `apps/web/src/app/dashboard/portfolio/page.tsx`
- Modify: `apps/web/src/app/dashboard/finscore/page.tsx`
- Modify: `apps/web/src/app/dashboard/themes/page.tsx`
- Modify: `apps/web/src/app/dashboard/settings/page.tsx`

**Step 1: Portfolio page — real holdings + P&L**

Use `useHoldings()` + `usePortfolioSummary()`. Add "Add Holding" form that calls `useAddHolding()` mutation.

**Step 2: Screener page — real stock search + filters**

Use `useStockSearch()` for search. Filters query against real stock data from API.

**Step 3: Charts page — real candle data**

Use `useStockHistory(symbol, period, interval)` → feed into lightweight-charts library.

**Step 4: FinScore page — fetch from API (still mock scores until Phase 2)**

Use `useFinScore(symbol)`. This will still return mock data from the backend until Phase 2 builds the real engine, but the wiring is in place.

**Step 5: Themes page — fetch from API**

Use `useThemes()`. Same pattern — data from API, intelligence comes in Phase 2.

**Step 6: Settings page — real user profile from Clerk**

Use Clerk's `useUser()` hook for profile data. Persist preferences to database.

**Step 7: Commit**

```bash
git add apps/web/src/app/dashboard/
git commit -m "feat: wire all dashboard pages to real API"
```

---

### Task 10: WebSocket for Live Price Updates

**Files:**
- Create: `spike-platform/apps/api/app/api/v1/endpoints/websocket.py`
- Create: `spike-platform/apps/web/src/lib/api/websocket.ts`
- Modify: `spike-platform/apps/web/src/components/dashboard/market-overview.tsx`

**Step 1: Add WebSocket endpoint to FastAPI**

```python
@router.websocket("/ws/prices")
async def price_stream(websocket: WebSocket):
    await websocket.accept()
    # Subscribe to Redis PubSub for price updates
    # Forward to client as they arrive
```

**Step 2: Create WebSocket manager in frontend**

```typescript
class PriceWebSocket {
  subscribe(symbols: string[], onUpdate: (quote: StockQuote) => void): void
  unsubscribe(symbols: string[]): void
  disconnect(): void
}
```

**Step 3: Integrate live prices into market overview**

Market overview component subscribes to NIFTY, SENSEX, and watchlist symbols. Prices update in real-time without polling.

**Step 4: Commit**

```bash
git add apps/api/app/api/v1/endpoints/websocket.py apps/web/src/lib/api/websocket.ts
git commit -m "feat: add WebSocket for live price streaming"
```

---

## Phase 1 Exit Criteria

When complete, the following must be true:

- [ ] Docker Compose starts PostgreSQL + Redis
- [ ] FastAPI starts and connects to both
- [ ] `GET /api/v1/market/indices` returns real NIFTY/SENSEX prices
- [ ] `GET /api/v1/stocks/RELIANCE.NS/quote` returns real price
- [ ] `POST /api/v1/portfolio/holdings` persists to database
- [ ] Dashboard home page shows real market data
- [ ] Sidebar portfolio widget shows real portfolio value
- [ ] Charts page renders real OHLCV candles
- [ ] WebSocket pushes live price updates
- [ ] Zero hardcoded mock data remains in any visible page

---

## Prerequisites

Before starting:
1. Install Docker Desktop
2. Create Angel One account + generate API credentials
3. Have the dev server running (`./start-dev.sh`)
