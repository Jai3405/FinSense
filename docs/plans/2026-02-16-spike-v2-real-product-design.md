# SPIKE v2 — From Mock to Real Product

**Date:** 2026-02-16
**Status:** Approved
**Author:** Jay + Claude (Architecture session)

---

## Problem Statement

SPIKE has a polished Next.js frontend (12 dashboard pages, collapsible sidebar, light mint theme) and a real Python ML trading system (PPO agent, paper trading, yfinance data). But these two systems are completely disconnected. Every number on every page is hardcoded. The FastAPI backend returns static mock data. No database exists. The frontend never calls the backend.

## Decisions

- **Data source:** Angel One SmartAPI (free tier) for real-time + historical Indian market data, yfinance as fallback
- **LLM approach:** Abstraction layer supporting Claude, Gemini, GPT-4o — swap providers without touching feature code
- **Database:** PostgreSQL + TimescaleDB extension for time-series data
- **Cache:** Redis for live prices + pub/sub broadcasting
- **Async jobs:** Celery for FinScore recalculation, bulk data ingestion, ML inference

---

## Architecture

### Layer 1: Data Foundation

```
Angel One SmartAPI ──┐
yfinance (fallback) ─┤──→ Data Service (Python) ──→ PostgreSQL + TimescaleDB
NSE Bhavcopy (EOD) ──┘         │                         (OHLCV, fundamentals, scores)
                               ▼
                        Redis (Cache + PubSub)
                         (live prices, sessions)
```

**Data flows:**
- Real-time prices: Angel One WebSocket → Redis PubSub → Frontend WebSocket
- Historical candles: Angel One REST → TimescaleDB (backfill on startup)
- Fundamentals (PE, ROE, margins): yfinance quarterly → PostgreSQL
- EOD bulk: NSE Bhavcopy daily → TimescaleDB

### Layer 2: Intelligence Engine

**FinScore Engine (Python, computed server-side, cached):**
- Fundamental sub-score (0-100): PE vs sector, ROE, debt ratio, margin trends, revenue growth
- Technical sub-score (0-100): RSI, MACD, Bollinger position, volume trends, moving averages
- Sentiment/Momentum sub-score (0-100): Delivery %, FII/DII flows, price momentum, relative strength
- Weighted composite → 0-100 overall score + signal (Strong Buy to Strong Sell)
- AI-generated explanation via LLM abstraction layer
- Recalculated daily via Celery job, cached in PostgreSQL

**LLM Abstraction Layer:**
```python
class LLMProvider(ABC):
    async def chat(self, messages, system_prompt, context) -> str
    async def stream(self, messages, system_prompt, context) -> AsyncIterator[str]

class ClaudeAdapter(LLMProvider): ...
class GeminiAdapter(LLMProvider): ...
class OpenAIAdapter(LLMProvider): ...
```

Consumed by:
- Strategy-GPT: Chat with market context injection (real prices, FinScores, portfolio)
- Legend Agents: Same LLM, different system prompts (Buffett philosophy, Lynch philosophy, etc.) + real data
- FinScore explanations: Why did this stock score 87?
- Screener NLP: "show me profitable midcaps under PE 20" → SQL query generation

**PPO Trading Agent (existing, to be connected):**
- Autopilot signals via existing PPOInference
- Paper trading execution via existing PaperTradingExecutor
- Backtest engine for Strategy-GPT recommendations

### Layer 3: Connection (Frontend ↔ Backend)

**Frontend API client:**
- `src/lib/api/client.ts` — base fetch wrapper with auth headers
- `src/lib/api/hooks/` — React Query hooks per domain (useStocks, useFinScore, usePortfolio, etc.)
- `src/lib/api/websocket.ts` — WebSocket manager for live price updates

**Communication patterns:**
- REST for all queries and mutations (cacheable, simple)
- WebSocket for live prices only (pushed from Redis PubSub via FastAPI)
- React Query for client-side caching + stale-while-revalidate

---

## Build Phases

### Phase 1: Plumbing (Foundation)
- Docker Compose: PostgreSQL + TimescaleDB + Redis
- Angel One SmartAPI integration (auth flow, WebSocket connection, REST endpoints)
- Data ingestion service: stream prices → Redis, store candles → TimescaleDB
- Replace all mock FastAPI endpoints with real data queries
- Build frontend API client + React Query hooks
- Wire every dashboard page to real backend data

**Exit criteria:** Dashboard shows real NIFTY prices, real stock quotes, real market status

### Phase 2: Intelligence
- FinScore engine v1 (fundamental + technical + momentum scoring)
- LLM abstraction layer with at least one adapter (Gemini free tier to start)
- Strategy-GPT: real AI chat with market context injection
- Legend Agents: persona system prompts + real data
- Screener: real SQL filters against stock fundamentals in PostgreSQL
- Connect PPO agent → Autopilot page (paper trading signals)

**Exit criteria:** FinScore returns real scores, Strategy-GPT gives real analysis, Screener filters real data

### Phase 3: Depth
- Charts page: TradingView-style with lightweight-charts library + real candles
- Portfolio: manual holdings entry + real P&L calculation
- Themes: dynamic baskets computed from screener criteria
- Discover: AI-curated signals from FinScore deltas + momentum
- Settings: user preferences persisted to database
- Autopilot: live paper trading with real PPO signals + equity curve

**Exit criteria:** Every page fully functional with real data, every button does something

### Phase 4: Polish
- Performance optimization (eliminate waterfalls, dynamic imports, bundle splitting)
- Error handling, loading skeletons, empty states for every page
- Mobile responsive layout
- Deploy: Vercel (frontend) + Railway/Render (FastAPI + PostgreSQL + Redis)
- Monitoring + error tracking

**Exit criteria:** Production-ready, deployed, accessible

---

## Tech Stack Summary

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 15, React Query, lightweight-charts, Tailwind |
| Backend | FastAPI, SQLAlchemy, Celery |
| Database | PostgreSQL + TimescaleDB |
| Cache | Redis |
| Market Data | Angel One SmartAPI, yfinance (fallback) |
| ML | PyTorch PPO agent (existing) |
| AI/LLM | Abstraction layer (Claude/Gemini/GPT adapters) |
| Auth | Clerk |
| Deploy | Vercel + Railway |

---

## What We're NOT Building (YAGNI)

- No real money trading (paper only for now)
- No broker order execution integration (analytics only)
- No mobile app (responsive web is enough)
- No social features (no followers, no sharing)
- No admin panel (not needed until there are users)
