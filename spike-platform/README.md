# SPIKE - AI Wealth Intelligence Platform

India's first AI-powered wealth intelligence platform with institutional-grade insights, personalized strategies, and autonomous portfolio management.

## 🚀 Quick Start

```bash
# Install dependencies
pnpm install

# Start development servers
./start-dev.sh

# Or manually
pnpm dev
```

**Services:**
- Web App: http://localhost:3000
- API Server: http://localhost:8000
- API Docs: http://localhost:8000/api/v1/docs

## 📁 Project Structure

```
spike-platform/
├── apps/
│   ├── web/          # Next.js 15 frontend
│   └── api/          # FastAPI backend
├── packages/
│   ├── ui/           # Shared UI components
│   ├── config/       # Shared configurations
│   ├── db/           # Database schemas & migrations
│   └── finsense/     # FinSense AI engine
└── infrastructure/   # Docker, K8s configs
```

## 🛠 Tech Stack

### Frontend
- Next.js 15 with App Router
- TypeScript 5.4
- Tailwind CSS 4.0
- shadcn/ui components
- TanStack Query
- Zustand for state

### Backend
- FastAPI (Python 3.12)
- SQLAlchemy 2.0 (async)
- PostgreSQL 16 + TimescaleDB
- Redis 7 for caching
- Clerk for authentication

### AI/ML
- Claude 3.5 / GPT-4 Turbo
- PyTorch 2.2
- Custom FinScore algorithm

## 🔐 Environment Variables

Copy `.env.example` to `.env.local` and fill in:

```env
# Database
DATABASE_URL=postgresql://...

# Clerk Auth
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_...
CLERK_SECRET_KEY=sk_...

# AI Services
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...
```

## 📊 Core Features

### FinScore
Universal 0-10 stock rating combining 9 dimensions:
- Quality & Fundamentals
- Price Momentum
- Value & Valuation
- Sentiment Analysis
- Risk Assessment
- Institutional Flows
- Regime Fit
- Sector Dynamics
- Technical Signals

### Legend Agents
AI trained on legendary investor philosophies:
- Warren Buffett (Value)
- Peter Lynch (GARP)
- Ray Dalio (All-Weather)

### Strategy-GPT
Natural language to quantitative strategy conversion.

### Portfolio Autopilot
Autonomous portfolio management with regime-adaptive allocation.

## 🏛 SEBI Compliance

- SEBI RA/IA Regulations compliant
- Data localized in India (AWS Mumbai)
- KYC requirements built-in
- Mandatory risk disclosures
- PFUTP compliance

## 📜 License

Proprietary - All rights reserved.
