# SPIKE Product Specification
## Production-Grade Wealth Intelligence Platform

**Version:** 1.0
**Date:** January 27, 2026
**Classification:** Internal - Confidential

---

## REGULATORY FRAMEWORK

### SEBI Compliance Requirements

**SPIKE operates as a Research Analyst (RA) / Investment Adviser (IA) platform.**

| Regulation | Requirement | SPIKE Implementation |
|------------|-------------|----------------------|
| **SEBI RA Regulations 2014** | Must be registered to provide research/recommendations | Apply for SEBI RA registration |
| **SEBI IA Regulations 2013** | Required if providing personalized advice | Structured as "information" not "advice" initially |
| **PFUTP Rules** | No front-running, no manipulation | Strict data isolation, audit logs |
| **Data Localization** | Financial data must be stored in India | AWS Mumbai / GCP Mumbai only |
| **KYC Requirements** | Must verify users for certain features | Integrate DigiLocker, Video KYC |
| **Risk Disclosure** | Must disclose risks prominently | Mandatory risk screens, disclaimers |
| **Grievance Redressal** | Must have complaint mechanism | In-app support, SEBI SCORES integration |

### Compliance Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPLIANCE LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Disclaimer  │  │    KYC      │  │   Audit     │             │
│  │   Engine    │  │   Module    │  │   Logger    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │    Risk     │  │  Grievance  │  │  Regulatory │             │
│  │  Profiling  │  │   System    │  │  Reporting  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
│  MANDATORY DISCLAIMERS:                                         │
│  • "Past performance is not indicative of future results"       │
│  • "Investments are subject to market risks"                    │
│  • "SPIKE provides information, not personalized advice"        │
│  • "Consult a SEBI-registered advisor for personal advice"      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### What SPIKE Can and Cannot Do (Legally)

| CAN DO | CANNOT DO |
|--------|-----------|
| Provide stock scores (FinScore) | Guarantee returns |
| Show historical analysis | Promise future performance |
| Offer educational content | Provide personalized investment advice (without IA license) |
| Display market data | Execute trades (without broker license) |
| Aggregate portfolio data | Hold customer funds |
| Alert on price/sentiment changes | Manipulate or front-run |
| Provide research reports | Claim SEBI registration without having it |

---

## TECHNOLOGY STACK (2026 Cutting Edge)

### Why These Choices

Every technology choice is deliberate - optimized for **scale, speed, security, and developer productivity**.

### Frontend Stack

```yaml
Framework: Next.js 15 (App Router)
  Why:
    - React Server Components (faster initial load)
    - Edge runtime (global low latency)
    - Built-in API routes
    - Streaming SSR
    - Turbopack (10x faster builds)

Language: TypeScript 5.4 (Strict Mode)
  Why:
    - Type safety prevents runtime errors
    - Better IDE support
    - Self-documenting code
    - Required for financial applications

Styling: Tailwind CSS 4.0 + shadcn/ui
  Why:
    - Design system consistency
    - Rapid development
    - Small bundle size
    - Accessible components out of the box

State Management: Zustand + TanStack Query v5
  Why:
    - Zustand: Simple, fast, minimal boilerplate
    - TanStack Query: Server state, caching, real-time sync
    - No Redux complexity

Charts: TradingView Lightweight Charts + Recharts
  Why:
    - TradingView: Professional financial charts (candlestick, indicators)
    - Recharts: Portfolio visualizations, pie charts, etc.

Real-time: Socket.io + Server-Sent Events
  Why:
    - WebSocket for bidirectional (trading)
    - SSE for unidirectional (alerts, prices)

Forms: React Hook Form + Zod
  Why:
    - Performance (minimal re-renders)
    - Type-safe validation
    - Required for financial forms
```

### Backend Stack

```yaml
Primary API: FastAPI (Python 3.12)
  Why:
    - Native async/await
    - Automatic OpenAPI docs
    - Pydantic v2 validation (50x faster)
    - ML ecosystem compatibility
    - Type hints

Secondary Services: Go 1.22
  Why:
    - High-performance microservices
    - Real-time price streaming
    - Concurrent processing

API Protocol: REST + GraphQL (Strawberry)
  Why:
    - REST: Simple CRUD operations
    - GraphQL: Complex portfolio queries, reduce over-fetching

Authentication: Clerk
  Why:
    - Production-ready auth
    - Multi-factor authentication
    - Session management
    - Compliant with security standards
    - Social logins + Email/Phone

Background Jobs: Celery + Redis
  Why:
    - Distributed task processing
    - Scheduled jobs (EOD calculations)
    - Retry mechanisms

Message Queue: Apache Kafka
  Why:
    - High throughput event streaming
    - Real-time price ingestion
    - Audit log streaming
    - Decoupled microservices
```

### Database Layer

```yaml
Primary Database: PostgreSQL 16 + pgvector
  Why:
    - ACID compliance (required for financial)
    - pgvector for AI embeddings
    - Proven at scale
    - Rich querying

Time-Series: TimescaleDB (PostgreSQL extension)
  Why:
    - Optimized for OHLCV data
    - Automatic partitioning
    - Compression (10x storage savings)
    - Fast range queries

Cache: Redis 7 (Cluster Mode)
  Why:
    - Sub-millisecond latency
    - Session storage
    - Rate limiting
    - Real-time leaderboards/rankings

Search: Meilisearch
  Why:
    - Typo-tolerant stock search
    - Faceted filtering
    - Faster and simpler than Elasticsearch
    - Great for stock screeners

Vector Store: Pinecone
  Why:
    - Similarity search for stocks
    - Strategy-GPT semantic matching
    - Production-ready, managed
```

### AI/ML Stack

```yaml
LLM: Claude 3.5 Sonnet (Primary) + GPT-4 Turbo (Fallback)
  Why:
    - Claude: Better reasoning, safer
    - GPT-4: Fallback, diverse perspectives
    - Both: Production SLAs

Embeddings: text-embedding-3-large (OpenAI)
  Why:
    - State-of-the-art semantic similarity
    - Used for stock similarity, news matching

ML Framework: PyTorch 2.2 + Lightning
  Why:
    - Industry standard
    - Production deployment ready
    - Lightning for clean training code

ML Serving: vLLM + Triton Inference Server
  Why:
    - vLLM: Fast LLM inference
    - Triton: GPU-optimized model serving
    - Batching, caching built-in

Feature Store: Feast
  Why:
    - Consistent features train/serve
    - Point-in-time correctness
    - Prevents data leakage
```

### Infrastructure

```yaml
Cloud: AWS (Primary) + Cloudflare (Edge)
  Why:
    - AWS Mumbai region (data localization)
    - Cloudflare for edge caching, DDoS protection

Container Orchestration: Kubernetes (EKS)
  Why:
    - Auto-scaling
    - Self-healing
    - Rolling deployments
    - Industry standard

CI/CD: GitHub Actions + ArgoCD
  Why:
    - GitHub Actions: Build, test
    - ArgoCD: GitOps deployments
    - Full audit trail

Infrastructure as Code: Pulumi (TypeScript)
  Why:
    - Type-safe infrastructure
    - Same language as frontend
    - Better than Terraform for complex logic

Secrets Management: AWS Secrets Manager + Doppler
  Why:
    - Rotating secrets
    - Environment sync
    - Audit logs

Monitoring:
  - Metrics: Prometheus + Grafana
  - Logs: Loki
  - Traces: Jaeger
  - Errors: Sentry
  - Uptime: Better Uptime

Security:
  - WAF: Cloudflare
  - Secrets: Vault
  - Scanning: Snyk, Dependabot
  - Penetration Testing: Quarterly
```

### Mobile Stack

```yaml
Framework: React Native 0.74 + Expo SDK 52
  Why:
    - Shared logic with web
    - Native performance
    - Expo for faster development
    - OTA updates

Navigation: Expo Router
  Why:
    - File-based routing (like Next.js)
    - Deep linking support

State: Same as web (Zustand + TanStack Query)

Charts: react-native-wagmi-charts
  Why:
    - Native performance
    - Gesture support
    - Financial chart types
```

---

## DATA ARCHITECTURE

### Data Sources

```yaml
Market Data:
  Primary: NSE/BSE Official Feed (paid)
  Backup: TrueData / Global Datafeeds
  Real-time: WebSocket streaming
  Historical: TimescaleDB (5+ years OHLCV)

Fundamental Data:
  Source: Screener.in API / Trendlyne / TickerTape
  Frequency: Daily EOD update
  Coverage: All NSE/BSE listed companies

News & Sentiment:
  Sources:
    - Economic Times API
    - MoneyControl RSS
    - Twitter/X Financial handles
    - Reddit (r/IndiaInvestments, r/IndianStreetBets)
  Processing: Real-time NLP pipeline

Corporate Actions:
  Source: NSE Corporate Announcements
  Types: Dividends, Splits, Bonuses, Rights, Mergers

Insider/Institutional:
  Source: NSE Bulk/Block deals, SEBI disclosures
  Frequency: Daily

Mutual Funds:
  Source: AMFI, MFI API
  Coverage: All SEBI registered MFs
```

### Data Pipeline

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Sources │───▶│  Kafka  │───▶│  Flink  │───▶│   DB    │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
                                  │
                                  ▼
                           ┌─────────────┐
                           │  FinSense   │
                           │  Processing │
                           └─────────────┘
                                  │
                                  ▼
                           ┌─────────────┐
                           │   Redis     │
                           │   (Cache)   │
                           └─────────────┘
```

---

## SECURITY ARCHITECTURE

### Defense in Depth

```
Layer 1: Edge Security
├── Cloudflare WAF (OWASP rules)
├── DDoS protection
├── Bot detection
└── Rate limiting

Layer 2: Application Security
├── Input validation (Zod/Pydantic)
├── SQL injection prevention (parameterized queries)
├── XSS prevention (CSP headers)
├── CSRF tokens
└── JWT with short expiry + refresh tokens

Layer 3: Data Security
├── Encryption at rest (AES-256)
├── Encryption in transit (TLS 1.3)
├── Database column encryption (sensitive fields)
├── PII masking in logs
└── Data anonymization for analytics

Layer 4: Access Control
├── RBAC (Role-based access)
├── Principle of least privilege
├── API key scoping
└── IP allowlisting for admin

Layer 5: Audit & Monitoring
├── All actions logged
├── Anomaly detection
├── Real-time alerts
└── Compliance reporting
```

### Sensitive Data Handling

| Data Type | Storage | Encryption | Retention |
|-----------|---------|------------|-----------|
| PAN Number | PostgreSQL | Column-level AES | 7 years (tax) |
| Aadhaar | Not stored | N/A | Never |
| Phone/Email | PostgreSQL | At rest | Until deletion |
| Portfolio Holdings | PostgreSQL | At rest | Until deletion |
| Trade History | PostgreSQL | At rest | 7 years |
| Session Data | Redis | At rest | 24 hours |
| API Keys | Vault | HSM | Until revoked |

---

## USER EXPERIENCE PRINCIPLES

### Design Philosophy

```
1. CLARITY OVER COMPLEXITY
   - Every number explained
   - No jargon without tooltips
   - Progressive disclosure

2. CONFIDENCE THROUGH TRANSPARENCY
   - Show methodology
   - Explain AI decisions
   - Display data freshness

3. SPEED AS A FEATURE
   - <100ms API responses
   - Instant UI feedback
   - Optimistic updates

4. MOBILE-FIRST, DESKTOP-COMPLETE
   - Core flows work on mobile
   - Power features on desktop
   - Seamless sync

5. ACCESSIBILITY
   - WCAG 2.1 AA compliant
   - Screen reader support
   - Color blind friendly
```

### Onboarding Flow

```
Step 1: Welcome
├── Value proposition (10 seconds)
└── Continue with Google/Email

Step 2: Risk Profile (Required by SEBI)
├── Investment horizon
├── Risk tolerance
├── Income bracket
├── Experience level
└── Generate: Conservative/Moderate/Aggressive

Step 3: Goals
├── Wealth building
├── Retirement
├── Child education
├── House purchase
└── Custom goal

Step 4: Portfolio Setup
├── Import from broker (Zerodha/Groww)
├── Manual entry
└── Start fresh (themes)

Step 5: Personalization
├── Notification preferences
├── Sectors of interest
└── Investment style

Step 6: Dashboard
└── Personalized home with first actions
```

---

## FINSCORE ALGORITHM (Production Version)

### Scoring Methodology

```python
FinScore = weighted_sum([
    (quality_score, 0.20),      # Fundamentals
    (momentum_score, 0.15),     # Price trend
    (value_score, 0.15),        # Valuation
    (sentiment_score, 0.10),    # News/social
    (risk_score, 0.15),         # Volatility/drawdown
    (flow_score, 0.10),         # Institutional/insider
    (regime_fit, 0.10),         # Market condition fit
    (quality_momentum, 0.05),   # Consistency
]) * regime_multiplier * confidence_weight
```

### Quality Score (20%)
```python
quality_score = normalize([
    roe_score,           # ROE vs sector median
    roce_score,          # ROCE vs sector
    margin_stability,    # Margin consistency 5Y
    debt_score,          # D/E ratio
    cash_flow_score,     # FCF yield
    earnings_quality,    # Accruals, one-time items
])
```

### Momentum Score (15%)
```python
momentum_score = normalize([
    price_momentum_3m,
    price_momentum_6m,
    price_momentum_12m,
    relative_strength,   # vs Nifty
    trend_strength,      # ADX-based
    breakout_score,      # 52-week high proximity
])
```

### Value Score (15%)
```python
value_score = normalize([
    pe_percentile,       # vs own history
    pb_percentile,
    ev_ebitda_score,
    peg_ratio,
    dividend_yield,
    intrinsic_discount,  # DCF-based
])
```

### Risk Score (15%)
```python
risk_score = 10 - normalize([  # Inverted (lower risk = higher score)
    volatility_30d,
    volatility_90d,
    max_drawdown_1y,
    beta,
    var_95,
    liquidity_risk,
])
```

### Regime Fit (10%)
```python
# Market regime: Bull/Bear/Sideways/Volatile
regime = detect_market_regime()

regime_fit = {
    'Bull': favor(momentum=True, beta='high'),
    'Bear': favor(quality=True, defensive=True),
    'Sideways': favor(value=True, dividend=True),
    'Volatile': favor(liquidity=True, large_cap=True),
}[regime].score(stock)
```

---

## PRODUCT FEATURES (Detailed)

### 1. Portfolio Hub

**Core Functionality:**
- Multi-broker import (Zerodha, Groww, Upstox, Angel, ICICI Direct)
- Manual entry with smart stock search
- Real-time valuation
- Day P&L, Total P&L, XIRR
- Holdings breakdown (sector, market cap, factor)

**Intelligence Layer:**
- Portfolio Health Score (0-100)
- Concentration risk alerts
- Correlation heatmap
- Factor exposure analysis
- Benchmark comparison

**Actions:**
- Rebalancing suggestions
- "What if" scenario modeling
- Tax harvesting opportunities
- SIP tracking

---

### 2. Stock Intelligence

**Stock Page Sections:**
```
┌─────────────────────────────────────────────────────────────┐
│ HDFC BANK                           FinScore: 8.2/10       │
│ ₹1,642.50  ▲ +2.3% today                                   │
├─────────────────────────────────────────────────────────────┤
│ [Chart: TradingView embed with indicators]                 │
├─────────────────────────────────────────────────────────────┤
│ AI Summary                                                  │
│ "Strong quality score driven by consistent ROE and low     │
│  NPAs. Momentum improving after 3-month consolidation.     │
│  Fairly valued at current levels."                         │
├─────────────────────────────────────────────────────────────┤
│ FinScore Breakdown                                          │
│ Quality:   ████████░░ 8.5                                  │
│ Momentum:  ██████░░░░ 6.2                                  │
│ Value:     ███████░░░ 7.0                                  │
│ Risk:      ████████░░ 8.0                                  │
│ Sentiment: ███████░░░ 7.5                                  │
├─────────────────────────────────────────────────────────────┤
│ Legend Views                                                │
│ 🎯 Buffett: "Strong moat, good capital allocation" (8/10) │
│ 📈 Lynch: "Fair PEG, but growth slowing" (6/10)           │
│ ⚖️ Dalio: "Good risk-adjusted returns" (7/10)             │
├─────────────────────────────────────────────────────────────┤
│ [Fundamentals] [Technicals] [News] [Filings] [Peers]       │
└─────────────────────────────────────────────────────────────┘
```

---

### 3. Smart Themes

**Initial Themes (Launch):**

| Theme | Strategy | Rebalance |
|-------|----------|-----------|
| **India 2035** | Quality growth, infrastructure, consumption | Quarterly |
| **Quality Compounders** | High ROCE, low debt, consistent earnings | Quarterly |
| **Dividend Aristocrats** | Consistent dividend growers | Semi-annual |
| **Momentum Leaders** | Top RS stocks with quality filter | Monthly |
| **Defensive Shield** | Low beta, stable sectors | Quarterly |
| **Buffett's India** | Value + quality + moat | Quarterly |
| **Emerging Giants** | Mid-cap quality growth | Quarterly |

**Theme Page:**
- Constituents with FinScore
- Historical performance
- Risk metrics
- Overlap with user portfolio
- One-click invest (via broker)

---

### 4. Strategy-GPT

**Input:**
```
"Create a strategy that buys quality midcaps with
improving momentum but avoids overvalued stocks"
```

**Output:**
```yaml
Strategy: Quality Midcap Momentum
Universe: NSE Midcap 150

Entry Rules:
  - Market Cap: ₹5,000Cr - ₹50,000Cr
  - ROCE > 15%
  - Debt/Equity < 0.5
  - Price > 50 DMA > 200 DMA
  - RS Rank > 70th percentile
  - P/E < Sector Median

Exit Rules:
  - Price < 200 DMA
  - RS Rank < 30th percentile
  - ROCE deterioration > 30%

Position Sizing:
  - Equal weight
  - Max 20 stocks
  - Max 10% per stock

Backtest (5 Years):
  - CAGR: 22.4%
  - Sharpe: 1.15
  - Max DD: -18.2%
  - Win Rate: 62%
```

---

### 5. Alerts Engine

**Alert Types:**

| Category | Alerts |
|----------|--------|
| **Price** | Above/below level, % change, 52W high/low |
| **FinScore** | Score change, rating upgrade/downgrade |
| **Sentiment** | News spike, sentiment shift |
| **Technical** | Breakout, breakdown, crossovers |
| **Fundamental** | Earnings, dividend, corporate action |
| **Portfolio** | Concentration, drawdown, rebalance due |

**Delivery:**
- Push notification (mobile)
- In-app notification
- Email digest (daily/weekly)
- WhatsApp (premium)

---

## MONETIZATION (DETAILED)

### B2C Pricing

| Tier | Monthly | Annual | Features |
|------|---------|--------|----------|
| **Free** | ₹0 | ₹0 | 1 portfolio, Basic FinScore, 3 alerts, Limited themes |
| **Pro** | ₹299 | ₹2,499 | 5 portfolios, Full FinScore, Unlimited alerts, All themes, Screener |
| **Pro+** | ₹599 | ₹4,999 | Everything in Pro + Legend Agents, Strategy-GPT (5/mo), Priority support |
| **Premium** | ₹999 | ₹7,999 | Everything + Autopilot, Unlimited Strategy-GPT, Tax harvesting, API access |

### B2B Pricing

| Product | Pricing | Volume Discounts |
|---------|---------|------------------|
| **FinScore API** | ₹0.50/call | 50% off >100K/mo |
| **Sentiment API** | ₹0.25/call | 50% off >100K/mo |
| **White-label** | ₹1L/month | Custom |
| **Enterprise** | Custom | Based on AUM |

---

## LAUNCH STRATEGY

### Phase 1: Private Beta (Week 1-6)
- 100 hand-picked users
- Core features only
- Daily feedback calls
- Iterate rapidly

### Phase 2: Public Beta (Week 7-12)
- 5,000 users waitlist
- Invite-only access
- Premium features testing
- Load testing

### Phase 3: Launch (Week 13+)
- Full public access
- PR push
- Influencer partnerships
- Performance marketing

---

## SUCCESS METRICS (OKRs)

### Objective 1: Product-Market Fit
- KR1: 40% weekly active users (WAU/MAU)
- KR2: NPS > 50
- KR3: <5% monthly churn (paid users)

### Objective 2: Growth
- KR1: 50,000 registered users (6 months)
- KR2: 5,000 paid subscribers (6 months)
- KR3: ₹500Cr AUI (6 months)

### Objective 3: Revenue
- KR1: ₹25L MRR (6 months)
- KR2: 3 B2B clients (6 months)
- KR3: <12 month payback on CAC

---

## IMMEDIATE NEXT STEPS

### Week 1: Foundation
1. Set up monorepo (Turborepo)
2. Configure Next.js 15 + FastAPI
3. Set up PostgreSQL + Redis
4. Implement Clerk authentication
5. Create base UI components (shadcn)

### Week 2: Core Data
1. Set up market data pipeline
2. Implement stock data models
3. Create basic stock API
4. Build stock search

### Week 3: Portfolio
1. Manual portfolio entry
2. Holdings CRUD
3. Basic valuation
4. Day P&L calculation

### Week 4: Intelligence
1. FinScore v1 implementation
2. Stock page with score
3. Basic screener
4. Alert foundation

---

*This is not a demo. This is the product specification for SPIKE - a production financial platform that will serve millions of Indian investors.*
