# SPIKE Product Requirements Document (PRD)

**Document Version:** 1.0.0
**Last Updated:** January 27, 2026
**Status:** Living Document

---

## EXECUTIVE SUMMARY

SPIKE is building the world's first **AI Wealth Operating System** - a platform that combines:
- **12-agent AI intelligence engine (FinSense)** for institutional-grade market analysis
- **Universal stock scoring (FinScore)** comparable across all assets
- **Legendary investor personalities** that guide users like Buffett, Lynch, Dalio
- **Autonomous portfolio management (Autopilot)** that self-optimizes
- **Natural language strategy creation (Strategy-GPT)** democratizing quant investing
- **Enterprise API platform** for B2B revenue

This PRD defines both **existing features** and **killer innovations** that will make SPIKE a category-defining product.

---

## PART 1: EXISTING FEATURES (CURRENTLY IMPLEMENTED)

### 1.1 PPO Trading Agent (LIVE)

**Status:** WORKING
**Files:** `agents/ppo_agent.py`, `agents/ppo_trainer.py`, `agents/ppo_memory.py`

**What it does:**
- Actor-Critic neural network for trading decisions
- 33-feature state representation
- BUY/HOLD/SELL discrete actions
- Action masking to prevent invalid trades
- GAE-based advantage estimation

**Current Performance:**
- Sharpe Ratio: 0.29
- Win Rate: 60-75%
- Max Drawdown: <2%

**Technical Details:**
```
Architecture: 128-128 hidden layers
State Size: 33 features
Action Size: 3 (Buy, Hold, Sell)
Learning Rate: 0.001
Clip Epsilon: 0.2
Entropy Coef: 0.05
```

---

### 1.2 Trading Environment (LIVE)

**Status:** WORKING
**File:** `environment/trading_env.py`

**What it does:**
- Simulates trading with full Zerodha cost structure
- Supports intraday (MIS) and delivery (CNC) modes
- Tracks inventory, balance, portfolio value
- Calculates realistic transaction costs

**Zerodha Intraday Costs:**
| Cost | Rate |
|------|------|
| Brokerage | min(₹20, 0.03%) |
| STT (sell) | 0.025% |
| Exchange | 0.00297% |
| SEBI | 0.0001% |
| Stamp (buy) | 0.003% |
| GST | 18% on charges |

---

### 1.3 Real-Time Dashboard (LIVE)

**Status:** WORKING
**Files:** `dashboard/app_fastapi.py`, `dashboard/templates/spike_tailwind.html`

**Features:**
- Glassmorphism UI design
- WebSocket real-time updates
- Live equity curve chart
- Action distribution chart
- Activity feed with trade log
- AI confidence visualization
- Current position panel
- Market stats with flip cards

**Tech Stack:**
- FastAPI + Uvicorn (async)
- WebSocket bidirectional streaming
- Chart.js 4.4 visualization
- Tailwind CSS + custom animations

---

### 1.4 Feature Engineering (LIVE)

**Status:** WORKING
**File:** `utils/features.py`

**17 Technical Indicators:**
1. RSI (14-period)
2. MACD (12, 26, 9)
3. MACD Signal
4. MACD Histogram
5. Bollinger %B
6. ATR (14-period)
7. EMA Fast (12)
8. EMA Slow (26)
9. Trend Strength

**33 Total Features:**
- 19 price differences (sigmoid normalized)
- 1 volume feature
- 9 technical indicators
- 3 trend features

---

## PART 2: KILLER FEATURES (TO BUILD)

### 2.1 FinScore - Universal Stock Intelligence Score

**Priority:** P0 (Critical)
**Estimated Effort:** 4-6 weeks

**What it is:**
A 0-10 score for every stock that represents its overall investment quality, combining 9 dimensions through multi-agent fusion.

**The 9 Dimensions:**

| Dimension | Weight | Source Agent | Example Metrics |
|-----------|--------|--------------|-----------------|
| Quality | 15% | Value Agent | ROE, ROIC, margin stability |
| Trend | 15% | Momentum Agent | Trend strength, reliability |
| Risk | 15% | Risk Agent | Volatility, drawdown, VaR |
| Sentiment | 10% | Sentiment Agent | News tone, social drift |
| Macro Fit | 10% | Macro Agent | Rate sensitivity, sector cycles |
| Flow | 10% | Flow Agent | Promoter buying, FII/DII |
| Behavioral Fit | 10% | Behavioral Agent | User psychology match |
| Regime Fit | 10% | Regime Agent | Current market alignment |
| Alpha Forecast | 5% | Alpha Agent | Expected return probability |

**Why It's Killer:**
- No platform offers a true multi-agent fusion score
- Fully explainable with factor breakdown
- Dynamically adapts to market regimes
- Comparable across all stocks globally
- Patentable architecture

**Implementation Plan:**
1. Build individual agent scoring modules
2. Create fusion layer with learnable weights
3. Add regime-conditional weighting
4. Build explanation generation layer
5. Create API endpoints

---

### 2.2 Legend Agents - Invest Like the Greats

**Priority:** P1 (High)
**Estimated Effort:** 6-8 weeks

**What it is:**
AI agents that embody the investing philosophies of legendary investors, providing personalized recommendations as if Buffett, Lynch, or Dalio were analyzing your portfolio.

**Available Legends:**

| Legend | Philosophy | Key Factors |
|--------|------------|-------------|
| **Warren Buffett** | Value, moats, quality | ROIC>15%, Debt/Equity<0.5, margin stability |
| **Peter Lynch** | Growth at reasonable price | PEG<1.5, earnings acceleration, story clarity |
| **Charlie Munger** | Quality compounders | High ROIC, consistent margins, management quality |
| **Ray Dalio** | All-weather diversification | Macro balance, risk parity, regime adaptation |
| **Naval Ravikant** | Asymmetric optionality | Innovation, network effects, founder quality |
| **Jim Simons** | Quantitative signals | Statistical edge, pattern recognition |

**Architecture:**
```
┌─────────────────────────────────────┐
│     LEGEND AGENT ARCHITECTURE       │
├─────────────────────────────────────┤
│ 1. Philosophy Embedding Layer       │
│    - Vectorized mindset from books  │
│    - Key principles encoded         │
│    - Decision patterns learned      │
├─────────────────────────────────────┤
│ 2. Agent Reasoning Engine           │
│    - Apply philosophy to stock data │
│    - Generate Legend Score (0-100)  │
│    - Produce reasoning chain        │
├─────────────────────────────────────┤
│ 3. Personalization Overlay          │
│    - Adjust for user's risk level   │
│    - Consider holding period        │
│    - Factor in behavioral profile   │
└─────────────────────────────────────┘
```

**Sample Output:**
```
Legend: Warren Buffett
Stock: HDFC Bank
Legend Score: 82/100

"HDFC Bank exhibits characteristics I look for: durable competitive
advantage in Indian banking, consistent 18%+ ROE over 10 years,
conservative lending practices, and management with skin in the game.
The current price offers a modest discount to intrinsic value.
Would consider a position if you have a 5+ year horizon."

Confidence: High
Action: BUY (with patience)
```

**Why It's Killer:**
- No platform offers personality-driven AI investing
- Massive viral potential (imagine sharing "Buffett's take on your portfolio")
- Deep engagement loop
- Educational value
- Premium feature for monetization

---

### 2.3 Strategy-GPT - Natural Language to Quant Strategy

**Priority:** P1 (High)
**Estimated Effort:** 8-10 weeks

**What it is:**
Users describe a strategy in plain English, and SPIKE converts it to a fully functional quant strategy with rules, backtests, and optimizations.

**User Experience:**
```
User: "Build me a momentum strategy that buys breakouts
       but avoids overextended stocks with low volatility"

Strategy-GPT Output:
┌────────────────────────────────────────────────────┐
│ STRATEGY: Momentum Breakout v1.0                   │
├────────────────────────────────────────────────────┤
│ ENTRY RULES:                                       │
│ • Price > 20-day high (breakout)                   │
│ • RSI < 70 (not overextended)                      │
│ • ATR% < 3% (controlled volatility)                │
│ • Volume > 1.5x 20-day average                     │
├────────────────────────────────────────────────────┤
│ EXIT RULES:                                        │
│ • Trailing stop: 2x ATR                            │
│ • Take profit: +15% from entry                     │
│ • Time stop: 20 days max hold                      │
├────────────────────────────────────────────────────┤
│ BACKTEST (2020-2025):                              │
│ CAGR: 18.4%                                        │
│ Max Drawdown: -12.3%                               │
│ Sharpe: 1.21                                       │
│ Win Rate: 58%                                      │
├────────────────────────────────────────────────────┤
│ EXPLANATION:                                       │
│ This strategy captures momentum continuation while │
│ filtering out unstable breakouts. The volatility   │
│ filter ensures consistent performance across       │
│ market regimes.                                    │
└────────────────────────────────────────────────────┘
```

**6-Layer Architecture:**
1. **Intent Parsing (LLM)** - Understand user goals, timeframe, risk
2. **Rule Generator** - Convert intent to quant rules
3. **Multi-Agent Integration** - Pull signals from FinSense agents
4. **Backtesting Engine** - Run historical simulation
5. **Optimization Layer** - RL/evolutionary search for parameters
6. **Explanation Layer** - Generate human-readable insights

**Why It's Killer:**
- Democratizes quant investing for everyone
- No competitor offers NL→Strategy conversion
- Viral shareability (users share their strategies)
- Massive B2B value (wealth managers, fintechs)
- Deep engagement loop (iterate, improve, test)

---

### 2.4 Portfolio Autopilot - AI Wealth Management

**Priority:** P1 (High)
**Estimated Effort:** 10-12 weeks

**What it is:**
Fully autonomous portfolio management that allocates, rebalances, and protects your wealth using multi-agent intelligence.

**Modes:**

| Mode | Description | Best For |
|------|-------------|----------|
| **Passive** | Stable, low volatility, max safety | Beginners, risk-averse |
| **Balanced** | Mix stability + growth, adapts to regimes | Most users |
| **Aggressive** | Higher momentum, accepts drawdown | Risk-tolerant |
| **Custom** | Strategy-GPT defined | Advanced users |

**7-Layer Intelligence Pipeline:**
1. **Portfolio Deep Analysis** - Factor exposure, drawdown risk, concentration
2. **FinScore Integration** - Weight by stock quality
3. **Alpha Forecast Layer** - Predict returns, momentum stability
4. **Regime Awareness** - Switch strategies by market state
5. **Risk Intelligence** - Control volatility, correlation, liquidity
6. **Behavioral Intelligence** - Adapt to user's emotional profile
7. **Decision Engine** - Output weights, recommendations, actions

**Functions:**
- Auto-Rebalancing (maintain target weights)
- Auto-Allocation (shift to high-confidence assets)
- Auto-Risk Control (dynamic position sizing)
- Auto-Profit Booking (when signals weaken)
- Auto-Stoploss (adaptive, not static)
- Auto-Behavioral Correction (prevent panic selling)

**Why It's Killer:**
- Transforms SPIKE from tool to full AI wealth manager
- "Tesla Autopilot for investing"
- Highest retention feature
- Premium subscription driver
- Massive competitive moat

---

### 2.5 Regime Engine - Market State Intelligence

**Priority:** P0 (Critical)
**Estimated Effort:** 3-4 weeks

**What it is:**
Classifies the current market into one of 7 regimes and adjusts all SPIKE systems accordingly.

**7 Market Regimes:**

| Regime | Characteristics | Strategy Implications |
|--------|-----------------|----------------------|
| **Bull Strong** | Strong uptrend, low vol, FOMO | Trend-following, high exposure |
| **Bull Weak** | Uptrend fading, early vol expansion | Reduce exposure, tighten stops |
| **Sideways** | No clear trend, choppy | Mean-reversion, range trades |
| **Bear Weak** | Controlled decline, risk-off starts | Defensive tilt, reduce cyclicals |
| **Bear Strong** | Sharp decline, fear dominant | Cash, quality, hedges |
| **High Volatility** | Large swings, VIX spiking | Reduce size, wait for clarity |
| **Macro Stress** | Inflation/rate/credit shocks | Safe havens, gold, quality |

**How It Works:**
```
Inputs:
├── Price action (trend, volatility)
├── Sentiment (fear/greed index)
├── Macro indicators (rates, inflation)
├── Market breadth (advance/decline)
└── Volatility metrics (VIX, skew)
    │
    ▼
┌─────────────────────────┐
│ Regime Classification   │
│ (Multi-model ensemble)  │
└─────────────────────────┘
    │
    ▼
Outputs:
├── Current regime label
├── Confidence score
├── Transition probabilities
├── Strategy recommendations
└── Factor adjustments
```

**Why It's Critical:**
- All SPIKE features depend on regime awareness
- Massive alpha from regime-conditional strategies
- Risk management improvement
- User trust through transparency

---

### 2.6 Smart Themes - AI-Curated Investment Baskets

**Priority:** P2 (Medium)
**Estimated Effort:** 4-6 weeks

**What it is:**
Pre-built, AI-curated baskets of stocks unified by a theme, factor, or philosophy.

**Theme Categories:**

**1. Macro Themes:**
- India 2030 Growth
- Manufacturing Renaissance
- EV Revolution
- Digital India
- Renewable Energy
- Healthcare Boom

**2. Factor Themes:**
- Momentum Titans (high trend strength)
- Quality Leaders (high ROIC, stable margins)
- Value Picks (low P/E, high dividend)
- Low-Vol Stability (defensive, consistent)

**3. Legend Themes:**
- Buffett Compounders
- Lynch Growth Gems
- Dalio All-Weather
- Naval Optionality Portfolio

**4. Behavioral Themes:**
- Emotion-Proof (for anxious investors)
- High-Confidence Winners (for disciplined)
- Long-Term Patience Pack

**5. Tactical Themes:**
- Risk-On Momentum
- Defensive Shield
- Rate-Hike Beneficiaries
- Recession-Proof

**Theme Construction Pipeline:**
1. Universe screening (market cap, liquidity)
2. FinScore filtering (quality threshold)
3. Factor selection (based on theme)
4. Alpha forecasting (expected returns)
5. Regime modeling (current market fit)
6. Risk filtering (concentration, correlation)
7. Behavioral personalization (user fit)
8. Weight optimization (RL-based)

**Why It's Killer:**
- "Invest in an idea" simplicity
- Viral screenshot potential
- Alternative to mutual funds
- Mass-market adoption driver

---

### 2.7 Backtesting Lab - Professional Simulation

**Priority:** P2 (Medium)
**Estimated Effort:** 4-6 weeks

**What it is:**
A professional-grade backtesting engine with Bloomberg-level metrics, simplified for retail users.

**Outputs:**
- CAGR, Sharpe, Sortino, Calmar
- Maximum Drawdown, Recovery Time
- Win Rate, Profit Factor
- Monthly/Yearly Returns Heatmap
- Regime-Specific Performance
- Factor Attribution
- Stress Test Results

**Stress Tests:**
- 2008 Financial Crisis
- COVID-19 Crash
- Taper Tantrum 2013
- Rate Hike Cycle 2022
- Currency Crisis Scenario
- Sector Meltdown

**Why It's Important:**
- Validates Strategy-GPT outputs
- Builds user trust
- Professional credibility
- B2B selling point

---

### 2.8 Real-Time Alerts Engine

**Priority:** P2 (Medium)
**Estimated Effort:** 3-4 weeks

**Alert Types:**
1. **Priority Alerts** - Time-sensitive, high-impact
2. **Risk Alerts** - Volatility spikes, drawdown warnings
3. **Sentiment Alerts** - News tone shifts
4. **Regime Alerts** - Market state transitions
5. **Price Action Alerts** - Breakouts, reversals
6. **Corporate Action Alerts** - Earnings, dividends, splits

**Personalization:**
- Frequency based on user behavior
- Tone adjusted to emotional profile
- Relevance scored by portfolio exposure

---

### 2.9 Execution Intelligence

**Priority:** P3 (Lower)
**Estimated Effort:** 6-8 weeks

**What it does:**
Optimizes trade execution timing to minimize slippage and maximize fill quality.

**Components:**
- Smart timing (volatility windows)
- Slippage prediction
- Liquidity awareness
- TWAP/VWAP execution
- Order book analysis

**Broker Integration:**
- Zerodha Kite API
- Upstox
- Angel Broking
- Groww (when available)

---

## PART 3: IMPLEMENTATION PRIORITIES

### Phase 1: Foundation (Current - Feb 2026)
| Feature | Priority | Status |
|---------|----------|--------|
| Wipro PPO Training | P0 | In Progress |
| Candlestick Charts | P1 | Not Started |
| Risk Metrics Panel | P1 | Not Started |
| Regime Engine v1 | P0 | Not Started |

### Phase 2: Core Intelligence (Feb-Apr 2026)
| Feature | Priority | Effort |
|---------|----------|--------|
| FinScore v1.0 | P0 | 4-6 weeks |
| Sentiment Agent v1 | P1 | 3-4 weeks |
| Backtesting Lab | P2 | 4-6 weeks |
| Alerts Engine | P2 | 3-4 weeks |

### Phase 3: Killer Features (Apr-Jul 2026)
| Feature | Priority | Effort |
|---------|----------|--------|
| Legend Agents | P1 | 6-8 weeks |
| Strategy-GPT MVP | P1 | 8-10 weeks |
| Smart Themes v1 | P2 | 4-6 weeks |

### Phase 4: Full Autonomy (Jul-Dec 2026)
| Feature | Priority | Effort |
|---------|----------|--------|
| Portfolio Autopilot | P1 | 10-12 weeks |
| Execution Intelligence | P3 | 6-8 weeks |
| B2B API Platform | P1 | 8-10 weeks |

---

## PART 4: SUCCESS METRICS

### Product KPIs
| Metric | Target | Timeline |
|--------|--------|----------|
| FinScore accuracy (vs future returns) | >65% correlation | Q2 2026 |
| Autopilot Sharpe Ratio | >0.8 | Q3 2026 |
| Strategy-GPT backtest accuracy | >80% | Q2 2026 |
| User retention (30-day) | >60% | Q3 2026 |
| B2B API calls/month | 1M+ | Q4 2026 |

### Technical KPIs
| Metric | Target |
|--------|--------|
| Dashboard latency | <100ms |
| WebSocket stability | >99.9% uptime |
| Model inference time | <50ms |
| Backtest speed | 1000 trades/sec |

### Business KPIs
| Metric | Target | Timeline |
|--------|--------|----------|
| Premium subscribers | 10,000 | Q4 2026 |
| B2B enterprise clients | 10 | Q4 2026 |
| AUM (personal trading) | ₹1Cr | Q2 2026 |
| Monthly revenue | ₹10L | Q4 2026 |

---

## PART 5: RISKS & MITIGATIONS

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Overfitting models | High | Medium | Walk-forward validation, ensemble methods |
| Regime change failure | High | Medium | Continuous retraining, regime detection |
| Regulatory issues | High | Low | Legal counsel, compliance-first design |
| Competitor copying | Medium | Medium | Patent filings, rapid iteration |
| Technical debt | Medium | High | Code reviews, test coverage |

---

## APPENDIX: PATENT OPPORTUNITIES

1. **Multi-Agent Fusion Scoring System** (FinScore)
2. **Philosophy-Embedded Investment Agents** (Legend Agents)
3. **Natural Language to Quant Strategy Compiler** (Strategy-GPT)
4. **Behavioral-Aware Portfolio Optimization** (Autopilot)
5. **Regime-Conditioned Dynamic Weighting**
6. **Synthetic Market Environment for RL Training**
7. **Multi-Timeframe Trend Consensus Algorithm**
8. **Sentiment Drift Prediction Model**

---

*This PRD is a living document. Update as features ship and priorities shift.*
