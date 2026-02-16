# SPIKE KNOWLEDGE BASE - THE BIBLE

**Last Updated:** January 27, 2026
**Version:** 1.0.0
**Maintained By:** Claude AI + Founders

---

## TABLE OF CONTENTS
1. [Project Identity](#1-project-identity)
2. [Current State](#2-current-state)
3. [Architecture Overview](#3-architecture-overview)
4. [FinSense - The 12-Agent System](#4-finsense---the-12-agent-system)
5. [Core Products](#5-core-products)
6. [Technical Implementation](#6-technical-implementation)
7. [Business Model](#7-business-model)
8. [Roadmap & Milestones](#8-roadmap--milestones)
9. [Development Log](#9-development-log)
10. [Active Decisions & Context](#10-active-decisions--context)

---

## 1. PROJECT IDENTITY

### What is SPIKE?
**SPIKE** is an **AI Wealth Operating System** - the world's first platform that unifies:
- A **retail super-app** for intelligent, automated investing
- An **enterprise-grade B2B financial AI infrastructure platform**
- A **multi-agent AI engine (FinSense)** capable of reasoning about markets
- A **universal stock scoring system (FinScore)**
- **Personality-driven investment agents (Legend Agents)**
- **Strategy generation AI (Strategy-GPT)**
- **Synthetic market environments for RL training**
- **Risk, sentiment, macro, and behavioral intelligence**

### Mission Statement
> "Make professional-grade financial intelligence accessible, explainable, and autonomous for every investor and institution."

### Vision Statement
> "To build the world's most trusted AI Wealth Operating System—powering autonomous investing for individuals, fintechs, and financial institutions globally."

### Core Beliefs (The SPIKE Worldview)
1. **Finance Is No Longer Human-Scale** - AI should augment human financial decision-making
2. **Investing Should Be Intelligent, Not Emotional** - Behavioral finance + data-driven decisions
3. **Intelligence Must Be Democratic** - Institutional-grade tools for everyone
4. **The Future of Wealth Will Be Autonomous** - Portfolios that self-manage
5. **People Want Decisions, Not Data** - Clarity over complexity
6. **AI Must Be Explainable** - No black boxes, full transparency
7. **Multi-Agent AI > Single Model** - Collective intelligence beats monolithic models
8. **Every Investor Is Unique** - Deep personalization required
9. **APIs Are the New Distribution Layer** - B2B infrastructure is core
10. **Highest Moat = IP + AI + Data + Personalization**

### Brand Identity
- **Brand Essence:** "Intelligence, Simplified"
- **Archetype:** The Sage + The Architect (wisdom + precision)
- **Colors:** Royal Midnight Blue (#0A0E29), Imperial Gold (#D4AF37), Deep Space Black (#00010A)
- **Tone:** Calm, intelligent, professional, minimal, authoritative
- **Taglines:** "Clarity Above Chaos", "Your AI Wealth OS", "Powered by FinSense"

---

## 2. CURRENT STATE

### What's Built (January 2026)

#### Core ML Components
| Component | Status | Details |
|-----------|--------|---------|
| PPO Agent | WORKING | 450 episodes trained on RELIANCE.NS, Sharpe 0.29 |
| DQN Agent | LEGACY | Replaced by PPO for better performance |
| Trading Environment | WORKING | Full Zerodha intraday cost modeling |
| Feature Engineering | WORKING | 17 technical indicators, 33 features |
| Action Masking | WORKING | Prevents invalid trades |

#### Infrastructure
| Component | Status | Details |
|-----------|--------|---------|
| FastAPI Backend | WORKING | WebSocket support, async |
| Dashboard (Tailwind) | WORKING | Glassmorphism design, real-time updates |
| Paper Trading Engine | WORKING | Full simulation with realistic costs |
| Data Pipeline | WORKING | yfinance integration, multi-stock support |
| Training Scripts | WORKING | `./train_wipro.sh` with nohup + monitoring |

#### Key Metrics Achieved
- **Sharpe Ratio:** 0.29 (target: >0.5)
- **Win Rate:** 60-75%
- **Max Drawdown:** <2%
- **Code Quality:** 75+ unit tests, >50% coverage

### What's In Progress
1. **Wipro Training** - Configured for 450 episodes with Zerodha intraday costs
2. **Dashboard Enhancements** - Candlestick charts, risk metrics panel
3. **Multi-stock UI** - Switch between stocks in dashboard

### Key Files
```
/Users/jay/FinSense-1/
├── agents/
│   ├── ppo_agent.py          # Main PPO actor-critic network
│   ├── ppo_memory.py         # Rollout storage + GAE
│   ├── ppo_trainer.py        # PPO update mechanism
│   └── dqn_agent.py          # Legacy DQN (reference only)
├── environment/
│   └── trading_env.py        # Trading simulation with Zerodha costs
├── dashboard/
│   ├── app_fastapi.py        # FastAPI + WebSocket backend
│   ├── templates/spike_tailwind.html  # Main dashboard UI
│   └── static/js/premium_ultimate.js  # Chart.js + animations
├── config.yaml               # All hyperparameters
├── train_ppo.py              # Main training script
├── train_wipro.sh            # Training launcher with monitoring
└── start_spike_tailwind.sh   # Dashboard launcher
```

---

## 3. ARCHITECTURE OVERVIEW

### Three-Layer Ecosystem

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 1: SPIKE RETAIL APP                    │
│         (B2C Super-App for Retail Investors)                    │
│  FinScore | Legend Agents | Portfolio Autopilot | Strategy-GPT  │
│  Backtesting Lab | AI Insights | Behavioral Intelligence        │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                   LAYER 2: FINSENSE AI ENGINE                   │
│              (Core Intelligence - 12 Agents)                     │
│  Alpha | Risk | Sentiment | Macro | Value | Momentum | Portfolio │
│  Regime | Behavioral | Insider | Alt Data | Simulation          │
│                    → FUSION ENGINE → FinScore                   │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                  LAYER 3: SPIKE API PLATFORM                    │
│               (B2B Infrastructure for Enterprises)               │
│  FinScore API | Sentiment API | Risk API | Portfolio API        │
│  Strategy-GPT API | Backtesting API | Regime API                │
└─────────────────────────────────────────────────────────────────┘
```

### The SPIKE Flywheel
```
Retail Users → Generate Data → Improve FinSense → Strengthen API Value
    ↑                                                        ↓
    └──── More Retail Users ← Better Product ← More R&D ← Revenue ← Enterprise Clients
```

---

## 4. FINSENSE - THE 12-AGENT SYSTEM

### Agent Descriptions

| Agent | Purpose | Key Outputs |
|-------|---------|-------------|
| **Alpha Forecasting** | Predict short/medium/long-term returns | Return probability, confidence |
| **Risk Agent** | Calculate VaR, CVaR, drawdown, volatility | Risk Score (0-100) |
| **Sentiment Agent** | LLM-powered news/social analysis | Sentiment Score, drift direction |
| **Macro Agent** | Track inflation, rates, liquidity, commodities | Macro regime, sector signals |
| **Value Agent** | Analyze fundamentals, ROIC, margins | Quality Score, valuation |
| **Momentum Agent** | Detect trend strength, reliability | Trend Score, reversal probability |
| **Portfolio Agent** | Optimize allocation (RL + Black-Litterman) | Weight recommendations |
| **Regime Agent** | Classify market state | Bull/Bear/Sideways/Volatility |
| **Behavioral Agent** | Model user psychology | Panic/FOMO/Overconfidence detection |
| **Insider Flow Agent** | Track promoter/FII/DII activity | Flow Score, accumulation signals |
| **Alternative Data Agent** | Satellite, web traffic, app data | Alternative signals |
| **Simulation Agent** | Synthetic market generation for RL | Training environments |

### Fusion Engine Formula (Conceptual)
```
FinScore = F(
    Q,      # Quality (Value Agent)
    T,      # Trend (Momentum Agent)
    R,      # Risk (Risk Agent)
    S,      # Sentiment (Sentiment Agent)
    M,      # Macro (Macro Agent)
    I,      # Insider Flow (Flow Agent)
    B,      # Behavioral Fit (Behavioral Agent)
    G,      # Regime Fit (Regime Agent)
    A,      # Alpha Forecast (Alpha Agent)
    regime_state,
    volatility_state,
    agent_confidences,
    cross_agent_consistency
)
```

### Market Regimes (7 States)
1. **Bull Strong** - Strong uptrend, low volatility, FOMO sentiment
2. **Bull Weak** - Uptrend losing steam, early volatility expansion
3. **Sideways Neutral** - No clear trend, mean-reversion dominates
4. **Bear Weak** - Controlled downtrend, early risk-off
5. **Bear Strong** - Sharp decline, fear-dominated, volatility spikes
6. **High Volatility** - Large price swings, VIX rising
7. **Macro Stress** - Inflation/credit/rate shocks, geopolitical events

---

## 5. CORE PRODUCTS

### 5.1 FinScore (0-10 Universal Stock Rating)

**What it measures (9 dimensions):**
1. Quality (ROE, ROIC, margins)
2. Trend Strength (momentum stability)
3. Risk (volatility, drawdown)
4. Sentiment (news tone, social)
5. Macro Alignment (rate sensitivity, FX)
6. Insider & Institutional Flow
7. Behavioral Compatibility (user fit)
8. Regime Fitness (current market fit)
9. Expected Return (Alpha prediction)

**Why it's revolutionary:**
- Multi-agent fusion score
- Fully explainable with factor breakdown
- RL-enhanced predictive layer
- Comparable across all stocks globally

### 5.2 Legend Agents (AI Investing Personalities)

**Available Legends:**
- Warren Buffett (value, moats, ROIC)
- Peter Lynch (growth, PEG, story)
- Naval Ravikant (optionality, asymmetry)
- Ray Dalio (macro, all-weather)
- Charlie Munger (quality, consistency)
- Jim Simons (quant, statistical)

**Architecture:**
1. **Philosophy Embedding Layer** - Vectorized mindset from books/letters
2. **Agent Reasoning Engine** - Apply philosophy to FinSense outputs
3. **Personalization Overlay** - Adjust for user's profile

**Output:** Legend Score (0-100), Buy/Hold/Avoid, Reasoning, Confidence

### 5.3 Portfolio Autopilot

**Modes:**
1. Passive (Safe) - Stable, low volatility, max safety
2. Balanced (Default) - Mix stability + growth, adapts to regimes
3. Aggressive (Growth) - Higher momentum exposure, accepts drawdown
4. Custom (Strategy-GPT) - User-defined via natural language

**Functions:**
- Auto-Rebalancing
- Auto-Allocation
- Auto-Risk Control
- Auto-Profit Booking
- Auto-Stoploss (adaptive)
- Auto-Behavioral Correction

### 5.4 Strategy-GPT

**What it does:**
User describes strategy in natural language → SPIKE outputs:
- Complete rules & signals
- Factor weights & filters
- Risk layer & constraints
- Backtest results
- Explanations & improvements

**Example:**
Input: "Build me a Buffett-style long-term compounding portfolio"
Output: Rules for ROIC>15%, Debt/Equity<0.5, etc. + backtest showing 15.7% CAGR

### 5.5 Smart Themes (AI Baskets)

**Types:**
1. **Macro Themes** - India 2030, EV Revolution, Manufacturing Boom
2. **Factor Themes** - Momentum Titans, Quality Leaders, Value Picks
3. **Legend Themes** - Buffett Compounding, Naval Optionality
4. **Behavioral Themes** - Low-Volatility Comfort, Emotion-Proof
5. **Tactical Themes** - Risk-On Momentum, Defensive Shield

**Features:**
- FinScore-weighted selection
- Regime-aware rebalancing
- Behavioral personalization
- Dynamic factor adjustment

---

## 6. TECHNICAL IMPLEMENTATION

### 6.1 Current PPO Configuration
```yaml
ppo:
  lr: 0.001           # Learning rate (3x faster than 0.0003)
  gamma: 0.99         # Discount factor
  gae_lambda: 0.95    # GAE bias-variance tradeoff
  clip_eps: 0.2       # PPO clipping range
  value_coef: 0.5     # Value loss weight
  entropy_coef: 0.05  # Entropy regularization (5x higher)
  epochs: 4           # PPO epochs per rollout
  batch_size: 128     # Mini-batch size
```

### 6.2 State Representation (33 Features)
- Price differences (19) - Sigmoid-normalized daily changes
- Volume feature (1) - Volume change ratio
- Technical indicators (9) - RSI, MACD(3), Bollinger %B, ATR
- Trend features (3) - EMA diff, EMA slope, trend strength

### 6.3 Zerodha Intraday Costs
```yaml
brokerage: min(₹20, 0.03%)
stt_sell: 0.025%      # Sell side only
exchange_charges: 0.00297%
sebi_charges: 0.0001%
stamp_duty_buy: 0.003%
gst: 18% on charges
```

### 6.4 Reward Function
```
reward = ((new_portfolio_value / prev_portfolio_value) - 1) * 100
# Penalties:
- Action change: -0.01%
- Idle holding with no inventory: -0.02%
- Invalid trade attempt: -0.01%
```

### 6.5 Action Masking
```python
mask = [True, True, True]  # [BUY, HOLD, SELL]
if len(inventory) == 0:
    mask[2] = False  # Cannot SELL without inventory
if balance < current_price or len(inventory) >= max_positions:
    mask[0] = False  # Cannot BUY
```

---

## 7. BUSINESS MODEL

### Revenue Streams

#### A. Retail (B2C)
- **Freemium tier** - Basic FinScore, limited themes
- **Premium subscription** - Full access, Autopilot, Strategy-GPT
- **AUM-based fees** - For managed portfolios

#### B. Enterprise (B2B)
- **API subscriptions** - FinScore API, Sentiment API, etc.
- **White-label dashboards** - For brokers/fintechs
- **Custom integrations** - Enterprise deployments

### Pricing Philosophy
- Management Fee: 2% of AUM
- Performance Fee: 20% above 10% hurdle
- API pricing: Volume-based tiers

### Growth Path
1. **Bootstrap (Months 1-3)** - ₹10-25L personal capital, build track record
2. **Seed (Months 4-6)** - ₹50L-2Cr external, scale infra
3. **Series A (Year 1+)** - ₹5-10Cr, SEBI registration, manage external capital

---

## 8. ROADMAP & MILESTONES

### Phase 1: Foundation (COMPLETE)
- [x] PPO agent with action masking
- [x] Technical indicators (17 features)
- [x] Percentage-based reward function
- [x] Production-ready transaction costs
- [x] Enterprise dashboard with glassmorphism

### Phase 2: Current Sprint (January 2026)
- [x] Wipro training configuration
- [x] Zerodha intraday costs
- [ ] Candlestick chart visualization
- [ ] Risk metrics panel (VaR, CVaR)
- [ ] Multi-stock UI switching

### Phase 3: Short-term (February 2026)
- [ ] FinScore v1.0 implementation
- [ ] Basic sentiment analysis (news headlines)
- [ ] Backtesting comparison view
- [ ] Walk-forward validation

### Phase 4: Medium-term (Q1-Q2 2026)
- [ ] Legend Agents v1.0 (Buffett, Lynch)
- [ ] Strategy-GPT MVP
- [ ] Zerodha Kite API integration
- [ ] Smart Themes v1.0
- [ ] Mobile app (React Native)

### Phase 5: Long-term (Q3-Q4 2026)
- [ ] Full 12-agent FinSense
- [ ] Portfolio Autopilot v1.0
- [ ] B2B API platform launch
- [ ] SEBI registration preparation
- [ ] International expansion (US, SEA)

---

## 9. DEVELOPMENT LOG

### January 27, 2026
- Configured Wipro training with Zerodha intraday costs
- Created `train_wipro.sh` with nohup + monitoring
- Created `monitor_training.sh` for live progress tracking
- Updated startup script with ASCII art and loading animations
- Created comprehensive KNOWLEDGE.md

### January 26, 2026
- Fixed right sidebar overflow (Activity Feed pushing off AI Confidence)
- Implemented glassmorphism dashboard redesign (via Gemini)
- Updated colors to Emerald (#10B981) palette
- Added flip card animations for market stats
- Pushed to experimental/reward-tuning branch

### January 23, 2026
- Complete Tailwind CSS rebuild to fix CSS caching issues
- Implemented exact 3-column layout matching WhatsApp screenshot
- Created GEMINI_FRONTEND_BRIEF.md for frontend enhancement handoff

### Earlier Milestones
- PPO training completed (450 episodes on RELIANCE.NS)
- Achieved Sharpe 0.29, Win Rate 60-75%
- Dashboard WebSocket implementation
- Chart.js real-time visualization

---

## 10. ACTIVE DECISIONS & CONTEXT

### Current Training Configuration
```yaml
Stock: WIPRO.NS
Episodes: 450
Trading Mode: Intraday (Zerodha MIS)
Initial Balance: ₹50,000
Max Positions: 40
Window Size: 20
```

### Active Branch
`experimental/reward-tuning`

### Running Services
- Dashboard: `http://localhost:8000` (FastAPI)
- Start command: `./start_spike_tailwind.sh`

### Key Decisions Made
1. **PPO over DQN** - Better for continuous action refinement
2. **Percentage-based rewards** - Fixed scale mismatch issues
3. **Single-stock training** - Isolate learning before multi-stock
4. **Zerodha costs** - Realistic Indian market simulation
5. **Tailwind CSS** - Bypassed CSS caching nightmare

### Open Questions
1. Optimal entropy coefficient for exploration/exploitation balance?
2. Best approach for multi-stock generalization?
3. When to integrate real-time data vs historical?

---

## APPENDIX A: PATENTABLE COMPONENTS

1. **Multi-Agent Fusion Engine** - Combining 12 specialized agents
2. **FinScore Algorithm** - Universal 0-10 stock rating
3. **Legend Agents Architecture** - Philosophy embedding + reasoning
4. **Strategy-GPT** - NL to quant strategy compiler
5. **Behavioral Intelligence Engine** - User psychology modeling
6. **Regime-Aware Dynamic Weighting** - Adaptive scoring
7. **Synthetic Market Simulator** - RL training environments

---

## APPENDIX B: QUICK COMMANDS

```bash
# Start dashboard
./start_spike_tailwind.sh

# Train on Wipro
./train_wipro.sh

# Monitor training
./monitor_training.sh

# Quick live view of training
tail -f logs/wipro_training_*.log

# Stop training
kill $(cat logs/training.pid)

# Check if port 8000 is running
lsof -ti:8000
```

---

## APPENDIX C: CONTACTS & RESOURCES

- **Founder:** Jayaditya Reddy Yeruva
- **Co-founder:** Adip Krishna Guduru
- **Repository:** /Users/jay/FinSense-1
- **Branch:** experimental/reward-tuning

---

*This document is the single source of truth for SPIKE. Update it continuously as development progresses.*
