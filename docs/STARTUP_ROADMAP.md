# FinSense Startup Roadmap
**Senior Quant Executive Assessment & Strategic Plan**

---

## Executive Summary

**Current Status:** Prototype RL trading system with fundamental configuration issues resolved.

**Key Finding:** Agent is learning, but position limits were misconfigured. After 300 episodes, the policy shows clear preference structure (BUY > HOLD > SELL), but could only deploy 12.5% of capital due to max_positions=5 constraint.

**Immediate Fix Applied:** Increased max_positions to 50, allowing proper capital deployment.

---

## Phase 1: Foundation (Current - Week 4)
**Goal:** Prove the RL agent can trade profitably on backtests

### Week 1-2: Configuration & Quick Validation ✅ IN PROGRESS
- [x] Fix position sizing limits (max_positions: 5 → 50)
- [ ] **Run 50-episode test** with new config (6-12 hours)
- [ ] Verify agent trades actively on test set (target: 50-200 trades)
- [ ] Confirm positive Sharpe ratio (target: >0.3)

**Success Criteria:**
- Test set trades: >50 trades (not 0)
- Sharpe ratio: >0.2
- Max drawdown: <20%
- Win rate: >50%

### Week 3-4: Full Training & Validation
If 50-episode test passes:
- [ ] Run full 300-episode training
- [ ] Comprehensive evaluation on all 5 stocks separately
- [ ] Walk-forward validation (train on 2020-2023, test on 2024)
- [ ] Stress test on different market conditions (bull, bear, sideways)

**Success Criteria:**
- Consistent performance across stocks
- Sharpe >0.4 on out-of-sample data
- Survives 2022 bear market test

---

## Phase 2: Production Readiness (Week 5-8)
**Goal:** Paper trading + risk management

### Risk Management System
```python
Key components needed:
1. Position sizing based on Kelly Criterion
2. Stop-loss at portfolio level (-5% daily drawdown limit)
3. Maximum position size validation
4. Cash reserve requirements (5-10%)
5. Correlation-based portfolio limits (for multi-stock)
```

### Paper Trading Infrastructure
- [ ] Zerodha Kite Connect API integration
- [ ] Real-time data pipeline (WebSocket)
- [ ] Order management system (OMS)
- [ ] Live monitoring dashboard
- [ ] Alert system (Telegram/Email)

**Success Criteria:**
- 3 months paper trading profitability
- Sharpe >0.3 in live conditions
- No catastrophic failures (max DD <15%)
- Execution latency <500ms

---

## Phase 3: Regulatory & Legal (Week 9-12)
**Goal:** Compliance for managing external capital

### SEBI Registration Requirements
1. **Investment Adviser (IA) Registration** - If providing advice
2. **Portfolio Management Services (PMS)** - If managing >₹50 lakhs
3. **Alternative Investment Fund (AIF Category III)** - For algorithmic trading fund

**Estimated Costs:**
- IA Registration: ₹1-2 lakhs + ₹10k annual
- PMS License: ₹10-20 lakhs + compliance costs
- AIF Setup: ₹50 lakhs - ₹1 crore (legal, compliance, infrastructure)

**Timeline:** 6-12 months for approvals

### Minimum Capital Requirements
- **SEBI PMS:** ₹5 crores net worth, ₹2 crores liquid
- **AIF Category III:** ₹20 crores corpus minimum
- **Algo Trading API:** ₹5 lakhs-₹10 lakhs from brokers

---

## Phase 4: Scaling Strategy (Month 4-6)

### Technical Scaling
1. **Multi-Strategy Ensemble**
   - Trend-following PPO (current)
   - Mean-reversion DQN
   - Volatility arbitrage
   - Combine with weighted allocation

2. **Multi-Timeframe**
   - Daily (current)
   - Hourly for intraday
   - 5-min for high-frequency opportunities

3. **Extended Universe**
   - Nifty 50 stocks (50 agents)
   - Sector rotation
   - F&O strategies (futures, options)

### Infrastructure Requirements
- **Compute:** GPU server (₹2-5 lakhs)
- **Data:** Real-time feeds (₹50k-₹2L/year)
- **Backup:** Redundant systems, failover
- **Monitoring:** 24/7 ops team or alerting

---

## Critical Risks & Mitigation

### 1. Overfitting Risk ⚠️ HIGH
**Problem:** Agent works in backtest but fails in live trading

**Mitigation:**
- Walk-forward validation mandatory
- Never optimize on test set
- Use ensemble of strategies
- Regular retraining (monthly)
- Track live vs backtest performance divergence

### 2. Market Regime Change ⚠️ HIGH
**Problem:** Agent trained in bull market fails in bear/volatile market

**Mitigation:**
- Train on multiple market cycles (2015-2024 data)
- Regime detection system (volatility clustering)
- Automatic position reduction in high volatility
- Human override capability

### 3. Transaction Cost Slippage ⚠️ MEDIUM
**Problem:** Backtest assumes perfect execution, reality is worse

**Mitigation:**
- Conservative slippage assumptions (0.1% vs 0.04%)
- Limit order usage instead of market orders
- Trade liquid stocks only (>₹10 crore daily volume)
- Monitor implementation shortfall

### 4. Regulatory Changes ⚠️ MEDIUM
**Problem:** SEBI changes algo trading rules

**Mitigation:**
- Legal counsel on retainer
- Industry association membership
- Adaptable architecture (can switch brokers/strategies)
- Diversify across asset classes

### 5. Technology Failures ⚠️ HIGH
**Problem:** API downtime, network issues, bugs

**Mitigation:**
- Kill switches (automatic shutdown on anomalies)
- Maximum daily loss limits (hard stop at -2%)
- Redundant internet connections
- Extensive testing and simulation

---

## Realistic Capital Projections

### Self-Funded Path (₹10-50 Lakhs)
**Year 1:**
- Start with ₹10L own capital
- Target: 15-20% annual return = ₹1.5-2L profit
- Reinvest profits
- Use for proof of concept

**Year 2:**
- Scale to ₹25-50L (own capital + friends/family)
- Target: 20% return = ₹5-10L profit
- Build track record
- Apply for SEBI registration

**Year 3:**
- Raise ₹2-5 crores (Angel/HNI investors)
- Setup PMS/AIF structure
- Charge 2% management + 20% performance fee
- Revenue: ₹4-10L management + performance fees

### Funded Path (Venture Capital)
**Seed Round (₹50L - ₹2Cr):**
- 18 months runway
- Build team (2-3 quants, 1 engineer)
- Expand to 20+ strategies
- Get 100+ investors on platform

**Series A (₹5-10Cr):**
- Scale to ₹50-100Cr AUM
- Full regulatory compliance
- Marketing and distribution
- Revenue: Management fees + carried interest

---

## Team Requirements

### Immediate (Month 1-3)
- **You:** Strategy, RL development
- **Quant Researcher (Contract):** Validate models, backtest
- **DevOps (Part-time):** Infrastructure, monitoring

### Scaling (Month 4-12)
- **Senior Quant:** Multi-strategy development
- **Risk Manager:** Real-time risk monitoring
- **Compliance Officer:** SEBI regulations
- **Full-stack Engineer:** Dashboard, API
- **Operations:** Trade execution monitoring

---

## Key Performance Indicators (KPIs)

### Research KPIs
- Sharpe Ratio: >0.5 (backtest), >0.3 (live)
- Win Rate: >52%
- Max Drawdown: <15%
- Profit Factor: >1.5
- Sortino Ratio: >0.7

### Business KPIs
- Assets Under Management (AUM): ₹5Cr (Year 1), ₹50Cr (Year 3)
- Management Fee Revenue: 2% of AUM
- Performance Fee: 20% above hurdle rate (10%)
- Client Retention: >80%
- Monthly Profitability: 12 of 12 months

### Operational KPIs
- System Uptime: >99.5%
- Execution Latency: <500ms
- Compliance Violations: 0
- Risk Limit Breaches: 0

---

## Next Steps (Immediate)

**Today:**
1. ✅ Fix config.yaml position limits
2. ⏳ Run 50-episode test (start now)
3. ⏳ Analyze results in 6-12 hours

**This Week:**
4. If test passes → Full 300-episode training
5. If test fails → Debug action masking/position sizing
6. Document findings in research log

**Next Week:**
7. Walk-forward validation
8. Multi-stock performance analysis
9. Write technical whitepaper
10. Create investor presentation

---

## Funding Requirements & Timeline

### Bootstrap Phase (₹5-10 Lakhs)
- Personal capital for development
- 3-6 months to proven backtest
- No external funding needed

### Seed Phase (₹50 Lakhs - ₹2 Crores)
- After 6 months live track record
- For team building and infrastructure
- Angel investors or quant-focused VCs

### Growth Phase (₹5-10 Crores)
- After 18 months profitable live trading
- Scale to institutional AUM
- Series A from fintech VCs

---

## Success Probability Assessment

**Technical Feasibility:** 70%
- RL for trading is proven (Renaissance, Two Sigma use ML)
- Your approach is sound (PPO, action masking)
- Main risk: overfitting and regime changes

**Business Viability:** 40%
- High competition (thousands of quant funds)
- Regulatory complexity in India
- Need exceptional Sharpe (>1.0) to attract capital
- Track record takes 2-3 years minimum

**Realistic Outcome:**
- Best case (10%): Build ₹100Cr+ AUM fund, ₹2-5Cr annual revenue
- Good case (30%): Manage ₹10-20Cr, profitable boutique shop
- Base case (50%): Manage own capital profitably (₹50L-2Cr)
- Fail case (10%): Strategy doesn't work live, pivot required

---

## My Honest Recommendation

**Option 1: Conservative (Recommended)**
1. Spend next 3 months perfecting the strategy
2. Trade with ₹10-25L own capital for 12 months
3. Build verifiable track record
4. If successful → raise capital from HNIs
5. If unsuccessful → low downside (learned valuable skills)

**Option 2: Aggressive (High Risk)**
1. Raise ₹50L-1Cr seed funding now
2. Build team immediately
3. Rush to market in 6 months
4. High burn rate, pressure to perform
5. If strategy fails → wasted investor money, reputation damage

**My vote: Option 1.** Prove it works with your own capital first. No one will give you ₹2-5 crores to manage without a multi-year live track record anyway.

---

## The Hard Truth

**You are here:** Prototype with config bugs, 0 live trades

**To get funding, you need:**
- 12-24 months live profitable track record
- Sharpe >1.0 consistently
- Regulatory registrations (₹20L-50L cost)
- Full risk management infrastructure
- Clean code, documentation, disaster recovery

**Timeline to first external ₹1 crore:** 18-36 months minimum

**But:** If you can generate 15-20% annual returns consistently with Sharpe >0.5, you don't need external capital. Compounding ₹25 lakhs at 20% for 10 years = ₹1.5 crores. Add your salary and you're financially independent.

Focus on making the strategy work first. Capital follows performance.

---

**Next Immediate Action:** Let's validate the fix works. Run the 50-episode test and see if the agent trades properly now.

Ready?
