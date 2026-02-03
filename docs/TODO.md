# SPIKE Master TODO & Roadmap

**Last Updated:** January 27, 2026
**Status:** Active Development

---

## IMMEDIATE (This Week)

### Training
- [ ] Run Wipro PPO training (450 episodes)
  - Command: `./train_wipro.sh`
  - Monitor: `./monitor_training.sh`
- [ ] Evaluate trained Wipro model
- [ ] Compare Wipro vs Reliance performance

### Dashboard Enhancements
- [ ] Add candlestick chart to dashboard
  - Use Chart.js financial charts or lightweight-charts
  - Show OHLC data
  - Sync with WebSocket updates
- [ ] Add risk metrics panel
  - Portfolio VaR (95%, 99%)
  - CVaR (tail risk)
  - Current volatility
  - Max drawdown
- [ ] Multi-stock selector dropdown
  - Switch between WIPRO/RELIANCE in UI
  - Persist selection

### Code Quality
- [ ] Run full test suite
- [ ] Fix any failing tests
- [ ] Update config comments

---

## SHORT-TERM (February 2026)

### Regime Engine v1
- [ ] Define 7 regime states with thresholds
- [ ] Build regime classifier
  - Inputs: VIX, price action, breadth, sentiment
  - Output: regime label + confidence
- [ ] Add regime indicator to dashboard
- [ ] Implement regime-conditional reward shaping
- [ ] Test regime detection accuracy

### FinScore v1 (Basic)
- [ ] Define 9-dimension framework
- [ ] Implement individual dimension scoring:
  - [ ] Quality score (fundamentals)
  - [ ] Trend score (momentum)
  - [ ] Risk score (volatility)
  - [ ] Sentiment score (basic news)
  - [ ] Macro score (sector sensitivity)
  - [ ] Flow score (FII/DII data)
- [ ] Create fusion layer (weighted average initially)
- [ ] Build FinScore API endpoint
- [ ] Add FinScore display to dashboard

### Backtesting Lab v1
- [ ] Build backtesting engine
  - Walk-forward validation
  - Multi-metric output
  - Regime-specific performance
- [ ] Create backtest UI component
- [ ] Implement benchmark comparison (vs Nifty)
- [ ] Add stress test scenarios

### Data Pipeline
- [ ] Add more stocks to data loader
  - TCS, INFY, HDFCBANK, ICICIBANK
- [ ] Implement data caching (SQLite/Parquet)
- [ ] Add data validation checks
- [ ] Set up automated data refresh

---

## MEDIUM-TERM (March-April 2026)

### Sentiment Agent v1
- [ ] News headline scraping
  - Economic Times, Moneycontrol, LiveMint
- [ ] Basic sentiment classification
  - LLM-based or FinBERT
- [ ] Sentiment score per stock
- [ ] Sentiment drift tracking
- [ ] Integration with FinScore

### Legend Agents v1
- [ ] Define Buffett philosophy rules
  - ROIC > 15%
  - Debt/Equity < 0.5
  - Margin stability
  - Management quality signals
- [ ] Build Buffett scoring engine
- [ ] Create Legend Agent API
- [ ] Build Legend Agent UI card
- [ ] Add Peter Lynch agent
- [ ] Add Charlie Munger agent

### Strategy-GPT MVP
- [ ] Intent parsing layer (LLM)
  - Extract: goal, timeframe, risk level
  - Map to strategy parameters
- [ ] Rule generator
  - Convert intent to entry/exit rules
- [ ] Backtest integration
- [ ] Basic explanation generation
- [ ] UI for strategy creation

### Smart Themes v1
- [ ] Define 5 initial themes
  - Quality Compounders
  - Momentum Titans
  - Buffett Favorites
  - India 2030 Growth
  - Defensive Stability
- [ ] Theme construction pipeline
- [ ] Theme performance tracking
- [ ] Theme comparison UI

### Real-Time Alerts v1
- [ ] Alert types implementation
  - Price alerts
  - Sentiment alerts
  - Risk alerts
- [ ] Alert delivery (in-app)
- [ ] Alert personalization
- [ ] Alert history view

---

## LONG-TERM (May-August 2026)

### Portfolio Autopilot v1
- [ ] Portfolio analysis engine
  - Factor exposure calculation
  - Concentration risk
  - Correlation matrix
- [ ] Rebalancing logic
- [ ] Risk-based position sizing
- [ ] Regime-aware allocation
- [ ] User control levels (manual/semi/full)
- [ ] Autopilot dashboard

### Execution Intelligence
- [ ] Zerodha Kite API integration
- [ ] Order placement
- [ ] Position tracking
- [ ] Slippage measurement
- [ ] Smart timing engine

### Full FinSense (12 Agents)
- [ ] Alpha Forecasting Agent
- [ ] Risk Agent (enhanced)
- [ ] Sentiment Agent (enhanced)
- [ ] Macro Agent
- [ ] Value Agent
- [ ] Momentum Agent
- [ ] Portfolio Agent
- [ ] Regime Agent
- [ ] Behavioral Agent
- [ ] Insider Flow Agent
- [ ] Alternative Data Agent
- [ ] Simulation Agent

### B2B API Platform
- [ ] API authentication
- [ ] Rate limiting
- [ ] API documentation
- [ ] Pricing tiers
- [ ] Developer portal

### Mobile App
- [ ] React Native setup
- [ ] Core screens
  - Dashboard
  - FinScore
  - Portfolio
  - Themes
- [ ] Push notifications
- [ ] Biometric auth

---

## EPIC-LEVEL (Q4 2026 - 2027)

### Advanced Features
- [ ] Options trading support
- [ ] Multi-asset (commodities, forex)
- [ ] International markets (US, SEA)
- [ ] Social features (copy trading)
- [ ] Gamification elements

### Enterprise
- [ ] White-label solution
- [ ] Custom integrations
- [ ] Enterprise support
- [ ] SLA agreements

### Regulatory
- [ ] SEBI PMS registration
- [ ] Compliance documentation
- [ ] Audit trails
- [ ] Risk disclosures

### Scale
- [ ] Cloud deployment (AWS/GCP)
- [ ] Kubernetes orchestration
- [ ] Real-time data streaming
- [ ] ML model serving infrastructure

---

## TECHNICAL DEBT & MAINTENANCE

### Code Quality
- [ ] Increase test coverage to >80%
- [ ] Add integration tests
- [ ] Performance profiling
- [ ] Memory leak detection
- [ ] Code documentation

### Infrastructure
- [ ] CI/CD pipeline
- [ ] Automated deployments
- [ ] Monitoring & alerting
- [ ] Log aggregation
- [ ] Database backups

### Security
- [ ] Security audit
- [ ] Penetration testing
- [ ] Data encryption
- [ ] Access controls
- [ ] GDPR compliance

---

## RESEARCH & EXPERIMENTS

### ML Experiments
- [ ] Try PPO with LSTM layers
- [ ] Test Transformer architecture
- [ ] Multi-stock curriculum learning
- [ ] Reward shaping experiments
- [ ] Hyperparameter optimization (Optuna)

### Alternative Approaches
- [ ] Model-based RL
- [ ] Imitation learning
- [ ] Meta-learning for regime adaptation
- [ ] Ensemble methods

### Data Experiments
- [ ] Alternative data sources
- [ ] Satellite imagery
- [ ] Web scraping signals
- [ ] App usage data

---

## COMPLETED

### January 2026
- [x] PPO agent implementation
- [x] Trading environment with Zerodha costs
- [x] 450 episode training on RELIANCE
- [x] Glassmorphism dashboard
- [x] WebSocket real-time updates
- [x] Chart.js visualizations
- [x] Wipro training configuration
- [x] Training scripts with monitoring
- [x] KNOWLEDGE.md creation
- [x] PRD.md creation

### Earlier
- [x] Project setup
- [x] Data pipeline (yfinance)
- [x] Feature engineering (17 indicators)
- [x] Action masking
- [x] Basic dashboard

---

## PRIORITY MATRIX

### P0 (Do Now)
- Wipro training
- Regime Engine v1
- FinScore v1

### P1 (Do Next)
- Legend Agents
- Strategy-GPT MVP
- Portfolio Autopilot

### P2 (Important)
- Smart Themes
- Backtesting Lab
- Alerts Engine

### P3 (Nice to Have)
- Execution Intelligence
- Mobile App
- Social Features

---

## NOTES

- Always update KNOWLEDGE.md after major changes
- Test on paper trading before any live integration
- Document all API changes
- Keep PRD.md in sync with actual development

---

*This TODO is the single source of truth for what needs to be built. Update daily.*
