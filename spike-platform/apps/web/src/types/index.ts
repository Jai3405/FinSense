// ==========================================
// SPIKE Platform Type Definitions
// ==========================================

// ----- User & Auth -----
export interface User {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  imageUrl?: string;
  plan: "free" | "pro" | "pro_plus" | "premium";
  riskProfile?: RiskProfile;
  createdAt: Date;
  updatedAt: Date;
}

export interface RiskProfile {
  score: number; // 1-10
  category: "conservative" | "moderate" | "aggressive" | "very_aggressive";
  investmentHorizon: "short" | "medium" | "long";
  monthlyIncome?: number;
  existingInvestments?: number;
  financialGoals: string[];
}

// ----- Stock & Market Data -----
export interface Stock {
  symbol: string;
  name: string;
  exchange: "NSE" | "BSE";
  sector: string;
  industry: string;
  marketCap: number;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  avgVolume: number;
  high52w: number;
  low52w: number;
  pe?: number;
  pb?: number;
  dividendYield?: number;
  finScore: FinScore;
}

export interface OHLCV {
  timestamp: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface MarketIndex {
  symbol: string;
  name: string;
  value: number;
  change: number;
  changePercent: number;
  high: number;
  low: number;
}

// ----- FinScore -----
export interface FinScore {
  overall: number; // 0-10
  components: FinScoreComponents;
  regime: MarketRegime;
  confidence: number; // 0-100
  signal: "strong_buy" | "buy" | "hold" | "sell" | "strong_sell";
  updatedAt: Date;
}

export interface FinScoreComponents {
  quality: number; // Fundamentals
  momentum: number; // Price momentum
  value: number; // Valuation
  sentiment: number; // News/social
  risk: number; // Volatility/drawdown
  flow: number; // Institutional flows
  regimeFit: number; // How well it fits current regime
  sectorScore: number; // Sector dynamics
  technical: number; // Technical signals
}

// ----- Market Regime -----
export type MarketRegime =
  | "bull_strong"
  | "bull_weak"
  | "bear_strong"
  | "bear_weak"
  | "sideways"
  | "volatile"
  | "recovery";

export interface RegimeAnalysis {
  current: MarketRegime;
  confidence: number;
  trend: "up" | "down" | "sideways";
  duration: number; // Days in current regime
  signals: string[];
  recommendation: string;
}

// ----- Portfolio -----
export interface Portfolio {
  id: string;
  userId: string;
  name: string;
  holdings: Holding[];
  totalValue: number;
  invested: number;
  returns: number;
  returnsPercent: number;
  todayChange: number;
  todayChangePercent: number;
  allocation: SectorAllocation[];
  riskMetrics: RiskMetrics;
  createdAt: Date;
  updatedAt: Date;
}

export interface Holding {
  id: string;
  symbol: string;
  name: string;
  quantity: number;
  avgPrice: number;
  currentPrice: number;
  invested: number;
  currentValue: number;
  returns: number;
  returnsPercent: number;
  allocation: number;
  finScore: number;
}

export interface SectorAllocation {
  sector: string;
  allocation: number;
  value: number;
}

export interface RiskMetrics {
  volatility: number;
  sharpeRatio: number;
  maxDrawdown: number;
  beta: number;
  var95: number; // Value at Risk 95%
  concentrationRisk: number;
}

// ----- Watchlist -----
export interface Watchlist {
  id: string;
  userId: string;
  name: string;
  stocks: WatchlistStock[];
  createdAt: Date;
  updatedAt: Date;
}

export interface WatchlistStock {
  symbol: string;
  addedAt: Date;
  alertPrice?: number;
  notes?: string;
}

// ----- AI Insights -----
export interface AIInsight {
  id: string;
  type: "opportunity" | "warning" | "trend" | "idea" | "rebalance";
  priority: "high" | "medium" | "low";
  title: string;
  description: string;
  action?: string;
  actionUrl?: string;
  relatedSymbols?: string[];
  createdAt: Date;
  expiresAt?: Date;
  dismissed: boolean;
}

// ----- Strategy -----
export interface Strategy {
  id: string;
  userId: string;
  name: string;
  description: string;
  rules: StrategyRule[];
  backtest?: BacktestResult;
  status: "draft" | "backtesting" | "active" | "paused";
  createdAt: Date;
  updatedAt: Date;
}

export interface StrategyRule {
  id: string;
  type: "entry" | "exit" | "filter";
  indicator: string;
  condition: string;
  value: number | string;
  logic: "and" | "or";
}

export interface BacktestResult {
  startDate: Date;
  endDate: Date;
  totalReturn: number;
  annualizedReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  totalTrades: number;
  avgHoldingPeriod: number;
  benchmarkReturn: number;
  alpha: number;
  equity: { date: Date; value: number }[];
}

// ----- Themes -----
export interface Theme {
  id: string;
  name: string;
  description: string;
  category: string;
  stocks: ThemeStock[];
  performance: {
    returns1m: number;
    returns3m: number;
    returns1y: number;
  };
  riskLevel: "low" | "medium" | "high";
  minInvestment: number;
  finScore: number;
  popularity: number;
  createdBy: "spike" | "user";
  createdAt: Date;
}

export interface ThemeStock {
  symbol: string;
  name: string;
  allocation: number;
  finScore: number;
}

// ----- Legend Agent -----
export interface LegendAgent {
  id: string;
  name: string;
  avatar: string;
  philosophy: string;
  style: string;
  metrics: string[];
  prompt: string;
}

export interface LegendAnalysis {
  agentId: string;
  symbol: string;
  verdict: "buy" | "hold" | "sell" | "avoid";
  confidence: number;
  reasoning: string;
  keyMetrics: { name: string; value: string; assessment: string }[];
  risks: string[];
  createdAt: Date;
}

// ----- API Response -----
export interface APIResponse<T> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
  };
  meta?: {
    page?: number;
    limit?: number;
    total?: number;
  };
}

// ----- Pagination -----
export interface PaginationParams {
  page: number;
  limit: number;
  sortBy?: string;
  sortOrder?: "asc" | "desc";
}

// ----- Filters -----
export interface StockFilters {
  sector?: string[];
  marketCap?: "small" | "mid" | "large";
  finScoreMin?: number;
  finScoreMax?: number;
  peMin?: number;
  peMax?: number;
  priceMin?: number;
  priceMax?: number;
}
