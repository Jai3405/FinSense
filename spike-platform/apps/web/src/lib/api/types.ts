export interface StockQuote {
  symbol: string;
  name: string;
  exchange: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
  high: number;
  low: number;
  open: number;
  prev_close: number;
  timestamp: string;
}

export interface StockSearchResult {
  symbol: string;
  name: string;
  exchange: string;
  sector: string;
}

export interface StockInfo {
  symbol: string;
  name: string;
  exchange: string;
  sector: string;
  industry: string;
  market_cap: number;
  pe_ratio: number | null;
  pb_ratio: number | null;
  dividend_yield: number | null;
  eps: number | null;
  high_52w: number;
  low_52w: number;
  avg_volume: number;
  description: string | null;
}

export interface OHLCV {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface StockHistory {
  symbol: string;
  period: string;
  interval: string;
  data: OHLCV[];
}

export interface TrendingStock {
  symbol: string;
  name: string;
  price: number;
  change_percent: number;
  volume: string;
  trend_score: number;
}

export interface MarketIndex {
  symbol: string;
  name: string;
  value: number;
  change: number;
  change_percent: number;
  high: number;
  low: number;
  volume?: number;
}

export interface SectorPerformance {
  name: string;
  change_percent: number;
  top_gainer: string;
  top_loser: string;
}

export interface MarketStatus {
  is_open: boolean;
  current_time: string;
  next_open: string;
  next_close: string;
}

export interface MarketRegime {
  regime: string;
  confidence: number;
  description: string;
}

export interface MarketBreadth {
  advances: number;
  declines: number;
  unchanged: number;
  advance_decline_ratio: number;
  new_highs: number;
  new_lows: number;
}

export interface PortfolioSummary {
  total_value: number;
  invested: number;
  returns: number;
  returns_percent: number;
  today_change: number;
  today_change_percent: number;
  holding_count: number;
}

export interface Holding {
  symbol: string;
  name: string;
  quantity: number;
  avg_price: number;
  current_price: number;
  invested: number;
  current_value: number;
  returns: number;
  returns_percent: number;
  allocation: number;
  finscore: number;
}

export interface WatchlistStock {
  symbol: string;
  name: string;
  price: number;
  change: number;
  change_percent: number;
  finscore: number;
  added_at: string;
}

export interface AddHoldingRequest {
  symbol: string;
  quantity: number;
  avg_price: number;
}

export interface AddToWatchlistRequest {
  symbol: string;
  alert_price?: number;
  notes?: string;
}
