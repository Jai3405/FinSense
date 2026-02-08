"use client";

import { useState } from "react";
import {
  Search,
  Plus,
  Minus,
  Maximize2,
  Settings2,
  TrendingUp,
  TrendingDown,
  BarChart2,
  CandlestickChart,
  LineChart,
  Activity,
  Layers,
  Bell,
  Share2,
  Download,
  Clock,
} from "lucide-react";

// Color constants
const colors = {
  bg: "#FFFFFF",
  bgHover: "#F0FBF9",
  bgMint: "#F5FFFC",
  bgDark: "#0D3331",
  border: "#B8DDD7",
  textPrimary: "#0D3331",
  textSecondary: "#3D6B66",
  textMuted: "#6B9B94",
  accent: "#00897B",
  gain: "#00B386",
  gainBg: "#E6F9F4",
  loss: "#F45B69",
  lossBg: "#FEF0F1",
  inputBg: "#F8FFFE",
};

const timeframes = ["1D", "1W", "1M", "3M", "6M", "1Y", "5Y", "MAX"];
const chartTypes = [
  { icon: CandlestickChart, name: "Candle" },
  { icon: LineChart, name: "Line" },
  { icon: BarChart2, name: "Bar" },
  { icon: Activity, name: "Area" },
];

const indicators = [
  { name: "SMA", description: "Simple Moving Average", active: true },
  { name: "EMA", description: "Exponential Moving Average", active: true },
  { name: "RSI", description: "Relative Strength Index", active: false },
  { name: "MACD", description: "Moving Average Convergence Divergence", active: false },
  { name: "BB", description: "Bollinger Bands", active: false },
  { name: "VWAP", description: "Volume Weighted Average Price", active: false },
];

const watchlistStocks = [
  { symbol: "RELIANCE", price: 2545.80, change: 0.8 },
  { symbol: "TCS", price: 3892.50, change: -0.5 },
  { symbol: "HDFCBANK", price: 1685.30, change: 1.2 },
  { symbol: "INFY", price: 1565.25, change: 1.5 },
  { symbol: "ICICIBANK", price: 1012.40, change: -0.3 },
];

const stockData = {
  symbol: "RELIANCE",
  name: "Reliance Industries Ltd",
  price: 2545.80,
  change: 20.45,
  changePercent: 0.81,
  open: 2530.00,
  high: 2558.90,
  low: 2525.15,
  prevClose: 2525.35,
  volume: "12.5M",
  avgVolume: "10.2M",
  marketCap: "₹17.2L Cr",
  pe: 24.8,
  weekHigh52: 2856.15,
  weekLow52: 2180.00,
};

export default function ChartsPage() {
  const [selectedTimeframe, setSelectedTimeframe] = useState("1D");
  const [selectedChartType, setSelectedChartType] = useState(0);
  const [showIndicators, setShowIndicators] = useState(false);

  const isPositive = stockData.changePercent >= 0;

  return (
    <div className="h-[calc(100vh-8rem)]">
      <div className="grid grid-cols-6 gap-4 h-full">
        {/* Watchlist Sidebar */}
        <div
          className="rounded-xl overflow-hidden flex flex-col"
          style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
        >
          <div className="px-3 py-2.5" style={{ borderBottom: `1px solid ${colors.border}` }}>
            <div className="relative">
              <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5" style={{ color: colors.textMuted }} />
              <input
                type="text"
                placeholder="Search..."
                className="w-full pl-7 pr-2 py-1.5 text-xs rounded-lg focus:outline-none"
                style={{ backgroundColor: colors.bgMint, color: colors.textPrimary }}
              />
            </div>
          </div>
          <div className="flex-1 overflow-auto">
            {watchlistStocks.map((stock) => (
              <div
                key={stock.symbol}
                className="px-3 py-2.5 cursor-pointer transition-colors"
                style={{ borderBottom: `1px solid ${colors.border}` }}
                onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
                onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium" style={{ color: colors.textPrimary }}>
                    {stock.symbol}
                  </span>
                  <span
                    className="text-xs font-medium"
                    style={{ color: stock.change >= 0 ? colors.gain : colors.loss }}
                  >
                    {stock.change >= 0 ? "+" : ""}{stock.change.toFixed(2)}%
                  </span>
                </div>
                <span className="text-xs" style={{ color: colors.textMuted }}>
                  ₹{stock.price.toLocaleString("en-IN")}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Main Chart Area */}
        <div className="col-span-4 flex flex-col gap-4">
          {/* Stock Header */}
          <div
            className="rounded-xl p-4"
            style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div>
                  <div className="flex items-center gap-2">
                    <h1 className="text-lg font-semibold" style={{ color: colors.textPrimary }}>
                      {stockData.symbol}
                    </h1>
                    <span className="text-sm" style={{ color: colors.textMuted }}>
                      {stockData.name}
                    </span>
                  </div>
                  <div className="flex items-center gap-3 mt-1">
                    <span className="text-2xl font-bold" style={{ color: colors.textPrimary }}>
                      ₹{stockData.price.toLocaleString("en-IN")}
                    </span>
                    <div className="flex items-center gap-1">
                      {isPositive ? (
                        <TrendingUp className="w-4 h-4" style={{ color: colors.gain }} />
                      ) : (
                        <TrendingDown className="w-4 h-4" style={{ color: colors.loss }} />
                      )}
                      <span
                        className="text-sm font-medium"
                        style={{ color: isPositive ? colors.gain : colors.loss }}
                      >
                        {isPositive ? "+" : ""}₹{stockData.change.toFixed(2)} ({isPositive ? "+" : ""}{stockData.changePercent.toFixed(2)}%)
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  className="p-2 rounded-lg transition-colors"
                  style={{ color: colors.textMuted }}
                  onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
                  onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
                >
                  <Bell className="w-4 h-4" />
                </button>
                <button
                  className="p-2 rounded-lg transition-colors"
                  style={{ color: colors.textMuted }}
                  onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
                  onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
                >
                  <Share2 className="w-4 h-4" />
                </button>
                <button
                  className="p-2 rounded-lg transition-colors"
                  style={{ color: colors.textMuted }}
                  onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
                  onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
                >
                  <Download className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>

          {/* Chart Controls */}
          <div
            className="rounded-xl p-3 flex items-center justify-between"
            style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
          >
            <div className="flex items-center gap-4">
              {/* Timeframes */}
              <div className="flex items-center gap-1">
                {timeframes.map((tf) => (
                  <button
                    key={tf}
                    onClick={() => setSelectedTimeframe(tf)}
                    className="px-2.5 py-1 text-xs font-medium rounded-md transition-colors"
                    style={{
                      backgroundColor: selectedTimeframe === tf ? colors.accent : "transparent",
                      color: selectedTimeframe === tf ? "#FFFFFF" : colors.textMuted,
                    }}
                  >
                    {tf}
                  </button>
                ))}
              </div>

              <div className="w-px h-5" style={{ backgroundColor: colors.border }} />

              {/* Chart Types */}
              <div className="flex items-center gap-1">
                {chartTypes.map((type, index) => {
                  const Icon = type.icon;
                  return (
                    <button
                      key={type.name}
                      onClick={() => setSelectedChartType(index)}
                      className="p-1.5 rounded-md transition-colors"
                      style={{
                        backgroundColor: selectedChartType === index ? colors.bgMint : "transparent",
                        color: selectedChartType === index ? colors.accent : colors.textMuted,
                      }}
                    >
                      <Icon className="w-4 h-4" />
                    </button>
                  );
                })}
              </div>

              <div className="w-px h-5" style={{ backgroundColor: colors.border }} />

              {/* Indicators */}
              <button
                onClick={() => setShowIndicators(!showIndicators)}
                className="flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-md transition-colors"
                style={{
                  backgroundColor: showIndicators ? colors.bgMint : "transparent",
                  color: showIndicators ? colors.accent : colors.textMuted,
                }}
              >
                <Layers className="w-4 h-4" />
                Indicators
              </button>
            </div>

            <div className="flex items-center gap-2">
              <button className="p-1.5 rounded-md" style={{ color: colors.textMuted }}>
                <Minus className="w-4 h-4" />
              </button>
              <button className="p-1.5 rounded-md" style={{ color: colors.textMuted }}>
                <Plus className="w-4 h-4" />
              </button>
              <button className="p-1.5 rounded-md" style={{ color: colors.textMuted }}>
                <Maximize2 className="w-4 h-4" />
              </button>
              <button className="p-1.5 rounded-md" style={{ color: colors.textMuted }}>
                <Settings2 className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Chart Canvas */}
          <div
            className="flex-1 rounded-xl flex items-center justify-center"
            style={{ backgroundColor: colors.bgDark, border: `1px solid ${colors.border}` }}
          >
            <div className="text-center">
              <CandlestickChart className="w-16 h-16 mx-auto mb-4" style={{ color: colors.accent }} />
              <p className="text-sm" style={{ color: colors.textMuted }}>
                TradingView Chart Integration
              </p>
              <p className="text-xs mt-1" style={{ color: colors.textMuted }}>
                Interactive candlestick chart with technical indicators
              </p>
            </div>
          </div>
        </div>

        {/* Right Panel - Stock Info */}
        <div
          className="rounded-xl overflow-hidden flex flex-col"
          style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
        >
          <div className="px-3 py-2.5" style={{ borderBottom: `1px solid ${colors.border}` }}>
            <h3 className="text-xs font-semibold" style={{ color: colors.textPrimary }}>
              Stock Info
            </h3>
          </div>
          <div className="flex-1 overflow-auto p-3 space-y-3">
            {[
              { label: "Open", value: `₹${stockData.open.toLocaleString()}` },
              { label: "High", value: `₹${stockData.high.toLocaleString()}` },
              { label: "Low", value: `₹${stockData.low.toLocaleString()}` },
              { label: "Prev Close", value: `₹${stockData.prevClose.toLocaleString()}` },
              { label: "Volume", value: stockData.volume },
              { label: "Avg Volume", value: stockData.avgVolume },
              { label: "Market Cap", value: stockData.marketCap },
              { label: "P/E", value: stockData.pe.toString() },
              { label: "52W High", value: `₹${stockData.weekHigh52.toLocaleString()}` },
              { label: "52W Low", value: `₹${stockData.weekLow52.toLocaleString()}` },
            ].map((item) => (
              <div key={item.label} className="flex items-center justify-between">
                <span className="text-xs" style={{ color: colors.textMuted }}>{item.label}</span>
                <span className="text-xs font-medium" style={{ color: colors.textPrimary }}>{item.value}</span>
              </div>
            ))}
          </div>

          {/* Indicators Panel */}
          {showIndicators && (
            <div style={{ borderTop: `1px solid ${colors.border}` }}>
              <div className="px-3 py-2.5" style={{ borderBottom: `1px solid ${colors.border}` }}>
                <h3 className="text-xs font-semibold" style={{ color: colors.textPrimary }}>
                  Indicators
                </h3>
              </div>
              <div className="p-2 space-y-1">
                {indicators.map((indicator) => (
                  <button
                    key={indicator.name}
                    className="w-full flex items-center justify-between px-2 py-1.5 rounded-lg text-xs transition-colors"
                    style={{
                      backgroundColor: indicator.active ? colors.gainBg : "transparent",
                    }}
                  >
                    <span style={{ color: indicator.active ? colors.gain : colors.textSecondary }}>
                      {indicator.name}
                    </span>
                    <span style={{ color: colors.textMuted }}>{indicator.active ? "On" : "Off"}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
