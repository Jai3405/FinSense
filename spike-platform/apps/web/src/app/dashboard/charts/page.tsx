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
} from "lucide-react";
import { useStockQuote, useStockSearch, useWatchlist } from "@/lib/api/hooks";

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

function formatCompactVolume(vol: number): string {
  if (vol >= 1_000_000) return `${(vol / 1_000_000).toFixed(1)}M`;
  if (vol >= 1_000) return `${(vol / 1_000).toFixed(1)}K`;
  return vol.toString();
}

function LoadingSkeleton() {
  return (
    <div className="h-[calc(100vh-8rem)]">
      <div className="rounded-2xl p-6" style={{ backgroundColor: "#FFFFFF", border: "1px solid #B8DDD7" }}>
        <div className="animate-pulse space-y-4">
          <div className="h-5 w-40 rounded" style={{ backgroundColor: "#F5FFFC" }} />
          <div className="h-4 w-full rounded" style={{ backgroundColor: "#F5FFFC" }} />
          <div className="h-4 w-3/4 rounded" style={{ backgroundColor: "#F5FFFC" }} />
        </div>
      </div>
    </div>
  );
}

export default function ChartsPage() {
  const [selectedSymbol, setSelectedSymbol] = useState("RELIANCE");
  const [selectedTimeframe, setSelectedTimeframe] = useState("1D");
  const [selectedChartType, setSelectedChartType] = useState(0);
  const [showIndicators, setShowIndicators] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  const { data: stockData, isLoading: quoteLoading } = useStockQuote(selectedSymbol);
  const { data: watchlistStocks, isLoading: watchlistLoading } = useWatchlist();
  const { data: searchResults } = useStockSearch(searchQuery);

  if (quoteLoading && !stockData) {
    return <LoadingSkeleton />;
  }

  const isPositive = (stockData?.change_percent ?? 0) >= 0;

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
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-7 pr-2 py-1.5 text-xs rounded-lg focus:outline-none"
                style={{ backgroundColor: colors.bgMint, color: colors.textPrimary }}
              />
            </div>
            {/* Search results dropdown */}
            {searchResults && searchResults.length > 0 && searchQuery.length >= 2 && (
              <div
                className="mt-1 rounded-lg overflow-hidden"
                style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
              >
                {searchResults.slice(0, 5).map((result) => (
                  <div
                    key={result.symbol}
                    className="px-2 py-1.5 cursor-pointer transition-colors text-xs"
                    style={{ borderBottom: `1px solid ${colors.border}` }}
                    onClick={() => {
                      setSelectedSymbol(result.symbol);
                      setSearchQuery("");
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
                    onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
                  >
                    <span className="font-medium" style={{ color: colors.textPrimary }}>{result.symbol}</span>
                    <span className="ml-1" style={{ color: colors.textMuted }}>{result.name}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
          <div className="flex-1 overflow-auto">
            {watchlistLoading ? (
              <div className="p-3">
                <div className="animate-pulse space-y-3">
                  {[1, 2, 3, 4, 5].map((i) => (
                    <div key={i} className="h-8 rounded" style={{ backgroundColor: "#F5FFFC" }} />
                  ))}
                </div>
              </div>
            ) : (
              (watchlistStocks ?? []).map((stock) => (
                <div
                  key={stock.symbol}
                  className="px-3 py-2.5 cursor-pointer transition-colors"
                  style={{
                    borderBottom: `1px solid ${colors.border}`,
                    backgroundColor: stock.symbol === selectedSymbol ? colors.bgMint : "transparent",
                  }}
                  onClick={() => setSelectedSymbol(stock.symbol)}
                  onMouseEnter={(e) => {
                    if (stock.symbol !== selectedSymbol) e.currentTarget.style.backgroundColor = colors.bgHover;
                  }}
                  onMouseLeave={(e) => {
                    if (stock.symbol !== selectedSymbol) e.currentTarget.style.backgroundColor = "transparent";
                  }}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium" style={{ color: colors.textPrimary }}>
                      {stock.symbol}
                    </span>
                    <span
                      className="text-xs font-medium"
                      style={{ color: stock.change_percent >= 0 ? colors.gain : colors.loss }}
                    >
                      {stock.change_percent >= 0 ? "+" : ""}{stock.change_percent.toFixed(2)}%
                    </span>
                  </div>
                  <span className="text-xs" style={{ color: colors.textMuted }}>
                    {"\u20B9"}{stock.price.toLocaleString("en-IN")}
                  </span>
                </div>
              ))
            )}
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
                      {stockData?.symbol ?? selectedSymbol}
                    </h1>
                    <span className="text-sm" style={{ color: colors.textMuted }}>
                      {stockData?.name ?? ""}
                    </span>
                  </div>
                  <div className="flex items-center gap-3 mt-1">
                    <span className="text-2xl font-bold" style={{ color: colors.textPrimary }}>
                      {"\u20B9"}{stockData?.price?.toLocaleString("en-IN") ?? "..."}
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
                        {isPositive ? "+" : ""}{"\u20B9"}{stockData?.change?.toFixed(2) ?? "0.00"} ({isPositive ? "+" : ""}{stockData?.change_percent?.toFixed(2) ?? "0.00"}%)
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
            {stockData ? (
              [
                { label: "Open", value: `\u20B9${stockData.open?.toLocaleString() ?? "-"}` },
                { label: "High", value: `\u20B9${stockData.high?.toLocaleString() ?? "-"}` },
                { label: "Low", value: `\u20B9${stockData.low?.toLocaleString() ?? "-"}` },
                { label: "Prev Close", value: `\u20B9${stockData.prev_close?.toLocaleString() ?? "-"}` },
                { label: "Volume", value: formatCompactVolume(stockData.volume ?? 0) },
              ].map((item) => (
                <div key={item.label} className="flex items-center justify-between">
                  <span className="text-xs" style={{ color: colors.textMuted }}>{item.label}</span>
                  <span className="text-xs font-medium" style={{ color: colors.textPrimary }}>{item.value}</span>
                </div>
              ))
            ) : (
              <div className="animate-pulse space-y-3">
                {[1, 2, 3, 4, 5].map((i) => (
                  <div key={i} className="h-4 rounded" style={{ backgroundColor: "#F5FFFC" }} />
                ))}
              </div>
            )}
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
