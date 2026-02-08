"use client";

import { useState } from "react";
import {
  Search,
  Filter,
  ChevronDown,
  X,
  Target,
  TrendingUp,
  TrendingDown,
  Save,
  Play,
  RotateCcw,
} from "lucide-react";

// Color constants
const colors = {
  bg: "#FFFFFF",
  bgHover: "#F0FBF9",
  bgMint: "#F5FFFC",
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

interface FilterOption {
  id: string;
  label: string;
  type: "range" | "select" | "multiselect";
  options?: string[];
  min?: number;
  max?: number;
  unit?: string;
}

interface Stock {
  symbol: string;
  name: string;
  sector: string;
  price: number;
  change: number;
  marketCap: number;
  pe: number;
  roe: number;
  debtEquity: number;
  finScore: number;
}

const filterGroups = [
  {
    name: "Fundamentals",
    filters: [
      { id: "pe", label: "P/E Ratio", type: "range" as const, min: 0, max: 100 },
      { id: "pb", label: "P/B Ratio", type: "range" as const, min: 0, max: 20 },
      { id: "roe", label: "ROE %", type: "range" as const, min: 0, max: 50, unit: "%" },
      { id: "roce", label: "ROCE %", type: "range" as const, min: 0, max: 50, unit: "%" },
    ],
  },
  {
    name: "Valuation",
    filters: [
      { id: "marketcap", label: "Market Cap", type: "select" as const, options: ["Large Cap", "Mid Cap", "Small Cap"] },
      { id: "dividend", label: "Dividend Yield", type: "range" as const, min: 0, max: 10, unit: "%" },
      { id: "eps", label: "EPS Growth", type: "range" as const, min: -50, max: 100, unit: "%" },
    ],
  },
  {
    name: "Technical",
    filters: [
      { id: "rsi", label: "RSI", type: "range" as const, min: 0, max: 100 },
      { id: "above50dma", label: "Above 50 DMA", type: "select" as const, options: ["Yes", "No"] },
      { id: "above200dma", label: "Above 200 DMA", type: "select" as const, options: ["Yes", "No"] },
    ],
  },
  {
    name: "Sector",
    filters: [
      { id: "sector", label: "Sector", type: "multiselect" as const, options: ["IT", "Banking", "Pharma", "Auto", "FMCG", "Energy", "Metals", "Realty"] },
    ],
  },
];

const presetScreeners = [
  { name: "High Growth", description: "EPS growth > 20%, ROE > 15%", count: 45 },
  { name: "Value Picks", description: "P/E < 15, Dividend > 2%", count: 32 },
  { name: "Quality Stocks", description: "ROCE > 20%, Debt/Equity < 0.5", count: 28 },
  { name: "Momentum", description: "RSI 50-70, Above 200 DMA", count: 52 },
];

const mockResults: Stock[] = [
  { symbol: "HDFCBANK", name: "HDFC Bank Ltd", sector: "Banking", price: 1685.30, change: 1.2, marketCap: 1280000, pe: 18.5, roe: 16.8, debtEquity: 0.0, finScore: 92 },
  { symbol: "TCS", name: "Tata Consultancy Services", sector: "IT", price: 3892.50, change: -0.5, marketCap: 1420000, pe: 28.2, roe: 42.5, debtEquity: 0.1, finScore: 89 },
  { symbol: "RELIANCE", name: "Reliance Industries", sector: "Energy", price: 2545.80, change: 0.8, marketCap: 1720000, pe: 24.8, roe: 9.2, debtEquity: 0.4, finScore: 85 },
  { symbol: "INFY", name: "Infosys Ltd", sector: "IT", price: 1565.25, change: 1.5, marketCap: 650000, pe: 25.1, roe: 31.2, debtEquity: 0.1, finScore: 87 },
  { symbol: "ICICIBANK", name: "ICICI Bank Ltd", sector: "Banking", price: 1012.40, change: -0.3, marketCap: 710000, pe: 16.2, roe: 15.4, debtEquity: 0.0, finScore: 84 },
  { symbol: "BHARTIARTL", name: "Bharti Airtel Ltd", sector: "Telecom", price: 1185.90, change: 2.1, marketCap: 680000, pe: 45.2, roe: 12.8, debtEquity: 1.2, finScore: 78 },
  { symbol: "SUNPHARMA", name: "Sun Pharma Industries", sector: "Pharma", price: 1245.60, change: 0.4, marketCap: 298000, pe: 32.5, roe: 14.2, debtEquity: 0.2, finScore: 76 },
  { symbol: "TATAMOTORS", name: "Tata Motors Ltd", sector: "Auto", price: 742.35, change: 2.4, marketCap: 272000, pe: 8.5, roe: 28.5, debtEquity: 0.8, finScore: 74 },
];

function formatMarketCap(value: number): string {
  if (value >= 100000) return `₹${(value / 100000).toFixed(1)}L Cr`;
  return `₹${(value / 1000).toFixed(0)}K Cr`;
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 2,
  }).format(value);
}

function getScoreColor(score: number): string {
  if (score >= 80) return colors.gain;
  if (score >= 60) return "#F5A623";
  return colors.loss;
}

export default function ScreenerPage() {
  const [activeFilters, setActiveFilters] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState("");

  return (
    <div className="max-w-6xl">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-xl font-semibold mb-1" style={{ color: colors.textPrimary }}>
          Stock Screener
        </h1>
        <p className="text-sm" style={{ color: colors.textMuted }}>
          Filter and discover stocks based on fundamental and technical criteria
        </p>
      </div>

      {/* Preset Screeners */}
      <div className="mb-6">
        <p className="text-xs font-medium uppercase tracking-wide mb-3" style={{ color: colors.textMuted }}>
          Quick Screens
        </p>
        <div className="flex gap-3">
          {presetScreeners.map((preset) => (
            <button
              key={preset.name}
              className="px-4 py-2.5 rounded-lg text-left transition-colors"
              style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
              onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = colors.bg)}
            >
              <p className="text-sm font-medium" style={{ color: colors.textPrimary }}>
                {preset.name}
              </p>
              <p className="text-xs" style={{ color: colors.textMuted }}>
                {preset.description}
              </p>
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-4 gap-6">
        {/* Filters Panel */}
        <div
          className="rounded-xl p-4"
          style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
        >
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold" style={{ color: colors.textPrimary }}>
              Filters
            </h2>
            <button
              className="flex items-center gap-1 text-xs"
              style={{ color: colors.accent }}
            >
              <RotateCcw className="w-3 h-3" />
              Reset
            </button>
          </div>

          {filterGroups.map((group) => (
            <div key={group.name} className="mb-4">
              <p
                className="text-xs font-medium uppercase tracking-wide mb-2"
                style={{ color: colors.textMuted }}
              >
                {group.name}
              </p>
              <div className="space-y-2">
                {group.filters.map((filter) => (
                  <button
                    key={filter.id}
                    className="w-full flex items-center justify-between px-3 py-2 rounded-lg text-sm transition-colors"
                    style={{ backgroundColor: colors.bgMint }}
                    onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
                    onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = colors.bgMint)}
                  >
                    <span style={{ color: colors.textSecondary }}>{filter.label}</span>
                    <ChevronDown className="w-4 h-4" style={{ color: colors.textMuted }} />
                  </button>
                ))}
              </div>
            </div>
          ))}

          <div className="flex gap-2 mt-6">
            <button
              className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-medium text-white"
              style={{ backgroundColor: colors.accent }}
            >
              <Play className="w-4 h-4" />
              Run Screen
            </button>
            <button
              className="px-3 py-2.5 rounded-lg transition-colors"
              style={{ border: `1px solid ${colors.border}`, color: colors.textSecondary }}
            >
              <Save className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Results */}
        <div className="col-span-3">
          {/* Search & Active Filters */}
          <div className="flex items-center gap-3 mb-4">
            <div className="relative flex-1">
              <Search
                className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4"
                style={{ color: colors.textMuted }}
              />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search results..."
                className="w-full pl-10 pr-4 py-2 text-sm rounded-lg focus:outline-none"
                style={{
                  backgroundColor: colors.inputBg,
                  border: `1px solid ${colors.border}`,
                  color: colors.textPrimary,
                }}
              />
            </div>
            <span className="text-sm" style={{ color: colors.textMuted }}>
              {mockResults.length} stocks found
            </span>
          </div>

          {/* Results Table */}
          <div
            className="rounded-xl overflow-hidden"
            style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
          >
            <table className="w-full">
              <thead>
                <tr style={{ backgroundColor: colors.bgMint, borderBottom: `1px solid ${colors.border}` }}>
                  <th className="text-left text-xs font-medium uppercase tracking-wide px-4 py-3" style={{ color: colors.textMuted }}>Stock</th>
                  <th className="text-right text-xs font-medium uppercase tracking-wide px-4 py-3" style={{ color: colors.textMuted }}>Price</th>
                  <th className="text-right text-xs font-medium uppercase tracking-wide px-4 py-3" style={{ color: colors.textMuted }}>Change</th>
                  <th className="text-right text-xs font-medium uppercase tracking-wide px-4 py-3" style={{ color: colors.textMuted }}>Market Cap</th>
                  <th className="text-right text-xs font-medium uppercase tracking-wide px-4 py-3" style={{ color: colors.textMuted }}>P/E</th>
                  <th className="text-right text-xs font-medium uppercase tracking-wide px-4 py-3" style={{ color: colors.textMuted }}>ROE</th>
                  <th className="text-right text-xs font-medium uppercase tracking-wide px-4 py-3" style={{ color: colors.textMuted }}>D/E</th>
                  <th className="text-center text-xs font-medium uppercase tracking-wide px-4 py-3" style={{ color: colors.textMuted }}>FinScore</th>
                </tr>
              </thead>
              <tbody>
                {mockResults.map((stock, index) => {
                  const isPositive = stock.change >= 0;
                  return (
                    <tr
                      key={stock.symbol}
                      className="cursor-pointer transition-colors"
                      style={{ borderBottom: `1px solid ${colors.border}` }}
                      onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
                      onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
                    >
                      <td className="px-4 py-3">
                        <p className="text-sm font-medium" style={{ color: colors.textPrimary }}>{stock.symbol}</p>
                        <p className="text-xs" style={{ color: colors.textMuted }}>{stock.sector}</p>
                      </td>
                      <td className="px-4 py-3 text-right text-sm font-medium" style={{ color: colors.textPrimary }}>
                        {formatCurrency(stock.price)}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span
                          className="text-sm font-medium"
                          style={{ color: isPositive ? colors.gain : colors.loss }}
                        >
                          {isPositive ? "+" : ""}{stock.change.toFixed(2)}%
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right text-sm" style={{ color: colors.textSecondary }}>
                        {formatMarketCap(stock.marketCap)}
                      </td>
                      <td className="px-4 py-3 text-right text-sm" style={{ color: colors.textSecondary }}>
                        {stock.pe.toFixed(1)}
                      </td>
                      <td className="px-4 py-3 text-right text-sm" style={{ color: colors.textSecondary }}>
                        {stock.roe.toFixed(1)}%
                      </td>
                      <td className="px-4 py-3 text-right text-sm" style={{ color: colors.textSecondary }}>
                        {stock.debtEquity.toFixed(1)}
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center justify-center gap-1">
                          <Target className="w-4 h-4" style={{ color: getScoreColor(stock.finScore) }} />
                          <span className="text-sm font-medium" style={{ color: getScoreColor(stock.finScore) }}>
                            {stock.finScore}
                          </span>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
